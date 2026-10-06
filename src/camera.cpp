#include "camera.hpp"

#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/formats.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sys/mman.h>
#include <vector>

using namespace libcamera;

struct Camera::Impl {
    std::unique_ptr<CameraManager> manager;
    std::shared_ptr<libcamera::Camera> camera;
    std::unique_ptr<FrameBufferAllocator> allocator;

    std::unique_ptr<CameraConfiguration> configuration;
    Stream* stream = nullptr;

    std::vector<std::unique_ptr<Request>> requests;

    struct MappedBuffer {
        void* data = MAP_FAILED;
        size_t length = 0;
    };

    std::map<FrameBuffer*, MappedBuffer> mapped;

    std::mutex mutex;
    std::condition_variable condition;
    std::queue<cv::Mat> frames;

    bool running = false;
    bool stopping = false;

    void complete(Request* request)
    {
        if (request->status() == Request::RequestCancelled)
            return;

        FrameBuffer* buffer = request->findBuffer(stream);

        if (!buffer)
            return;

        const FrameMetadata& metadata = buffer->metadata();

        if (metadata.status != FrameMetadata::FrameSuccess)
            return;

        auto it = mapped.find(buffer);

        if (it == mapped.end())
            return;

        const auto& mapping = it->second;

        if (mapping.data == MAP_FAILED)
            return;

        cv::Mat frame(
            configuration->size.height,
            configuration->size.width,
            CV_8UC3,
            mapping.data,
            configuration->stride);

        {
            std::lock_guard lock(mutex);

            if (frames.size() >= 2)
                frames.pop();

            frames.push(frame.clone());
        }

        condition.notify_one();

        if (!stopping) {
            request->reuse(Request::ReuseBuffers);
            camera->queueRequest(request);
        }
    }
};

Camera::Camera()
    : impl_(std::make_unique<Impl>())
{
}

Camera::~Camera()
{
    stop();
}

bool Camera::start()
{
    auto& p = *impl_;

    if (p.running)
        return true;

    p.manager = std::make_unique<CameraManager>();

    if (p.manager->start()) {
        std::cerr << "[camera] Failed to start CameraManager\n";
        p.manager.reset();
        return false;
    }

    if (p.manager->cameras().empty()) {
        std::cerr << "[camera] No cameras found\n";
        p.manager->stop();
        p.manager.reset();
        return false;
    }

    p.camera = p.manager->cameras()[0];

    if (p.camera->acquire()) {
        std::cerr << "[camera] Failed to acquire camera\n";
        p.camera.reset();
        p.manager->stop();
        p.manager.reset();
        return false;
    }

    p.configuration =
        p.camera->generateConfiguration(
            {StreamRole::ViewFinder});

    if (!p.configuration) {
        std::cerr
            << "[camera] Failed to generate configuration\n";

        p.camera->release();
        p.camera.reset();
        p.manager->stop();
        p.manager.reset();

        return false;
    }

    auto& cfg = p.configuration->at(0);

    cfg.pixelFormat = formats::RGB888;
    cfg.size = {1920, 1080};
    cfg.bufferCount = 4;

    const auto status =
        p.configuration->validate();

    if (status == CameraConfiguration::Invalid) {
        std::cerr
            << "[camera] Invalid configuration\n";

        p.configuration.reset();
        p.camera->release();
        p.camera.reset();
        p.manager->stop();
        p.manager.reset();

        return false;
    }

    if (p.camera->configure(p.configuration.get())) {
        std::cerr
            << "[camera] Failed to configure camera\n";

        p.configuration.reset();
        p.camera->release();
        p.camera.reset();
        p.manager->stop();
        p.manager.reset();

        return false;
    }

    p.stream = cfg.stream();

    p.allocator =
        std::make_unique<FrameBufferAllocator>(
            p.camera);

    if (p.allocator->allocate(p.stream) < 0) {
        std::cerr
            << "[camera] Failed to allocate buffers\n";

        p.allocator.reset();
        p.configuration.reset();
        p.camera->release();
        p.camera.reset();
        p.manager->stop();
        p.manager.reset();

        return false;
    }

    const auto& buffers =
        p.allocator->buffers(p.stream);

    for (const auto& buffer : buffers) {
        const auto& planes = buffer->planes();

        if (planes.empty())
            continue;

        const auto& plane = planes[0];

        const size_t length = plane.length;

        void* memory = mmap(
            nullptr,
            length,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            plane.fd.get(),
            0);

        if (memory == MAP_FAILED) {
            std::cerr
                << "[camera] Failed to map framebuffer\n";

            continue;
        }

        p.mapped.emplace(
            buffer.get(),
            Impl::MappedBuffer{
                memory,
                length
            });
    }

    for (const auto& buffer : buffers) {
        auto request =
            p.camera->createRequest();

        if (!request) {
            std::cerr
                << "[camera] Failed to create request\n";

            stop();
            return false;
        }

        if (request->addBuffer(
                p.stream,
                buffer.get())) {
            std::cerr
                << "[camera] Failed to add buffer\n";

            stop();
            return false;
        }

        p.requests.push_back(
            std::move(request));
    }

    p.camera->requestCompleted.connect(
        &p,
        [&](Request* request) {
            p.complete(request);
        });

    if (p.camera->start()) {
        std::cerr
            << "[camera] Failed to start capture\n";

        stop();
        return false;
    }

    p.running = true;

    for (auto& request : p.requests) {
        if (p.camera->queueRequest(
                request.get())) {
            std::cerr
                << "[camera] Failed to queue request\n";

            stop();
            return false;
        }
    }

    std::cout
        << "[camera] Camera started: "
        << cfg.size.width
        << "x"
        << cfg.size.height
        << '\n';

    return true;
}

bool Camera::read(cv::Mat& frame)
{
    auto& p = *impl_;

    std::unique_lock lock(p.mutex);

    const bool ready =
        p.condition.wait_for(
            lock,
            std::chrono::seconds(2),
            [&] {
                return !p.frames.empty()
                    || !p.running;
            });

    if (!ready || p.frames.empty())
        return false;

    frame = std::move(p.frames.front());
    p.frames.pop();

    return true;
}

void Camera::stop()
{
    auto& p = *impl_;

    if (!p.manager)
        return;

    p.stopping = true;

    if (p.camera && p.running)
        p.camera->stop();

    p.running = false;

    if (p.camera)
        p.camera->requestCompleted.disconnect(
            &p);

    p.requests.clear();

    for (auto& [buffer, mapping] : p.mapped) {
        if (mapping.data != MAP_FAILED)
            munmap(
                mapping.data,
                mapping.length);
    }

    p.mapped.clear();

    {
        std::lock_guard lock(p.mutex);

        while (!p.frames.empty())
            p.frames.pop();
    }

    p.allocator.reset();
    p.configuration.reset();

    p.stream = nullptr;

    if (p.camera) {
        p.camera->release();
        p.camera.reset();
    }

    p.manager->stop();
    p.manager.reset();

    p.stopping = false;
}