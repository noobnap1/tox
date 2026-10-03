#include "vision.hpp"

#include <opencv2/objdetect/face.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <linux/videodev2.h>
#include <stop_token>
#include <string>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>

namespace {

bool is_camera_device(const std::string& device) {
    const int fd = ::open(
        device.c_str(),
        O_RDWR | O_NONBLOCK);

    if (fd < 0) {
        return false;
    }

    v4l2_capability capability{};

    const bool queried =
        ::ioctl(
            fd,
            VIDIOC_QUERYCAP,
            &capability) == 0;

    ::close(fd);

    if (!queried) {
        return false;
    }

    if (capability.capabilities & V4L2_CAP_DEVICE_CAPS) {
        return
            (capability.device_caps & V4L2_CAP_VIDEO_CAPTURE) ||
            (capability.device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE);
    }

    return
        (capability.capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
        (capability.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE);
}

}

Vision::Vision(
    int camera,
    std::filesystem::path detector,
    std::filesystem::path recognizer)
    : camera_(camera),
      detector_(std::move(detector)),
      recognizer_(std::move(recognizer)) {}

Vision::~Vision() {
    stop();
}

void Vision::set_gallery(std::vector<Person> gallery) {
    gallery_ = std::move(gallery);
}

void Vision::start() {
    thread_ = std::jthread([this](std::stop_token st) {
        run(st);
    });
}

void Vision::stop() {
    if (thread_.joinable()) {
        thread_.request_stop();
        thread_.join();
    }
}

VisionState Vision::snapshot() const {
    std::lock_guard lock(mutex_);
    return state_;
}

void Vision::run(std::stop_token st) {
    cv::setNumThreads(1);

    using clock = std::chrono::steady_clock;

    constexpr auto period =
        std::chrono::milliseconds(66);

    constexpr auto retry_period =
        std::chrono::seconds(5);

    cv::VideoCapture cap;

    bool camera_present = false;
    bool no_camera_reported = false;

    cv::Ptr<cv::FaceDetectorYN> det =
        cv::FaceDetectorYN::create(
            detector_.string(),
            "",
            cv::Size(640, 640),
            0.7f,
            0.3f,
            5000);

    cv::Ptr<cv::FaceRecognizerSF> rec;

    if (!recognizer_.empty() && !gallery_.empty()) {
        rec =
            cv::FaceRecognizerSF::create(
                recognizer_.string(),
                "");
    }

    cv::Mat frame;
    cv::Mat faces;

    while (!st.stop_requested()) {
        if (!cap.isOpened()) {
            const std::string device =
                "/dev/video" +
                std::to_string(camera_);

            if (!is_camera_device(device)) {
                if (!no_camera_reported) {
                    std::cout
                        << "[vision] No camera"
                        << std::endl;

                    no_camera_reported = true;
                }

                {
                    std::lock_guard lock(mutex_);

                    VisionState s;

                    s.seq =
                        state_.seq + 1;

                    s.frame = {};
                    s.faces = 0;
                    s.face = false;
                    s.name.clear();
                    s.score = 0.0f;
                    s.box = {};

                    state_ =
                        std::move(s);
                }

                std::this_thread::sleep_for(
                    retry_period);

                continue;
            }

            if (cap.open(
                    camera_,
                    cv::CAP_V4L2)) {

                cap.set(
                    cv::CAP_PROP_FRAME_WIDTH,
                    640);

                cap.set(
                    cv::CAP_PROP_FRAME_HEIGHT,
                    480);

                cap.set(
                    cv::CAP_PROP_BUFFERSIZE,
                    1);

                camera_present = true;
                no_camera_reported = false;

                std::cout
                    << "[vision] Camera connected"
                    << std::endl;
            } else {
                cap.release();

                if (!no_camera_reported) {
                    std::cout
                        << "[vision] No camera"
                        << std::endl;

                    no_camera_reported = true;
                }

                {
                    std::lock_guard lock(mutex_);

                    VisionState s;

                    s.seq =
                        state_.seq + 1;

                    s.frame = {};
                    s.faces = 0;
                    s.face = false;
                    s.name.clear();
                    s.score = 0.0f;
                    s.box = {};

                    state_ =
                        std::move(s);
                }

                std::this_thread::sleep_for(
                    retry_period);

                continue;
            }
        }

        const auto t0 =
            clock::now();

        if (!cap.read(frame) ||
            frame.empty()) {

            cap.release();

            if (camera_present) {
                camera_present = false;

                std::cout
                    << "[vision] Camera disconnected"
                    << std::endl;
            }

            no_camera_reported = false;

            {
                std::lock_guard lock(mutex_);

                VisionState s;

                s.seq =
                    state_.seq + 1;

                s.frame = {};
                s.faces = 0;
                s.face = false;
                s.name.clear();
                s.score = 0.0f;
                s.box = {};

                state_ =
                    std::move(s);
            }

            std::this_thread::sleep_for(
                retry_period);

            continue;
        }

        det->setInputSize(
            frame.size());

        det->detect(
            frame,
            faces);

        VisionState s;

        s.frame =
            frame.size();

        s.faces =
            faces.rows;

        if (faces.rows > 0) {
            int best = 0;

            float best_area =
                0.0f;

            for (int i = 0;
                 i < faces.rows;
                 ++i) {

                const float area =
                    faces.at<float>(
                        i,
                        2) *
                    faces.at<float>(
                        i,
                        3);

                if (area > best_area) {
                    best_area = area;
                    best = i;
                }
            }

            s.face = true;

            s.box = cv::Rect(
                static_cast<int>(
                    faces.at<float>(
                        best,
                        0)),

                static_cast<int>(
                    faces.at<float>(
                        best,
                        1)),

                static_cast<int>(
                    faces.at<float>(
                        best,
                        2)),

                static_cast<int>(
                    faces.at<float>(
                        best,
                        3)));

            if (rec) {
                cv::Mat aligned;
                cv::Mat feat;

                rec->alignCrop(
                    frame,
                    faces.row(best),
                    aligned);

                rec->feature(
                    aligned,
                    feat);

                feat =
                    feat.clone();

                for (const auto& p :
                     gallery_) {

                    const double score =
                        rec->match(
                            feat,
                            p.feature,
                            cv::FaceRecognizerSF::DisType::FR_COSINE);

                    if (
                        score > s.score &&
                        score >= 0.363
                    ) {
                        s.score =
                            static_cast<float>(
                                score);

                        s.name =
                            p.name;
                    }
                }
            }
        }

        {
            std::lock_guard lock(mutex_);

            s.seq =
                state_.seq + 1;

            state_ =
                std::move(s);
        }

        std::this_thread::sleep_until(
            t0 + period);
    }

    cap.release();
}