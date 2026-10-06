#pragma once

#include <opencv2/core.hpp>

#include <memory>

class Camera {
public:
    Camera();
    ~Camera();

    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    bool start();
    bool read(cv::Mat& frame);
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};