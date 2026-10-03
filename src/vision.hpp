#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

struct VisionState {
    bool face = false;
    int faces = 0;
    cv::Rect box;        // largest face, frame pixels
    cv::Size frame;
    std::string name;    // empty = unknown
    float score = 0.f;
    uint64_t seq = 0;
};

class Vision {
public:
    struct Person {
        std::string name;
        cv::Mat feature;
    };

    Vision(int camera, std::filesystem::path detector, std::filesystem::path recognizer = {});
    ~Vision();

    // Call before start().
    void set_gallery(std::vector<Person> gallery);

    void start();
    void stop();

    VisionState snapshot() const;

private:
    void run(std::stop_token st);

    int camera_;
    std::filesystem::path detector_, recognizer_;
    std::vector<Person> gallery_;
    mutable std::mutex mutex_;
    VisionState state_;
    std::jthread thread_; // declared last so it joins first
};