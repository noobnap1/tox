#pragma once

#include "miniaudio.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

class AudioRecorder {
public:
    static constexpr uint32_t SAMPLE_RATE = 16000;
    static constexpr uint32_t CHANNELS = 1;

    AudioRecorder();
    ~AudioRecorder();

    std::vector<float> record(
        float speech_threshold = 0.015f,
        int silence_duration_ms = 1200,
        int no_speech_timeout_ms = 3000,
        int max_duration_ms = 30000,
        int preroll_ms = 250
    );

private:
    ma_device device{};
    std::vector<float> samples;
    std::mutex mutex;
    std::atomic<bool> capturing{false};

    static void data_callback(
        ma_device* device,
        void* output,
        const void* input,
        ma_uint32 frames
    );
};