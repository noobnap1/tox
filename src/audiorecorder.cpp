#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "audiorecorder.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>

AudioRecorder::AudioRecorder() {
    ma_device_config config = ma_device_config_init(ma_device_type_capture);

    config.capture.format = ma_format_f32;
    config.capture.channels = CHANNELS;
    config.sampleRate = SAMPLE_RATE;
    config.dataCallback = data_callback;
    config.pUserData = this;

    if (ma_device_init(nullptr, &config, &device) != MA_SUCCESS)
        throw std::runtime_error("failed to initialize microphone");

    if (ma_device_start(&device) != MA_SUCCESS) {
        ma_device_uninit(&device);
        throw std::runtime_error("failed to start microphone");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

AudioRecorder::~AudioRecorder() {
    ma_device_uninit(&device);
}

std::vector<float> AudioRecorder::record(
    float speech_threshold,
    int silence_duration_ms,
    int no_speech_timeout_ms,
    int max_duration_ms,
    int preroll_ms
) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        samples.clear();
    }

    capturing = true;

    const auto start = std::chrono::steady_clock::now();
    auto last_loud = start;

    bool has_spoken = false;
    size_t checked = 0;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        float rms = 0.0f;

        {
            std::lock_guard<std::mutex> lock(mutex);

            if (samples.size() > checked) {
                double sum = 0.0;

                for (size_t i = checked; i < samples.size(); ++i)
                    sum += samples[i] * samples[i];

                rms = static_cast<float>(
                    std::sqrt(sum / (samples.size() - checked))
                );

                checked = samples.size();
            }
        }

        const auto now = std::chrono::steady_clock::now();

        const auto ms = [&](auto a) {
            return std::chrono::duration_cast<
                std::chrono::milliseconds
            >(now - a).count();
        };

        if (rms > speech_threshold) {
            has_spoken = true;
            last_loud = now;
        }

        if (has_spoken &&
            ms(last_loud) >= silence_duration_ms)
            break;

        if (!has_spoken &&
            ms(start) >= no_speech_timeout_ms)
            break;

        if (ms(start) >= max_duration_ms)
            break;
    }

    capturing = false;

    std::lock_guard<std::mutex> lock(mutex);

    if (!has_spoken)
        return {};

    return samples;
}

void AudioRecorder::data_callback(
    ma_device* device,
    void*,
    const void* input,
    ma_uint32 frames
) {
    auto* self = static_cast<AudioRecorder*>(device->pUserData);

    if (!self || !input || !self->capturing)
        return;

    const auto* data =
        static_cast<const float*>(input);

    std::lock_guard<std::mutex> lock(self->mutex);

    self->samples.insert(
        self->samples.end(),
        data,
        data + frames
    );
}