#include "tts.hpp"

#include <espeak-ng/speak_lib.h>

bool TTS::init() {
    if (initialized) {
        return true;
    }

    if (espeak_Initialize(
            AUDIO_OUTPUT_PLAYBACK,
            0,
            nullptr,
            0
        ) == -1) {
        return false;
    }

    espeak_SetVoiceByName("en");
    espeak_SetParameter(espeakRATE, 150, 0);
    espeak_SetParameter(espeakPITCH, 35, 0);
    espeak_SetParameter(espeakRANGE, 10, 0);

    initialized = true;
    return true;
}

void TTS::speak(const std::string& text) {
    if (!initialized || text.empty()) {
        return;
    }

    espeak_Synth(
        text.c_str(),
        text.size() + 1,
        0,
        POS_CHARACTER,
        0,
        espeakCHARS_AUTO,
        nullptr,
        nullptr
    );

    espeak_Synchronize();
}

void TTS::shutdown() {
    if (!initialized) {
        return;
    }

    espeak_Synchronize();
    espeak_Terminate();

    initialized = false;
}