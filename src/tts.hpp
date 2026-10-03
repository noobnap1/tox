#pragma once

#include <string>

class TTS {
public:
    bool init();
    void speak(const std::string& text);
    void shutdown();

private:
    bool initialized = false;
};
