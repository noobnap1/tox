#include "enroll.hpp"

#include <exception>
#include <iostream>

constexpr int CAMERA_INDEX = 0;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: enroll <name> [photo.jpg ...]\n";
        return 1;
    }

    try {
        return enroll::command(argc, argv, TOX_MODEL_DIR, CAMERA_INDEX);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}