cmake_minimum_required(VERSION 4.4)

project(tox CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(llama CONFIG REQUIRED)
find_package(whisper CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ESPEAK_NG REQUIRED espeak-ng)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn objdetect)

add_library(miniaudio INTERFACE)

# Face enrollment code shared by tox (loads the gallery) and the enroll tool (creates it).
add_library(face_enroll STATIC
    src/enroll.cpp
)

target_include_directories(face_enroll PUBLIC
    src
)

target_link_libraries(face_enroll PUBLIC
    ${OpenCV_LIBS}
)

# Main voice chatbot.
add_executable(tox
    src/main.cpp
    src/tts.cpp
    src/vision.cpp
    src/audiorecorder.cpp
)

target_include_directories(tox PRIVATE
    ${ESPEAK_NG_INCLUDE_DIRS}
)

target_link_libraries(tox PRIVATE
    llama
    whisper
    miniaudio
    espeak-ng
    face_enroll
)

# Face enrollment tool. Not part of the default build: use `cmake --build --preset enroll`.
add_executable(enroll EXCLUDE_FROM_ALL
    src/enrollMain.cpp
)

target_link_libraries(enroll PRIVATE
    face_enroll
)