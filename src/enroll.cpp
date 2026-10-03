#include "enroll.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

// Detects the largest face in `frame`, aligns it, and appends its SFace feature as a new row of `feats`.
bool add_face(const cv::Mat& frame, cv::FaceDetectorYN& det, cv::FaceRecognizerSF& rec, cv::Mat& feats) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(640, 640));

    cv::Mat faces;
    det.setInputSize(resized.size());
    det.detect(resized, faces);

    if (faces.rows == 0) return false;

    int best = 0;
    float best_area = 0.0f;

    for (int i = 0; i < faces.rows; ++i) {
        const float a = faces.at<float>(i, 2) * faces.at<float>(i, 3);

        if (a > best_area) {
            best_area = a;
            best = i;
        }
    }

    cv::Mat aligned, feat;

    cv::Mat face = faces.row(best).clone();

    float sx = static_cast<float>(frame.cols) / 640.0f;
    float sy = static_cast<float>(frame.rows) / 640.0f;

    face.at<float>(0, 0) *= sx;
    face.at<float>(0, 1) *= sy;
    face.at<float>(0, 2) *= sx;
    face.at<float>(0, 3) *= sy;

    for (int j = 4; j < 14; j += 2) {
        face.at<float>(0, j) *= sx;
        face.at<float>(0, j + 1) *= sy;
    }

    rec.alignCrop(frame, face, aligned);
    rec.feature(aligned, feat);

    feats.push_back(feat.clone());

    return true;
}

int save(const std::string& name, const cv::Mat& feats, const fs::path& dir) {
    if (feats.empty()) {
        std::cerr << "[enroll] no faces found, nothing saved\n";
        return 1;
    }

    fs::create_directories(dir);

    cv::FileStorage out((dir / (name + ".yml")).string(), cv::FileStorage::WRITE);
    out << "features" << feats;

    std::cout << "[enroll] saved " << feats.rows << " samples for " << name << "\n";
    return 0;
}

} // namespace

namespace enroll {

std::vector<Vision::Person> load_gallery(const fs::path& dir) {
    std::vector<Vision::Person> gallery;

    if (!fs::exists(dir)) return gallery;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".yml") continue;

        cv::FileStorage in(entry.path().string(), cv::FileStorage::READ);
        cv::Mat feats;
        in["features"] >> feats;

        for (int i = 0; i < feats.rows; ++i) {
            gallery.push_back({entry.path().stem().string(), feats.row(i).clone()});
        }
    }

    return gallery;
}

int command(int argc, char** argv, const fs::path& model_dir, int camera) {
    const std::string name = argv[1];
    const fs::path dir = model_dir / "faces";

    auto det = cv::FaceDetectorYN::create(
        (model_dir / "face_detection_yunet_2023mar.onnx").string(), "", cv::Size(640, 640), 0.7f, 0.3f, 5000);
    auto rec = cv::FaceRecognizerSF::create((model_dir / "face_recognition_sface_2021dec.onnx").string(), "");

    cv::Mat feats;

    if (argc > 2) {
        for (int i = 2; i < argc; ++i) {
            const cv::Mat img = cv::imread(argv[i]);

            if (img.empty()) {
                std::cerr << "[enroll] can't read " << argv[i] << "\n";
                continue;
            }

            if (!add_face(img, *det, *rec, feats)) std::cerr << "[enroll] no face in " << argv[i] << "\n";
        }

        return save(name, feats, dir);
    }

    cv::VideoCapture cap(camera, cv::CAP_V4L2);

    if (!cap.isOpened()) {
        std::cerr << "[enroll] can't open camera " << camera << "\n";
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    constexpr int samples = 8;
    using clock = std::chrono::steady_clock;
    cv::Mat frame;

    std::cout << "[enroll] look at the camera and slowly turn your head a little between samples\n";

    for (int tries = 0; feats.rows < samples && tries < samples * 6; ++tries) {
        const auto until = clock::now() + std::chrono::milliseconds(700);

        while (clock::now() < until) cap.grab(); // drop stale frames while the person moves

        if (!cap.read(frame) || frame.empty()) continue;

        if (add_face(frame, *det, *rec, feats)) std::cout << "[enroll] sample " << feats.rows << "/" << samples << "\n";
    }

    return save(name, feats, dir);
}

} // namespace enroll