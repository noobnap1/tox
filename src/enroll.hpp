#pragma once

#include "vision.hpp"

#include <filesystem>
#include <vector>

namespace enroll {

// Loads every <dir>/<name>.yml into the gallery (one Person entry per stored feature).
std::vector<Vision::Person> load_gallery(const std::filesystem::path& dir);

// argv: <name> [images...]
//   <name>             -> captures samples from the camera
//   <name> a.jpg b.jpg -> uses photos (one face each works best)
int command(int argc, char** argv, const std::filesystem::path& model_dir, int camera);

} // namespace enroll