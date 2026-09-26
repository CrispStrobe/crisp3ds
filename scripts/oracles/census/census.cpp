// Independent CPU Census/Hamming block-matching baseline for local evaluation.
// It intentionally uses no OpenCV code or ground-truth input.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Image { int width, height; std::vector<std::uint8_t> pixels; };

std::string token(std::istream &in) {
  std::string result;
  while (in >> result) {
    if (result[0] == '#') { std::string comment; std::getline(in, comment); continue; }
    return result;
  }
  throw std::runtime_error("Truncated PGM header");
}
int number(const std::string &value, const char *name) {
  std::size_t consumed = 0;
  int result = std::stoi(value, &consumed);
  if (consumed != value.size()) throw std::runtime_error(std::string("Invalid ") + name);
  return result;
}
Image read_pgm(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open PGM: " + path.string());
  if (token(in) != "P5") throw std::runtime_error("Expected P5 PGM");
  const int width = number(token(in), "width"), height = number(token(in), "height");
  if (width < 8 || height < 8 || static_cast<std::uint64_t>(width) * height > 8'000'000)
    throw std::runtime_error("PGM dimensions out of bounds");
  if (number(token(in), "max value") != 255) throw std::runtime_error("Expected 8-bit PGM");
  if (in.get() != '\n') throw std::runtime_error("Expected newline after PGM header");
  Image image{width, height, std::vector<std::uint8_t>(static_cast<std::size_t>(width) * height)};
  in.read(reinterpret_cast<char *>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
  if (!in || in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("PGM payload length mismatch");
  return image;
}
std::vector<std::uint32_t> census(const Image &image) {
  const int w = image.width, h = image.height;
  std::vector<std::uint32_t> result(image.pixels.size(), 0);
  for (int y = 2; y < h - 2; ++y) for (int x = 2; x < w - 2; ++x) {
    const int center = image.pixels[static_cast<std::size_t>(y) * w + x];
    std::uint32_t bits = 0;
    for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx) {
      if (dy == 0 && dx == 0) continue;
      bits = (bits << 1) | (image.pixels[static_cast<std::size_t>(y + dy) * w + x + dx] < center);
    }
    result[static_cast<std::size_t>(y) * w + x] = bits;
  }
  return result;
}
void write_pfm(const std::filesystem::path &path, const Image &size, const std::vector<float> &disparity) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot open output PFM");
  out << "Pf\n" << size.width << ' ' << size.height << "\n-1.0\n";
  for (int y = size.height - 1; y >= 0; --y) {
    for (int x = 0; x < size.width; ++x) {
      const auto bits = std::bit_cast<std::uint32_t>(disparity[static_cast<std::size_t>(y) * size.width + x]);
      const char raw[4] = {static_cast<char>(bits), static_cast<char>(bits >> 8),
                           static_cast<char>(bits >> 16), static_cast<char>(bits >> 24)};
      out.write(raw, 4);
    }
  }
  if (!out) throw std::runtime_error("Failed writing output PFM");
}
std::vector<float> match(const Image &left, const Image &right, int disparities) {
  if (left.width != right.width || left.height != right.height)
    throw std::runtime_error("Left and right dimensions differ");
  const int w = left.width, h = left.height;
  if (disparities < 1 || disparities > 2048 ||
      static_cast<std::uint64_t>(w) * h * disparities > 1'500'000'000ULL)
    throw std::runtime_error("Disparity search exceeds bound");
  const auto left_bits = census(left), right_bits = census(right);
  const auto pixels = left.pixels.size();
  std::vector<float> result(pixels, std::numeric_limits<float>::infinity());
  std::vector<int> best(pixels, std::numeric_limits<int>::max());
  std::vector<std::uint8_t> costs(pixels);
  std::vector<std::uint32_t> integral(static_cast<std::size_t>(w + 1) * (h + 1));
  // Search d=0..disparities-1, matching left(x,y) against right(x-d,y).
  // Each disparity uses a 5x5 Census signature plus a 3x3 box sum.
  for (int d = 0; d < disparities; ++d) {
    std::fill(costs.begin(), costs.end(), 255);
    for (int y = 2; y < h - 2; ++y) for (int x = d + 2; x < w - 2; ++x) {
      const auto i = static_cast<std::size_t>(y) * w + x, j = i - d;
      const int census_cost = std::popcount(left_bits[i] ^ right_bits[j]);
      const int intensity_cost = std::abs(int(left.pixels[i]) - int(right.pixels[j])) / 16;
      costs[i] = static_cast<std::uint8_t>(census_cost + intensity_cost);
    }
    std::fill(integral.begin(), integral.end(), 0);
    for (int y = 0; y < h; ++y) {
      std::uint32_t row_sum = 0;
      for (int x = 0; x < w; ++x) {
        row_sum += costs[static_cast<std::size_t>(y) * w + x];
        integral[static_cast<std::size_t>(y + 1) * (w + 1) + x + 1] =
            integral[static_cast<std::size_t>(y) * (w + 1) + x + 1] + row_sum;
      }
    }
    for (int y = 3; y < h - 3; ++y) for (int x = d + 3; x < w - 3; ++x) {
      const auto top = static_cast<std::size_t>(y - 1) * (w + 1);
      const auto bottom = static_cast<std::size_t>(y + 2) * (w + 1);
      const int cost = static_cast<int>(integral[bottom + x + 2] - integral[top + x + 2]
                                       - integral[bottom + x - 1] + integral[top + x - 1]);
      const auto i = static_cast<std::size_t>(y) * w + x;
      if (cost < best[i]) { best[i] = cost; result[i] = static_cast<float>(d); }
    }
  }
  return result;
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::string left, right, output;
    int ndisp = 0;
    bool seen_left = false, seen_right = false, seen_ndisp = false, seen_output = false;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 >= argc) throw std::runtime_error("Expected --left, --right, --ndisp and --output");
      const std::string option = argv[i];
      if (option == "--left" && !seen_left) { left = argv[i + 1]; seen_left = true; }
      else if (option == "--right" && !seen_right) { right = argv[i + 1]; seen_right = true; }
      else if (option == "--ndisp" && !seen_ndisp) { ndisp = number(argv[i + 1], "ndisp"); seen_ndisp = true; }
      else if (option == "--output" && !seen_output) { output = argv[i + 1]; seen_output = true; }
      else throw std::runtime_error("Unknown option: " + option);
    }
    if (left.empty() || right.empty() || output.empty() || ndisp < 1)
      throw std::runtime_error("Expected --left, --right, --ndisp and --output");
    const auto left_image = read_pgm(left), right_image = read_pgm(right);
    if (ndisp >= left_image.width) throw std::runtime_error("Disparity count must be smaller than image width");
    if (std::filesystem::symlink_status(output).type() != std::filesystem::file_type::not_found)
      throw std::runtime_error("Output already exists");
    write_pfm(output, left_image, match(left_image, right_image, ndisp));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "census baseline: " << error.what() << '\n';
    return 1;
  }
}
