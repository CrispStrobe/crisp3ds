/* Inspect cameras and sparse references as parsed by the selected MVE code. */
#include <iomanip>
#include <iostream>
#include "mve/scene.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    auto scene = mve::Scene::create(argv[1]);
    auto bundle = scene->get_bundle();
    std::cout << std::setprecision(12);
    std::cout << "BUNDLE " << bundle->get_cameras().size() << ' '
              << bundle->get_features().size() << '\n';
    for (auto const& view : scene->get_views()) {
      if (!view) continue;
      auto const* image = view->get_image_proxy("undistorted");
      if (!image) return 3;
      auto const& camera = view->get_camera();
      float k[9];
      camera.fill_calibration(k, image->width, image->height);
      std::cout << "VIEW " << view->get_id() << ' ' << image->width << ' ' << image->height;
      for (float value : k) std::cout << ' ' << value;
      for (float value : camera.rot) std::cout << ' ' << value;
      for (float value : camera.trans) std::cout << ' ' << value;
      std::cout << '\n';
    }
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
