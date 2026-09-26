#include <png.h>

#include <cstdio>

int main() {
    const auto compiled = static_cast<unsigned long>(PNG_LIBPNG_VER);
    const auto loaded = static_cast<unsigned long>(png_access_version_number());
    std::printf("libpng compiled=%lu runtime=%lu\n", compiled, loaded);
    return compiled == loaded ? 0 : 1;
}
