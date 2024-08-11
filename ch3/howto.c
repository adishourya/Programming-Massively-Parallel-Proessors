// https://github.com/nothings/stb
// example to load in an image in C
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

int main() {
  int width, height, channels;
  unsigned char *img =
      stbi_load("some_image.png", &width, &height, &channels, 0);

  if (img == NULL) {
    printf("Error in loading the image\n");
    return 1;
  }

  printf(
      "Loaded image with a width of %dpx, a height of %dpx and %d channels\n",
      width, height, channels);

  // Process the image here
  // For example, access a pixel:
  // img[(y * width + x) * channels + channel_index]

  // Free the image memory
  stbi_image_free(img);

  return 0;
}
