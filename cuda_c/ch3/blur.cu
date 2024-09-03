// Incomplete ! I hate this .. also unneeded.
// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_runtime_wrapper.h>
#include <__clang_cuda_builtin_vars.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#define BLUR_SIZE 3

#include <cmath>

typedef struct Image {
  uint8_t *img_arr;
  int height;
  int width;
  int channel;
  int img_size;
} Image;

__global__ void BlurImg_Kernel(uint8_t *ans, const uint8_t *input,
                               const int width, const int height,
                               const int channel) {

  int thread_x_id = (blockDim.x * blockIdx.x) + threadIdx.x;
  int thread_y_id = (blockDim.y * blockIdx.y) + threadIdx.y;

  // valid threads
  if ((thread_x_id < width) && (thread_y_id > height)) {
    // blur window calculation
    int blur_windSum = 0;
    int blur_windCnt = 0; // px in wind

    for (int wind_x = -BLUR_SIZE; wind_x < BLUR_SIZE + 1; wind_x++) {
      for (int wind_y = -BLUR_SIZE; wind_y < BLUR_SIZE + 1; wind_y++) {
        //
        int x = thread_x_id + wind_x;
        int y = thread_y_id + wind_y;

        // verify we have a valid image pixel
        if ((x >= 0 && x <= width) && (y >= 0 && y <= height)) {
          blur_windSum +=
        }
      }
    }

  } else {
    printf("Thread : %d,%d Idling!\n", thread_x_id, thread_y_id);
  }
}

int main() {
  Image loaded_img;
  // force in to load a rgb image
  loaded_img.img_arr = stbi_load("some_image.png", &loaded_img.width,
                                 &loaded_img.height, &loaded_img.channel, 3);

  loaded_img.img_size =
      loaded_img.height * loaded_img.width * loaded_img.channel;
  printf("Dimension :%d,%d,%d\n", loaded_img.width, loaded_img.height,
         loaded_img.channel);

  // Device variables
  uint8_t *result_d, *input_d;

  // to copy back the answer
  uint8_t *result_h = (uint8_t *)malloc(loaded_img.img_size);

  // allocate space in gpu
  cudaMalloc((void **)&result_d, loaded_img.img_size);
  cudaMalloc((void **)&input_d, loaded_img.img_size);

  // copy the input image
  cudaMemcpy(input_d, loaded_img.img_arr, loaded_img.img_size,
             cudaMemcpyHostToDevice);

  // kernel run settings
  dim3 ThreadsPerBlock = dim3(16, 16, 1);
  int row_blocks = (int)ceil((float)loaded_img.width / 16);
  int col_blocks = (int)ceil((float)loaded_img.height / 16);
  dim3 BlocksPerGrid = dim3(row_blocks, col_blocks, 1);

  // each block would write on a single pixel value of the answer
  BlurImg_Kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(
      result_d, input_d, loaded_img.width, loaded_img.height,
      loaded_img.channel);

  // free the space from the gpu
  cudaFree(result_d);
  cudaFree(input_d);
  // write the image
  stbi_write_png("blurred_by_gpu.png", loaded_img.width, loaded_img.height,
                 loaded_img.channel, result_h,
                 loaded_img.height * loaded_img.channel);
  // free the host variable
  free(result_h);

  stbi_image_free(loaded_img.img_arr);
  printf("Done!\n");
  return 0;
}
