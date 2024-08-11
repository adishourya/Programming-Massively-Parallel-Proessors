// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_runtime_wrapper.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define INCORRECT_NUM_CHANNELS -3
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include <cmath>

typedef struct Image {
  uint8_t *img_arr;
  int height;
  int width;
  int channels;
  int img_size;
} Image;

__global__ void greyScaling_kernel(uint8_t *ans, uint8_t *input,
                                   int threads_needed, int channels) {
  //

  int thread_id = (blockDim.x * blockIdx.x) + threadIdx.x;

  if (thread_id < threads_needed) {
    int start_id = thread_id * channels;
    int r = input[start_id];
    int g = input[start_id + 1];
    int b = input[start_id + 2];

    uint8_t grey_pixel = (uint8_t)(0.2 * r + 0.2 * g + 0.2 * b);

    ans[start_id] = grey_pixel;
    ans[start_id + 1] = grey_pixel;
    ans[start_id + 2] = grey_pixel;
  } else {
    printf("why am i here ? just to suffer!!\n");
  }
}

int main() {
  //
  Image loaded_img;

  // last arg to force RGB image
  loaded_img.img_arr = stbi_load("some_image.png", &loaded_img.width,
                                 &loaded_img.height, &loaded_img.channels, 3);

  loaded_img.img_size =
      loaded_img.channels * loaded_img.height * loaded_img.width;

  printf("Img size : %d\n", loaded_img.img_size);

  // run settings
  // we will use one thread to evaluate one pixel i.e 3 elements of img_arr
  int threads_needed = loaded_img.img_size / loaded_img.channels;
  printf("Threads needed to process the image : %d\n", threads_needed);

  // 512 / 32 = 16 warps ... is optimal for new cards!
  int threads_per_block = 512;
  int blocks = threads_needed / threads_per_block;

  // allocate memory and send it to gpu
  uint8_t *result_d, *input_d;
  uint8_t *result_h = (uint8_t *)malloc(loaded_img.img_size);

  cudaMalloc((void **)&result_d, loaded_img.img_size);
  cudaMalloc((void **)&input_d, loaded_img.img_size);

  cudaMemcpy(input_d, loaded_img.img_arr, loaded_img.img_size,
             cudaMemcpyHostToDevice);

  greyScaling_kernel<<<blocks, threads_per_block>>>(
      result_d, input_d, threads_needed, loaded_img.channels);

  // take back answer to the host
  cudaMemcpy(result_h, result_d, loaded_img.img_size, cudaMemcpyDeviceToHost);
  cudaFree(result_d);
  cudaFree(result_h);

  // write the image!
  stbi_write_png("greyscaled_by_gpu.png", loaded_img.width, loaded_img.height,
                 loaded_img.channels, result_h,
                 loaded_img.width * loaded_img.channels);

  free(result_h);
  stbi_image_free(loaded_img.img_arr);
  return 0;
}
