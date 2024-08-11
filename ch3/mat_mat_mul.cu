// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_runtime_wrapper.h>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>

__global__ void Matmul_Kernel(int *ans, int *mat1, int *mat2) {
  int thread_x_id = (blockDim.x * blockIdx.x) + threadIdx.x;
  int thread_y_id = (blockDim.y * blockIdx.y) + threadIdx.y;

  if ((thread_x_id < 3) && (thread_y_id < 3)) {
    for (int i = 0; i < 3; i++) {
      ans[(thread_y_id * 3) + thread_x_id] +=
          mat1[(thread_y_id * 3) + i] * mat2[thread_x_id + (i * 3)];
    }
  }
}

int main() {
  // matrix @ vector multiplication
  // easy example!
  int mat1_h[3][3] = {
      {12, 0, 0},
      {0, -5, 0},
      {0, 0, -9},
  };

  int mat2_h[3][3] = {
      {-4, 2, 0},
      {0, 2, 0},
      {0, 0, -4},
  };

  // the idea is that each thread will do 1 dot product.

  int *mat1_d, *mat2_d, *ans_d;
  int *ans_h;
  ans_h = (int *)malloc(sizeof(mat1_h));

  cudaMalloc((void **)&mat1_d, sizeof(mat1_h));
  cudaMalloc((void **)&mat2_d, sizeof(mat1_h));
  cudaMalloc((void **)&ans_d, sizeof(mat1_h));

  cudaMemcpy(mat1_d, mat1_h, sizeof(mat1_h), cudaMemcpyHostToDevice);
  cudaMemcpy(mat2_d, mat2_h, sizeof(mat2_h), cudaMemcpyHostToDevice);
  // init the ans var to 0
  cudaMemset(ans_d, 0, sizeof(mat1_h));

  dim3 ThreadsperBlock = dim3(3, 3, 1);
  dim3 BlocksperGrid = dim3(1, 1, 1);

  Matmul_Kernel<<<BlocksperGrid, ThreadsperBlock>>>(ans_d, mat1_d, mat2_d);

  cudaMemcpy(ans_h, ans_d, sizeof(mat1_h), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 9; i++) {
    if (i % 3 == 0) {
      printf("\n");
    }
    printf("%d\t", ans_h[i]);
  }
  printf("\n");
  free(ans_h);
  cudaFree(mat1_d);
  cudaFree(mat2_d);
  printf("Done!\n");
  return 0;
}
