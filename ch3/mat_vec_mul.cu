// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_runtime_wrapper.h>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>

__global__ void Matmul_Kernel(int *ans, int *mat, int *vec) {
  int thread_id = (blockDim.x * blockIdx.x) + threadIdx.x;
  for (int i = 0; i < 3; i++) {
    // remember what you copied over was a 2d array for mat.
    // but in cuda everything is a single dim array (row major) vector
    // so doing this would be wrong: mat[thread_id][i].. you have to write the
    // flattened version
    ans[thread_id] += mat[thread_id * 3 + i] * vec[i];
  }
}

int main() {
  // matrix @ vector multiplication
  // easy example!
  int mat_h[3][3] = {
      {12, 0, 0},
      {0, -5, 0},
      {0, 0, -9},
  };

  int vec_h[3] = {-1, -2, -3};
  // the idea is that each thread will do 1 dot product.

  int *mat_d, *vec_d, *ans_d;
  int *ans_h;
  ans_h = (int *)malloc(sizeof(vec_h));

  cudaMalloc((void **)&mat_d, sizeof(mat_h));
  cudaMalloc((void **)&vec_d, sizeof(vec_h));
  cudaMalloc((void **)&ans_d, sizeof(vec_h));

  cudaMemcpy(mat_d, mat_h, sizeof(mat_h), cudaMemcpyHostToDevice);
  cudaMemcpy(vec_d, vec_h, sizeof(vec_h), cudaMemcpyHostToDevice);
  // init the ans var to 0
  cudaMemset(ans_d, 0, 3 * sizeof(int));

  dim3 ThreadsperBlock = dim3(3, 1, 1);
  dim3 BlocksperGrid = dim3(1, 1, 1);

  Matmul_Kernel<<<BlocksperGrid, ThreadsperBlock>>>(ans_d, mat_d, vec_d);

  cudaMemcpy(ans_h, ans_d, sizeof(vec_h), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 3; i++) {
    printf("%d\t", ans_h[i]);
  }
  printf("\n");
  free(ans_h);
  cudaFree(mat_d);
  cudaFree(vec_d);
  printf("Done!\n");
  return 0;
}
