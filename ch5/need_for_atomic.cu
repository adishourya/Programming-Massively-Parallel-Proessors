// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_runtime_wrapper.h>
// #include <__clang_cuda_intrinsics.h>
// #include <__clang_cuda_runtime_wrapper.h>
#include <cstdio>
#include <stdio.h>

#define ARRSIZE 5

void print_arr(int *arr, int dim) {
  for (int i = 0; i < dim; i++) {
    printf("arr[%d]=%d\n", i, arr[i]);
  }
}

// This example shows what happens if a lot of thread tries to write on the same
// address

__global__ void increment_naive_kernel(int *a, int dim) {
  // takes in an array and dim of the array from the gpu dram.
  int thread_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  int idx = thread_idx % dim; // so now each thread indices ranges from [0,dim)

  // each thread tries to read in and write on the same addr
  a[idx] = a[idx] + 1;
}

__global__ void increment_atomic(int *a, int dim) {
  // takes in an array and dim of the array from the gpu dram.
  int thread_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  int idx = thread_idx % dim; // so now each thread indices ranges from [0,dim)

  // this is kinda slow tho. A map operation is always faster
  atomicAdd(&a[idx], 1);
}

int main() {
  int *ans_arr;
  ans_arr = (int *)malloc(sizeof(int) * ARRSIZE);

  int *some_arr_d;
  cudaMalloc((void **)&some_arr_d, sizeof(int) * ARRSIZE);
  cudaMemset(some_arr_d, 0, sizeof(int) * ARRSIZE);

  // increment_naive_kernel<<<1000, 1024>>>(some_arr_d, ARRSIZE);
  // for smaller sized thread counts .. some might not even be able to write to
  // it . so give excessive #threads to see pronounced effect this for some run
  // printed out all 82/ all 85.. different in each run.

  increment_atomic<<<1000, 1024>>>(some_arr_d, ARRSIZE);
  // this shows each element as 204800
  // which is correct as : (1000 × 1024) / 5 = 204800

  cudaMemcpy(ans_arr, some_arr_d, sizeof(int) * ARRSIZE,
             cudaMemcpyDeviceToHost);

  cudaFree(some_arr_d);
  print_arr(ans_arr, ARRSIZE);

  free(ans_arr);
  return 0;
}
