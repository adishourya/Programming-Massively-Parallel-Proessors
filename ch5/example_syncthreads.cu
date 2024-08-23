// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_runtime_wrapper.h>
#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#define ARR_SIZE 32

// The need for barrier.
// classic example is read and write

// simple cpu code where this not a problem

void init_arr(int *a, int dim) {
  for (int i = 0; i < dim; i++) {
    a[i] = i;
  }
}

void print_arr(int *a, int dim) {
  printf("-----------\n");
  for (int i = 0; i < dim; i++) {
    printf("a[%d]=%d\n", i, a[i]);
  }
  printf("-----------\n");
}

void shift_left_cpu(int *a, int dim) {
  // takes in an array of size dim
  for (int i = 0; i < dim - 1; i++) {
    // that is read i + 1 and write it to i
    a[i] = a[i + 1];
  }
}

__global__ void shift_left_kernel(int *a) {
  // now we cant just shift left like cpu.
  // The threads in the block needs to work together

  // This is an automatic variable (private to the thread)
  int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

  // you cant just do this
  // if (idx < (ARR_SIZE - 1)){
  //   AS: to read idx + 1.. the other array must have done writing to it.
  //   And: This is a read and a write operation (2 ops)
  //   array[idx] = array[idx + 1];
  // }

  // we need 3 synchronization

  // we will keep the array in the shared memory (private to the block)
  // so that we dont have to read from the dram everytime we shift an index
  __shared__ int arr_sh[ARR_SIZE]; // this has to be static.

  // init
  // now every thread must have done their below mapping operation
  // before we proceed to shift. so we sync
  // first write operation
  arr_sh[idx] = idx;
  __syncthreads();

  if (idx < (ARR_SIZE - 1)) {
    // let every thread read the next element
    // first read operation
    // tmp is a thread privated variable.
    int tmp = arr_sh[idx + 1];
    __syncthreads();
    // second write operation
    // but this time we write it to the operand and not shared memory
    a[idx] = tmp;
    __syncthreads();
    // not really needed but good practice. so that no one does anything with it
    // . until the final bit is done!
  }
}

void cpu_code() {
  int some_arr[ARR_SIZE];
  // init array
  init_arr(some_arr, ARR_SIZE);
  // print array
  print_arr(some_arr, ARR_SIZE);
  // shift array with cpu
  shift_left_cpu(some_arr, ARR_SIZE);
  // print array
  print_arr(some_arr, ARR_SIZE);
}

void gpu_code() {
  int *some_arr_d;
  cudaMalloc((void **)&some_arr_d, sizeof(int) * ARR_SIZE);
  int *ans_arr;
  ans_arr = (int *)malloc(sizeof(int) * ARR_SIZE);
  // no need to init the input arr . as we will only use it to write on it.
  shift_left_kernel<<<1, ARR_SIZE>>>(some_arr_d);
  cudaMemcpy(ans_arr, some_arr_d, sizeof(int) * ARR_SIZE,
             cudaMemcpyDeviceToHost);
  cudaFree(some_arr_d);
  print_arr(ans_arr, ARR_SIZE);
  free(ans_arr);
}

int main() {
  // cpu_code();
  gpu_code();
  return 0;
}
