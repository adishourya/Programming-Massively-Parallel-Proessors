#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

#define SEQ 10
#define HSIZE 10

// Utility functions
void __initArr(float *arr, int dim) {
  for (int i = 0; i < dim; i++) {
    arr[i] = 2.0f; // Initialize with some values
  }
}

void printArr(float *arr, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("arr[%d][%d]=%f\t", i, j, arr[i * cols + j]);
    }
    printf("\n");
  }
}

// CUDA kernel for computing Flash Attention
__global__ void flash_normal(float *ans, float *q, float *KT, float *V,
                             int seq_len, int head_size) {
  int query_idx =
      blockIdx.y * blockDim.y + threadIdx.y; // Query index (token position)
  int head_idx =
      blockIdx.x * blockDim.x + threadIdx.x; // Head index (head size)

  if (query_idx < seq_len && head_idx < head_size) {
    // Compute dot product of q (1, h) and KT (h, T)
    float dot_product = 0.0f;
    for (int i = 0; i < head_size; i++) {
      dot_product += q[i] * KT[i * seq_len + query_idx];
    }

    // Shared memory for max values and exponentials
    __shared__ float max_val;
    __shared__ float exp_values[SEQ];
    __shared__ float sum_exp;

    if (threadIdx.x == 0) {
      max_val = -INFINITY;
      sum_exp = 0.0f;
    }
    __syncthreads();

    // Find maximum value for numerical stability
    atomicMax(reinterpret_cast<int *>(&max_val), __float_as_int(dot_product));
    __syncthreads();

    // Calculate exponentials and sum for softmax
    float exp_val = expf(dot_product - max_val);
    exp_values[query_idx] = exp_val;
    __syncthreads();

    // Sum exponentials
    if (threadIdx.x == 0) {
      for (int i = 0; i < seq_len; i++) {
        sum_exp += exp_values[i];
      }
    }
    __syncthreads();

    // Compute softmax value
    float softmax_value = exp_values[query_idx] / sum_exp;

    // Compute the weighted sum of values
    float result = 0.0f;
    for (int i = 0; i < seq_len; i++) {
      result += softmax_value * V[i * head_size + head_idx];
    }

    // Store the result
    ans[query_idx * head_size + head_idx] = result;
  }
}
// This isnt much parallel at all but it works!
// and for now its enough!

// Host function to handle memory allocation and kernel launch
void normal_implementation() {
  float *q, *KT, *V, *d_q, *d_KT, *d_V, *d_ans;

  size_t q_size = sizeof(float) * HSIZE;         // [1, h]
  size_t KT_size = sizeof(float) * HSIZE * SEQ;  // [h, T]
  size_t V_size = sizeof(float) * SEQ * HSIZE;   // [T, h]
  size_t ans_size = sizeof(float) * SEQ * HSIZE; // [T, h]

  // Allocate host memory
  q = (float *)malloc(q_size);
  KT = (float *)malloc(KT_size);
  V = (float *)malloc(V_size);
  float *ans = (float *)malloc(ans_size);

  // Initialize host matrices
  __initArr(q, HSIZE);
  __initArr(KT, HSIZE * SEQ);
  __initArr(V, SEQ * HSIZE);

  // Allocate device memory
  cudaMalloc(&d_q, q_size);
  cudaMalloc(&d_KT, KT_size);
  cudaMalloc(&d_V, V_size);
  cudaMalloc(&d_ans, ans_size);

  // Copy data from host to device
  cudaMemcpy(d_q, q, q_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_KT, KT, KT_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, V, V_size, cudaMemcpyHostToDevice);

  // Set up grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((HSIZE + blockSize.x - 1) / blockSize.x,
                (SEQ + blockSize.y - 1) / blockSize.y);

  // Launch CUDA kernel
  flash_normal<<<gridSize, blockSize>>>(d_ans, d_q, d_KT, d_V, SEQ, HSIZE);
  cudaDeviceSynchronize();

  // Copy result from device to host
  cudaMemcpy(ans, d_ans, ans_size, cudaMemcpyDeviceToHost);

  // Print result
  printArr(ans, SEQ, HSIZE);

  // Free device memory
  cudaFree(d_q);
  cudaFree(d_KT);
  cudaFree(d_V);
  cudaFree(d_ans);

  // Free host memory
  free(q);
  free(KT);
  free(V);
  free(ans);
}

int main() {
  normal_implementation();
  return 0;
}
