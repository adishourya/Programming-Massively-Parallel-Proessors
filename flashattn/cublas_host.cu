#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// But cublas can only be called from host code.
// But you should still know how to do this though!
int main() {
  // Initialize cuBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Define the size of the vectors
  const int N = 5;

  // Allocate host memory for the vectors
  float *h_A = (float *)malloc(N * sizeof(float));
  float *h_B = (float *)malloc(N * sizeof(float));

  // Initialize the host vectors
  for (int i = 0; i < N; i++) {
    h_A[i] = i + 1.0f; // Example: [1.0, 2.0, 3.0, 4.0, 5.0]
    h_B[i] = i + 2.0f; // Example: [2.0, 3.0, 4.0, 5.0, 6.0]
  }

  // Allocate device memory for the vectors
  float *d_A, *d_B;
  cudaMalloc((void **)&d_A, N * sizeof(float));
  cudaMalloc((void **)&d_B, N * sizeof(float));

  // Copy vectors from host memory to device memory
  cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

  // Compute the dot product using cuBLAS
  float result = 0.0f;
  cublasSdot(handle, N, d_A, 1, d_B, 1, &result);

  // Print the result
  printf("Dot product: %f\n", result);

  // Clean up resources
  cudaFree(d_A);
  cudaFree(d_B);
  free(h_A);
  free(h_B);
  cublasDestroy(handle);

  return 0;
}
