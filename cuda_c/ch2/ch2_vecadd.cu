#include <stdio.h>

// from chapter 2

__global__ void vecAddKernel(float *a, float *b, float *c, int dim) {
  // each i unique to a thread
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < dim) {
    c[i] = a[i] + b[i];
    printf("Thread : %d , a:%f , b:%f , c%f\n", i, a[i], b[i], c[i]);
  } else {
    printf("Warning! Thread : %d did not have anything to do!\n", i);
  } // save from other invalid threads
}

void vecAdd(float *a_h, float *b_h, float *c_h, int dim) {
  // a , b , c are float vecotors each of size dim
  // the suffix _h is for host . and _d would mean device
  // we want to calculate c = a + b; in a vectorized way

  // init device variables . These would live in devices
  // although they were init in host
  float *a_d, *b_d, *c_d;

  // update their address to device.
  // to update addr . we would need the addr of the addr . so that it could be
  // deref and updated

  cudaError_t cuda_err1 = cudaMalloc((void **)&a_d, dim * sizeof(float));
  cudaError_t cuda_err2 = cudaMalloc((void **)&b_d, dim * sizeof(float));
  cudaError_t cuda_err3 = cudaMalloc((void **)&c_d, dim * sizeof(float));

  // now that they live on device .. dont try to deref these here in the host
  // scope

  // copy over the data from host to device
  cudaMemcpy(a_d, a_h, dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, dim * sizeof(float), cudaMemcpyHostToDevice);

  // now call the kernel to do the operation
  // 1 grid each with 1024 threads .. but we only need 5 threads !.
  // But thats ok! . we handle it in the kernel
  // vecAddKernel<<<1, 1024>>>(a_d, b_d, c_d, dim);

  // proper way something like:
  float num_threads = 4.0; // Ada lovelace : 1024
  float blocks = ceil((float)dim / num_threads);
  printf("Number of Grids : %.1f,Threads : %.1f\n", blocks, num_threads);
  printf("Number of Threads needed : %d\n", dim);

  // This bit is from ch3
  // we dont really need a 3 dimensional threads and blocks
  dim3 one_dim_blocks(blocks, 1, 1);
  dim3 one_dim_threads(num_threads, 1, 1);

  // call the kernel
  vecAddKernel<<<one_dim_blocks, one_dim_threads>>>(a_d, b_d, c_d, dim);

  // now fetch back the answer
  // init ans var.. this will live on the host
  cudaMemcpy(c_h, c_d, dim * sizeof(float), cudaMemcpyDeviceToHost);

  // free the device memory .. we dont need them anymore
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  // show me the answer : c_h
  printf("\nPrinting from the host!\n");
  for (int i = 0; i < dim; i++) {
    printf("%d:%f\n", i, *(c_h));
    c_h++;
  }
}

int main() {
  // easy example but theres a lot going on
  // init some vectors.
  float a[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
  float b[] = {-1.1, -2.2, 3.3, -4.4, -5.5, -6.6, -7.7};
  float c[7];

  vecAdd(a, b, c, sizeof(c) / sizeof(float));
}
