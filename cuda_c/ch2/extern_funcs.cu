// Foreign function example: multiplication of a pair of floats
__global__ void vecAddKernel(float *a, float *b, float *c, int dim) {
  // each i unique to a thread
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < dim) {
    c[i] = a[i] + b[i];
  } else {
    printf("Warning! Thread : %d did not have anything to do!\n", i);
  } // save from other invalid threads
}
