#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

#define SEQ 10
#define HSIZE 10

// we will start with untiled code first
//----------- UTILS
void __initArr(float *arr, int dim) {
  for (int i = 0; i < dim; i++) {
    // arr[i] = (float)i;
    arr[i] = 1.0f;
  }
}

void printArr(float *arr, int dim) {
  for (int i = 0; i < dim; i++) {
    printf("arr[%d]=%f\t", i, arr[i]);
  }
}

__global__ void flash_normal(float *ans, float *q, float *KT, float *V) {
  int thread_x_id =
      (blockDim.x * blockIdx.x) + threadIdx.x; // as many x threads as head_size
  int thread_y_id = (blockDim.y * blockIdx.y) +
                    threadIdx.y; // as many y threads as tokens (seq_len)

  // lets make each thread do 1 dot product. and we will store the x vector in
  // the shared memory x = q @ KT (1,h) @ (h,t) = (1,t) we dont need to
  // materialize the whole attention matrix
  float x = 0.0f;
  // maximums
  float curr_max = 0.0f;
  float prev_max = 0.0f;
  // denominator
  float curr_den = 0.0f;
  float prev_den = 0.0f;
  // float out
  float out = 0.0f;
  // __shared__ float x[SEQ];
  for (int i = 0; i < SEQ; i++) {
    // x = dot(q , K[:,i]);
    if (x > curr_max) {
      prev_max = curr_max;
      curr_max = x;
    }
    prev_den = curr_den;
    curr_den = prev_den * (1 / exp(curr_max - prev_max)) + exp(x - curr_max);
    out = out * (prev_den / curr_den) * (1 / exp(curr_max - prev_max)) +
          (exp(x - curr_max) / curr_den) * V [i, :]
  }
}

// all the heads,bathces can be calculated parallely
// all the oj (j corresponds to query) can be done parallely
// so the smallest unit we need to fuse .. so this will mostly look sequential
// but we will do it with fewer in and out from hbm
void normal_implementation() {
  float *q;
  float *KT; // we will assume we already have transposed Keys Matrix
  float *V;

  // we assume Q = X(Wq) = B,T,d @ d,h = B,T,h

  size_t q_size = sizeof(float) * 1 * HSIZE;    // [1,h]
  size_t KT_size = sizeof(float) * HSIZE * SEQ; //[h,T]
  size_t V_size = sizeof(float) * SEQ * HSIZE;  //[T,h]

  q = (float *)malloc(q_size);
  KT = (float *)malloc(KT_size);
  V = (float *)malloc(V_size);

  free(q);
  free(KT);
  free(V);
}

int main() {
  //
}
