#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

// read tiled matmul before this!
// part of the motivation for flash attention was online softmax
// we will try to parallelize softmax in this file

// SOFTMAX has always been a 💢

// NOTE
// The attention matrix itself remains unrealized in the forward pass of
// flash attention and they get it for free while computing backwardpass .
// Recomputation is an integral part of flash attention but for now we will just
// focus on parallelizing softmax

// for my device :
// see how small the shared memory is !
// Device 0: NVIDIA GeForce RTX 4070 Laptop GPU
// Shared memory per block: 49152 bytes (12_288 floats) ~0.05 MB
// This memory still looks big enough but we will assume that the whole vector
// cannot fit inside the shared memory Number of multiprocessors: 36

#define N_ELEMENTS 2 * 12288

// great read! :
// https://github.com/ELS-RD/kernl/blob/main/tutorial/3%20-%20online%20softmax.ipynb
// most of the underneath comments are from the above notebook!

// In the case of transformer model, the softmax is applied to each row of a
// matrix of shape (sequence length, sequence length) and we apply on dim = -1
// and the SRAM limit for an // fp16 vector is around 128 tokens.

// most of the training is done on <= fp16 . so we defintely have to manage
// numerical stability of our softmax function

// normal softmax
// first the equation is :
// np.exp(z - max(z)) / np.sum(np.exp(z - max(z)))
// where z is logits

// Pytorch (eager) would need
// 1 pass to find the max of z
// 1 (2 if eager) pass to exponentiate and keep a running sum for the
// denominator 1 pass to broadcast the division with their new exponentiated
// values

// from the paper
// > . In this paper we propose a way to compute
// classical Softmax with fewer memory accesses and hypothesize that this
// reduction in memory accesses should improve Softmax performance on actual
// hardware. The benchmarks confirm this hypothesis: Softmax accelerates by up
// to 1.3x and Softmax+TopK combined and fused by up to 5x

// we fuse it by :
// calculating the exponentiation and the max logit in 1 pass.

// The achievement lies in the fact that you are supposed to know the maximum
// value of the vector to compute the denominator. At each step, our knowledge
// of the maximum value may evolve (we may meet a value bigger than our
// precedent maximum). When it happens, we just adjust the result of our
// computation of the precedent step.

// The adjustment would be on the fact that exp(a+b) = exp(a) * exp(b)

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

// ------ eager mode in cpu

void softmax_cpu(float *out, float *logits, int dim) {
  float max_logit = -1 * INFINITY;
  float normalization_term = 0;

  // one pass to find max
  for (int i = 0; i < dim; i++) {
    if (logits[i] > max_logit)
      max_logit = logits[i];
  }
  // printf("Max Logit was :%f\n", max_logit);

  // one pass to exponentiate and calculate denominator
  for (int i = 0; i < dim; i++) {
    logits[i] = exp((logits[i] - max_logit));
    normalization_term += logits[i];
  }

  // printf("Denominator was :%f\n", normalization_term);

  // one pass to broadcast
  for (int i = 0; i < dim; i++) {
    out[i] = logits[i] / normalization_term;
  }
}

void online_cpu_implementation(float *out, float *logits, int dim) {
  // we will do it 2 * O(n) by fusing finding max and sum of the denominator in
  // 1 pass.
  //  note your first instinct would be "isnt this for free ?" . i just keep a
  // running sum of the denominator and update the max value and at the end of
  // the loop i would just divide the whole denominator term by exp(max).
  // This is correct but this isnt safe as a unit to be added to the denominator
  // could be overflowed if it isnt atleast reduced by current max.

  // see attached notes for more details
  // ds[i+1] = ds[i][1/exp(max_i+1 - max_i)] + e(x_i+1 - max_i+1)
  float current_max = 0.0f;
  float past_max = 0.0f;
  float ds = 0.0f;
  for (int i = 0; i < dim; i++) {
    if (logits[i] > current_max) {
      past_max = current_max;
      current_max = logits[i];
    }
    ds = ds * (1 / exp(current_max - past_max)) + exp(logits[i] - current_max);
  }

  // and now just to get the denominator term
  for (int i = 0; i < dim; i++) {
    out[i] = exp(logits[i] - current_max) / ds;
  }
}

void normal_cpu_implementation() {
  float *some_arr;
  float *out_arr;
  int smalln_elements = 10;
  some_arr = (float *)malloc(sizeof(float) * smalln_elements);
  out_arr = (float *)malloc(sizeof(float) * smalln_elements);
  __initArr(some_arr, smalln_elements);
  softmax_cpu(out_arr, some_arr, smalln_elements);
  printArr(out_arr, smalln_elements);

  free(some_arr);
  free(out_arr);
}

void online_cpu_implementation() {
  float *some_arr;
  float *out_arr;
  int smalln_elements = 10;
  some_arr = (float *)malloc(sizeof(float) * smalln_elements);
  out_arr = (float *)malloc(sizeof(float) * smalln_elements);
  __initArr(some_arr, smalln_elements);
  softmax_cpu(out_arr, some_arr, smalln_elements);
  printArr(out_arr, smalln_elements);

  free(some_arr);
  free(out_arr);
}

// for normal gpu implementation (its only too easy .. but the idea was to
// safely update the denominator)
// 1 kernel to calcuate max and the sum (both redction operation : 1 kernel)
// 1 kernel to map to apply map operation to get the softmax of each element
// (1kernel) we cant optimize more than this .. but the key idea would now be
// used in flash attention

int main() {
  // code goes here
  // normal_cpu_implementation();
  online_cpu_implementation();
}
