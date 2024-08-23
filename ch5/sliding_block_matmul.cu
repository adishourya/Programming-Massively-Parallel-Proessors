#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>

// we will go through Tiling Matrix multiplication
// but i like to call it sliding block matrix multiplication

// Matrix Size
#define N 8
#define TILE_WIDTH 2
// TILE WIDTH would just be blockdim at kernel launch configuration

// The idea is we will slide blocks through the rows and columns of the matrices
// and while slidining we will store in the elements into shared memory.
// and keep a running sum to calculate dot product

// for example with N=8 for both A & B. and TILE_WIDTH = 2
// Block(y=0,x=0) will slide from left to right of A for the first 2 rows
// and slide from top to bottom for B for the first 2 columns.
// which would calculate 4 dot products (top-left) of the resultant matrix

// ========= This code is all about these adjustments
// But other blocks (except for Block 0,0)
// Blocks(y=0,x=1) starts at the right end of A ;which needs
// to be adjusted and starts at top right of B which is ok!

// Block(y=1,x=0) starts at the bottom left of B ;which needs
// to be adjusted and starts at bottom left of A which is ok!

// Block(y=1,x=1) starts at the bottom right of both the matrices . <- which
// needs to be adjusted

// These adjustments would be more mangable if we only use local thread indices
// and move the block as if they started at 0,0 This would become more clear
// later on...

//---------------UTILS

// Initialize operand matrices
void __initMat(int *arr, int dim) {
  // puts in some random numbers b/n [0,10)
  for (int i = 0; i < dim; i++) {
    // arr[i] = rand() % 10;
    arr[i] = i;
  }
}

// to check if the calculation is correct
void checkAnswers(int *ans1, int *ans2, int dim) {
  int num_mismatches = 0;
  for (int i = 0; i < dim; i++) {
    if (ans1[i] != ans2[i]) {
      num_mismatches++;
    }
  }
  if (num_mismatches == 0) {
    printf("Equal Array😻\n");
  } else {
    printf("Found mismatches : %d🤦🏻‍♂️\n", num_mismatches);
  }
}

// if you wanna see them printed
void print_small_mat(int *C, int dim) {
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      printf("C[%d,%d]=%d\t", i, j, C[i * dim + j]);
    }
    printf("\n");
  }
}

//---------------GOLD

// gold :: so easy!!.. but tiling improves the speed quite a lot
// we need to go through tiling so we can understand flash attention🌶️
__global__ void normalKernel(int *C, int *A, int *B) {
  // this is to test against for accuracy and speed
  // each thread performs 1 dot product. no shared mem

  int thread_x_id = (blockDim.x * blockIdx.x) + threadIdx.x;
  int thread_y_id = (blockDim.y * blockIdx.y) + threadIdx.y;

  // C[x,y] = A[x,:] * B[:,y]
  for (int i = 0; i < N; i++) {
    C[(thread_y_id * N) + thread_x_id] +=
        A[(thread_y_id * N) + i] * B[thread_x_id + (i * N)];
  }
}

// normal code :: for gold calculation
void normal_code() {
  size_t arr_size = N * N * sizeof(int);
  printf("Arr size in bytes : %zu\n", arr_size);

  int *A_h, *B_h, *C_h;
  A_h = (int *)malloc(arr_size);
  B_h = (int *)malloc(arr_size);
  C_h = (int *)malloc(arr_size);

  // no need to init C . since we only have to write into C
  __initMat(A_h, N * N);
  __initMat(B_h, N * N);

  // make device variables
  int *A_d, *B_d, *C_d;

  cudaMalloc((void **)&A_d, arr_size);
  cudaMalloc((void **)&B_d, arr_size);
  cudaMalloc((void **)&C_d, arr_size);

  cudaMemcpy(A_d, A_h, arr_size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, arr_size, cudaMemcpyHostToDevice);

  dim3 threads_per_block = dim3(TILE_WIDTH, TILE_WIDTH, 1);
  int blocks = (int)ceil(N / TILE_WIDTH);
  dim3 blocks_needed = dim3(blocks, blocks, 1);

  normalKernel<<<blocks_needed, threads_per_block>>>(C_d, A_d, B_d);

  cudaMemcpy(C_h, C_d, arr_size, cudaMemcpyDeviceToHost);
  print_small_mat(C_h, N);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  free(A_h);
  free(B_h);
  free(C_h);
}

//---------------TILED

__global__ void tiledMatmulKernel(int *C, int *A, int *B) {
  // we will only use local indices to see the movement of the blocks more
  // clearly thread private variables
  int tile_x = threadIdx.x;
  int tile_y = threadIdx.y;

  // keep portion in the shared memory and override it when the tile shifts
  // these shared memory are private to the block and these wont be passed to
  // other functions so these could be ndarray
  __shared__ int portion_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ int portion_B[TILE_WIDTH][TILE_WIDTH];

  // tile movements
  // it should move left to right for A
  // and it should move top to bottom for B

  int running_dot = 0;
  for (int i = 0; i < N / TILE_WIDTH; i++) {
    // for collecting portion A
    // Down movement
    // if blockIdx != 0 then move that many tiles down. (blockIdx.y *
    // TILE_WIDTH) * N if local y != 0 then move that move that many rows down
    // (tile_y * N) Right movement we move right with iteration (i*TILE_WIDTH)
    // local right movement (tile_x)
    portion_A[tile_y][tile_x] = A[(blockIdx.y * TILE_WIDTH) * N + (tile_y)*N +
                                  (i * TILE_WIDTH) + tile_x];
    // for collecting portion B
    // Right movement
    // if blockIdx.x != 0 then shift that many tiles width to the right
    // (blockIdx.x * TILE_WIDTH) local right movement (tile_x) Down movement we
    // move down with iteration (i*TILE_WIDTH) * N local down movements (tile_y
    // * N)
    portion_B[tile_y][tile_x] = B[(blockIdx.x * TILE_WIDTH) + tile_x +
                                  (i * TILE_WIDTH) * N + (tile_y * N)];

    // now that we have collected Tiled regions of elements.. calculate running
    // dot product loop inside the tile
    for (int k = 0; k < TILE_WIDTH; k++) {
      running_dot += portion_A[tile_y][k] * portion_B[k][tile_x];
    }
  }

  // now that the loop is done assign the dot product to the DRAM
  // you can define the global id's now
  int thread_y_id = (blockDim.y * blockIdx.y) + threadIdx.y;
  int thread_x_id = (blockDim.x * blockIdx.x) + threadIdx.x;
  C[(thread_y_id * N) + thread_x_id] = running_dot;
}

// normal code :: for gold calculation
void tiled_code() {
  size_t arr_size = N * N * sizeof(int);
  printf("Arr size in bytes : %zu\n", arr_size);

  int *A_h, *B_h, *C_h;
  A_h = (int *)malloc(arr_size);
  B_h = (int *)malloc(arr_size);
  C_h = (int *)malloc(arr_size);

  // no need to init C . since we only have to write into C
  __initMat(A_h, N * N);
  __initMat(B_h, N * N);

  // make device variables
  int *A_d, *B_d, *C_d;

  cudaMalloc((void **)&A_d, arr_size);
  cudaMalloc((void **)&B_d, arr_size);
  cudaMalloc((void **)&C_d, arr_size);

  cudaMemcpy(A_d, A_h, arr_size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, arr_size, cudaMemcpyHostToDevice);

  dim3 threads_per_block = dim3(TILE_WIDTH, TILE_WIDTH, 1);
  int blocks = (int)ceil(N / TILE_WIDTH);
  dim3 blocks_needed = dim3(blocks, blocks, 1);

  tiledMatmulKernel<<<blocks_needed, threads_per_block>>>(C_d, A_d, B_d);

  cudaMemcpy(C_h, C_d, arr_size, cudaMemcpyDeviceToHost);
  print_small_mat(C_h, N);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  free(A_h);
  free(B_h);
  free(C_h);
}

int main() {
  normal_code();
  tiled_code();
  return 0;
}
