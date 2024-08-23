#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ARRSIZE 1 << 25
// note the arr size has to be comically large to understand this example
// more importantly should need more than a handful of blocks to see the
// improvement in speed and truly understand the method

// we will start with a simple execution first and then later see to get
// speed-ups with shared memory techniques

#define THREADS_PER_BLOCK 1024
// we will calculate number of blocks needed later on!

void initalizeLargeArr(int *arr, int val) {
  for (int i = 0; i < ARRSIZE; i++) {
    arr[i] = val;
  }
}

void printLargeArr(int *arr) {
  // prints the first and last 2 places of the arr
  printf("%d,%d....%d,%d\n", arr[0], arr[1], arr[ARRSIZE - 2],
         arr[ARRSIZE - 1]);
}

void printarr(int *arr, int dim) {
  for (int i = 0; i < dim; i++) {
    printf("arr[%d]=%d\n", i, arr[i]);
  }
}

__global__ void sumReduceKernel(int *out_arr, int *arr) {
  // this arr is too big to fit in one single block.
  // and we cant communicate between blocks
  // so each block will reduce its portion of the arr.
  // and then we will give out a scalar (reduced sum) from each block
  // for the final reduction to happen in the cpu. we do final redcution in the
  // cpu instead of invoking another gpu kernel because it would probably need
  // very few threads . and gpu programming for such cases is not efficient

  // actual position in the original arr
  int glob_threadid = (blockDim.x * blockIdx.x) + threadIdx.x;
  // local to block
  int loc_threadid = threadIdx.x;

  // so each block would process only tot_threads portion of arr
  int tot_threads = blockDim.x;

  // now we will divide each block into 2 halves.
  // and then sum up their corresponding elements until there is a scalar left
  // we will keep storing the reduction of each step in the left half.

  for (int i = tot_threads / 2; i > 0; i = i >> 1) {
    // guard from summing up the right half of the first block with the left
    // half of the second block.
    // yeah that means most of the threads would go idle...
    if (loc_threadid < i) {
      // if local threadid is less than half . then global thread id would also
      // be in the left side of the alotted array
      arr[glob_threadid] += arr[glob_threadid + i];
    }
    __syncthreads();
  }

  // make 1 thread from each block write down its reduction
  if (loc_threadid == 0) {
    out_arr[blockIdx.x] = arr[glob_threadid];
  }
}

__global__ void sumReduceKernlShared(int *out_arr, int *arr) {
  // to understand the logic . read the other func

  __shared__ int portion[THREADS_PER_BLOCK];

  int glob_threadid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int loc_threadid = threadIdx.x;

  // so each block would process only tot_threads portion of arr
  int tot_threads = blockDim.x;

  // store in the portion . shared memory as it is lot faster
  // guard for the portion in the final block. if it is not completely divisible
  if (glob_threadid < (ARRSIZE)) {
    portion[loc_threadid] = arr[glob_threadid];
  } else {
    portion[loc_threadid] = 0;
  }
  __syncthreads();
  // wait till every thread in the block has written to the shared memory

  // now instead of dram . read and write from and to the shared memory
  for (int i = tot_threads / 2; i > 0; i = i >> 1) {
    if (loc_threadid < i) {
      portion[loc_threadid] += portion[loc_threadid + i];
    }
    __syncthreads();
  }

  // now finally write back to the dram
  if (loc_threadid == 0) {
    out_arr[blockIdx.x] = portion[0];
  }
}

void final_sum_cpu(int *arr, int dim) {
  int sum = 0;
  for (int i = 0; i < dim; i++) {
    sum += arr[i];
  }
  printf("\n=====================\n");
  printf("Final Reduction :%d , arr size:%d\n", sum, ARRSIZE);
  printf("=====================\n");
}

int code1() {
  // keep the array in heap . since its too large
  int *large_arr_h;
  large_arr_h = (int *)malloc(sizeof(int) * ARRSIZE);

  // init the large arr
  initalizeLargeArr(large_arr_h, 1);
  // print large arr
  printLargeArr(large_arr_h);

  // since max threads per block allowed for modern hardware is 1024.
  // we will definetely need more than 1 block

  int num_blocks = (((ARRSIZE) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  // int num_blocks = ceil((ARRSIZE) / THREADS_PER_BLOCK); // <- same thing
  printf("Blocks needed : %d , each with %d threads!", num_blocks,
         THREADS_PER_BLOCK);

  int *out_blk_d;
  int *input_d;

  cudaMalloc((void **)&input_d, sizeof(int) * ARRSIZE);
  cudaMalloc((void **)&out_blk_d, sizeof(int) * num_blocks);

  cudaMemcpy(input_d, large_arr_h, sizeof(int) * ARRSIZE,
             cudaMemcpyHostToDevice);

  sumReduceKernel<<<num_blocks, THREADS_PER_BLOCK>>>(out_blk_d, input_d);
  int out_arr_h[num_blocks];
  cudaMemcpy(out_arr_h, out_blk_d, sizeof(int) * num_blocks,
             cudaMemcpyDeviceToHost);

  // print out the reduction from each block
  // for (int i = 0; i < num_blocks; i++) {
  //   printf("%d\t", out_arr_h[i]);
  // }

  final_sum_cpu(out_arr_h, num_blocks);

  cudaFree(input_d);
  cudaFree(out_blk_d);
  free(large_arr_h);
  return 0;
}

int code2() {

  // keep the array in heap . since its too large
  int *large_arr_h;
  large_arr_h = (int *)malloc(sizeof(int) * ARRSIZE);

  // init the large arr
  initalizeLargeArr(large_arr_h, 1);
  // print large arr
  printLargeArr(large_arr_h);

  int num_blocks = (((ARRSIZE) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  // int num_blocks = ceil((ARRSIZE) / THREADS_PER_BLOCK); // <- same thing
  printf("Blocks needed : %d , each with %d threads!\n", num_blocks,
         THREADS_PER_BLOCK);

  int *out_blk_d;
  int *input_d;

  cudaMalloc((void **)&input_d, sizeof(int) * ARRSIZE);
  cudaMalloc((void **)&out_blk_d, sizeof(int) * num_blocks);

  cudaMemcpy(input_d, large_arr_h, sizeof(int) * ARRSIZE,
             cudaMemcpyHostToDevice);

  // each block just needs to store the portion of the array.
  // and the portion size is threads per block
  int memory_shared = THREADS_PER_BLOCK * sizeof(int);
  sumReduceKernlShared<<<num_blocks, THREADS_PER_BLOCK, memory_shared>>>(
      out_blk_d, input_d);
  int out_arr_h[num_blocks];
  cudaMemcpy(out_arr_h, out_blk_d, sizeof(int) * num_blocks,
             cudaMemcpyDeviceToHost);

  // print out the reduction from each block
  // for (int i = 0; i < num_blocks; i++) {
  //   printf("%d\t", out_arr_h[i]);
  // }

  final_sum_cpu(out_arr_h, num_blocks);

  cudaFree(input_d);
  cudaFree(out_blk_d);
  free(large_arr_h);
  return 0;
}

void cpuImplementaion() {
  // with divide and conquer
  int *large_arr;
  large_arr = (int *)malloc(sizeof(int) * ARRSIZE);
  if (large_arr == NULL) {
    printf("couldnt allocate!\n");
    exit(EXIT_FAILURE);
  }

  initalizeLargeArr(large_arr, 1);

  for (int i = (ARRSIZE) / 2; i > 0; i = i >> 1) {
    for (int j = 0; j < i; j++) {
      large_arr[j] += large_arr[j + i];
    }
  }
  printf("Reduced Sum :%d\n", large_arr[0]);
  free(large_arr);
}

int main() {
  // code1();
  code2();
  // cpuImplementaion();
  return 0;
}
