#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  // Get number of devices in your system. Something like nvidia-smi
  int devCount;
  cudaGetDeviceCount(&devCount); // update property

  cudaDeviceProp devProp;

  for (unsigned int i = 0; i < devCount; i++) {
    cudaGetDeviceProperties(&devProp, i);

    printf("Device %d:\n", i);
    printf("Name: %s\n", devProp.name);
    printf("Total Global Memory: %lu bytes\n", devProp.totalGlobalMem);
    printf("Shared Memory per Block: %lu bytes\n", devProp.sharedMemPerBlock);
    printf("Registers per Block: %d\n", devProp.regsPerBlock);
    printf("Warp Size: %d\n", devProp.warpSize);
    printf("Memory Pitch: %lu bytes\n", devProp.memPitch);
    printf("Max Threads per Block: %d\n", devProp.maxThreadsPerBlock);
    printf("Max Threads Dimension: %d x %d x %d\n", devProp.maxThreadsDim[0],
           devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("Max Grid Size: %d x %d x %d\n", devProp.maxGridSize[0],
           devProp.maxGridSize[1], devProp.maxGridSize[2]);
    printf("Clock Rate: %d kHz\n", devProp.clockRate);
    printf("Total Constant Memory: %lu bytes\n", devProp.totalConstMem);
    printf("Compute Capability: %d.%d\n", devProp.major, devProp.minor);
    printf("Texture Alignment: %lu bytes\n", devProp.textureAlignment);
    printf("Device Overlap: %d\n", devProp.deviceOverlap);
    printf("Multi-Processor Count: %d\n", devProp.multiProcessorCount);
    printf("Kernel Execution Timeout Enabled: %d\n",
           devProp.kernelExecTimeoutEnabled);
    printf("Integrated: %d\n", devProp.integrated);
    printf("Can Map Host Memory: %d\n", devProp.canMapHostMemory);
    printf("Compute Mode: %d\n", devProp.computeMode);
    printf("Concurrent Kernels: %d\n", devProp.concurrentKernels);
    printf("ECC Enabled: %d\n", devProp.ECCEnabled);
    printf("PCI Bus ID: %d\n", devProp.pciBusID);
    printf("PCI Device ID: %d\n", devProp.pciDeviceID);
    printf("PCI Domain ID: %d\n", devProp.pciDomainID);
    printf("Async Engine Count: %d\n", devProp.asyncEngineCount);
    printf("Unified Addressing: %d\n", devProp.unifiedAddressing);
    printf("Memory Clock Rate: %d kHz\n", devProp.memoryClockRate);
    printf("Memory Bus Width: %d bits\n", devProp.memoryBusWidth);
    printf("L2 Cache Size: %d bytes\n", devProp.l2CacheSize);
    printf("Max Threads per Multi-Processor: %d\n",
           devProp.maxThreadsPerMultiProcessor);
    printf("Stream Priorities Supported: %d\n",
           devProp.streamPrioritiesSupported);
    printf("Global L1 Cache Supported: %d\n", devProp.globalL1CacheSupported);
    printf("Local L1 Cache Supported: %d\n", devProp.localL1CacheSupported);
    printf("Max Shared Memory Per Multiprocessor: %lu bytes\n",
           devProp.sharedMemPerMultiprocessor);
    printf("Max Registers Per Multiprocessor: %d\n",
           devProp.regsPerMultiprocessor);
    printf("Managed Memory: %d\n", devProp.managedMemory);
    printf("Is Multi-GPU Board: %d\n", devProp.isMultiGpuBoard);
    printf("Multi-GPU Board Group ID: %d\n", devProp.multiGpuBoardGroupID);
    printf("Host Native Atomic Supported: %d\n",
           devProp.hostNativeAtomicSupported);
    printf("Cooperative Device Launch: %d\n", devProp.cooperativeLaunch);
    printf("Cooperative Multi-Device Launch: %d\n",
           devProp.cooperativeMultiDeviceLaunch);
    printf("Max Blocks per Multi-Processor: %d\n",
           devProp.maxBlocksPerMultiProcessor);
    printf(
        "----------------------------------------------------------------\n");
  }

  return 0;
}
