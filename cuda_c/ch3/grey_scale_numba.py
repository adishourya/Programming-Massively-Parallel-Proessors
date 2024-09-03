# %%
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# %%
# Create a float32 array (since the kernel expects float* a)
a_h = np.linspace(-1, 1, 16, dtype=np.float32)

# Allocate memory on the device (old way)
a_d = cuda.mem_alloc(a_h.nbytes)  # manually calculating memory size

# Alternatively (newer, more convenient way) -> `cuda.In` handles allocation and copy
# a_d = cuda.In(a_h)

# %%
# Copy memory from host to device
cuda.memcpy_htod(a_d, a_h)  # Use the traditional way to copy from host to device

# %%
# Define kernel code
mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (idx < 16) // ensure that we are within bounds
      a[idx] *= 2;
  }
  """)

# Get the function from the compiled CUDA module
func = mod.get_function("doublify")

# Launch the kernel, passing the device memory as argument
func(a_d, block=(16,1,1), grid=(1,1))

# %%
# Create an empty array to store results
a_doubled = np.empty_like(a_h)

# Copy the result from device to host
cuda.memcpy_dtoh(a_doubled, a_d)

# Print the original and doubled arrays
print(a_h, "\n", a_doubled)
