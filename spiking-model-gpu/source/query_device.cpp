// This program shows how to get a number of device properties from API
// calls in CUDA.
// By: Nick from CoffeeBeforeArch

#include <iostream>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void Playground()
{
    cudaDeviceProp prop;
    int device;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 9;
    prop.minor = 1;
    gpuErrchk(cudaChooseDevice(&device, &prop));
    cout << "Device closest to revision " << prop.major << "." << prop.minor << " is " << device << "\n";
}

int main()
{
    Playground();

    // We can get the number of devices in the system.
    int device_count;
    gpuErrchk( cudaGetDeviceCount(&device_count) );
    cout << "There are " << device_count << " GPU(s) in the system\n";

    for (int i = 0; i < device_count; i++)
    {
        // We can set the device if we have multiple GPUs.
        cudaSetDevice(i);

        // We can also get the properties from the GPU.
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);
        cout << "Device " << i << " is a " << device_prop.name << "\n";

        // We can also get information about the driver and runtime.
        int driver;
        int runtime;
        cudaDriverGetVersion(&driver);
        cudaRuntimeGetVersion(&runtime);
        cout << "Driver: " << driver << " Runtime: " << runtime << "\n";

        // We can compare this against the device capabilities.
        cout << "CUDA capability: " << device_prop.major << "." << device_prop.minor << "\n";

        // We can also get the amount of global memory.
        cout << "Global memory in GB: " << device_prop.totalGlobalMem / (1 << 30) << "\n";

        // The number of SMs.
        cout << "Number of SMs: " << device_prop.multiProcessorCount << "\n";

        // The warp size in threads.
        cout << "Warp size: " << device_prop.warpSize << " threads\n";

        // Maximum pitch in bytes allowed by memory copies.
        cout << "Maximum pitch allowed by memory copies: " << device_prop.memPitch << " bytes\n";

        // Maximum number of threads per block.
        cout << "Maximum number of threads per block: " << device_prop.maxThreadsPerBlock << "\n";

        // Maximum size of each dimension of a block
        cout << "Maximum size of each dimension of a block: [" << device_prop.maxThreadsDim[0] << "," << device_prop.maxThreadsDim[1] << "," << device_prop.maxThreadsDim[2] << "]\n";

        // Maximum size of each dimension of a grid.
        cout << "Maximum size of each dimension of a grid: [" << device_prop.maxGridSize[0] << "," << device_prop.maxGridSize[1] << "," << device_prop.maxGridSize[2] << "]\n";

        // The frequency.
        cout << "Max clock rate: " << device_prop.clockRate * 1e-6 << "GHz\n";

        // The L2 cache size.
        cout << "The L2 cache size in MB: " << device_prop.l2CacheSize / (1 << 20) << "\n";

        // The shared memoryy per block.
        cout << "Total shared memory per block in KB: " << device_prop.sharedMemPerBlock / (1 << 10) << "\n";
    }

    return 0;
}