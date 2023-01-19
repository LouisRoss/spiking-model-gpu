#include "cuda.h"

#if CUDA_VERSION < 12000
#pragma message ("Using CUDA kernel for version 11")
#include "NeuronModelGpu11.cu"
#else
#pragma message ("Using CUDA kernel for version 12")
#include "NeuronModelGpu12.cu"
#endif

