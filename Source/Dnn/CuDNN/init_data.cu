#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fp16_emu.h" 
#include "init_data.h"

template void InitGPUData<half1>( half1* data, int numElements, half1 value );
template void InitGPUData<float>( float* data, int numElements, float value );
template void InitGPUData<double>( double* data, int numElements, double value );

// Kernel and launcher to initialize GPU data to some constant value

template <typename T_ELEM>
__global__
void InitGPUData_Kernel(T_ELEM *data, int numElements, T_ELEM value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements) {
        data[tid] = value;
    }
}

template <typename T_ELEM>
void InitGPUData(T_ELEM *data, int numElements, T_ELEM value) {
    dim3 gridDim;
    dim3 blockDim;

    blockDim.x = 1024;
    gridDim.x  = (numElements + blockDim.x - 1) / blockDim.x;

    InitGPUData_Kernel<<<gridDim, blockDim>>>(data, numElements, value);
}


