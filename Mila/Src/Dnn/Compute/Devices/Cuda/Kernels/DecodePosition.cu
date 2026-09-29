// DecodePosition.cu

#include <cuda_runtime.h>
#include "CudaUtils.h"
#include "DecodePosition.cuh"

namespace Mila::Dnn::Compute::Cuda
{
    namespace
    {
        __global__ void set_decode_position_kernel( int* target, int position )
        {
            *target = position;
        }
    }

    void cuda_set_decode_position( int* target, int position, cudaStream_t stream )
    {
        set_decode_position_kernel<<<1, 1, 0, stream>>>( target, position );

        cudaCheck( cudaGetLastError() );
    }
}
