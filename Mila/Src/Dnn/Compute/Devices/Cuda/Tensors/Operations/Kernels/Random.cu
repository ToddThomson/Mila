#include "TensorOps.Random.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace Mila::Dnn::Compute::Cuda
{
    namespace
    {
        __global__ void scale_shift_kernel( float* data, size_t n, float min_val, float range )
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for ( ; idx < n; idx += stride )
            {
                data[ idx ] = data[ idx ] * range + min_val;
            }
        }
    }

    void launch_scale_shift( float* data, size_t n, float min_val, float max_val, cudaStream_t stream )
    {
        if ( n == 0 )
            return;

        float range = max_val - min_val;
        dim3 block( 256 );
        dim3 grid( static_cast<unsigned int>((n + 255) / 256) );

        scale_shift_kernel << <grid, block, 0, stream >> > (data, n, min_val, range);
    }
}