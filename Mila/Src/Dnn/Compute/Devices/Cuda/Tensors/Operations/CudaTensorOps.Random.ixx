/**
 * @file CudaTensorOps.Random.ixx
 * @brief CUDA random initialization partition for tensor buffers.
 *
 * Provides device-dispatched normal and Xavier initialization using cuRAND.
 */

module;
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <type_traits>
#include <stdexcept>
#include <cmath>
#include <time.h>

export module Compute.CudaTensorOps:Random;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.DeviceType;
import Compute.DeviceTraits;
import Compute.DeviceId;
import Cuda.Helpers;
import Cuda.Error;

namespace Mila::Dnn::Compute::Cuda
{
    using namespace Mila::Dnn::Compute;

    class CudaExecutionContext;

    export struct RandomOps
    {
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>&& TensorDataTypeTraits<TDataType>::is_float_type
        static void fill_normal(
            Tensor<TDataType, TMemoryResource>& tensor,
            float mean,
            float stddev,
            IExecutionContext* exec_context = nullptr )
        {
            std::size_t n = tensor.size();
            if ( n == 0 ) 
                return;

            float* dst = reinterpret_cast<float*>(tensor.rawData());
            if ( !dst ) 
                return;

            DeviceId device_id = tensor.getDeviceId();
            cudaStream_t stream = cudaStreamDefault;
            if ( exec_context )
            {
                auto* cuda_ctx = cast_context_<DeviceType::Cuda>( exec_context );
                stream = cuda_ctx->getStream();
            }
            int dev_index = device_id.index;
            Cuda::setCurrentDevice( dev_index );

            curandGenerator_t gen;
            curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT );

            // REVIEW: How to best seed the generator for reproducibility? Options:
            // - Use a fixed seed (e.g. 1234ULL) for deterministic behavior across runs (good for testing)
            // - Use a time-based seed for variability across runs (good for production)
            // - Use std::random_device{}() for non-deterministic seed (good for maximum randomness, but may have performance implications)
            curandSetPseudoRandomGeneratorSeed( gen, 1234ULL );// FIXME: static_cast<unsigned long long>(std::time( nullptr )) );
            curandSetStream( gen, stream );

            curandStatus_t status = curandGenerateNormal( gen, dst, n, mean, stddev );
            if ( status != CURAND_STATUS_SUCCESS )
            {
                curandDestroyGenerator( gen );
                throw std::runtime_error( "curandGenerateNormal failed" );
            }

            curandDestroyGenerator( gen );

            if ( !exec_context )
            {
                cudaError_t syncErr = cudaStreamSynchronize( stream );
                if ( syncErr != cudaSuccess )
                {
                    throw std::runtime_error( std::string( "cudaStreamSynchronize failed: " ) + cudaGetErrorString( syncErr ) );
                }
            }
        }

        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>&& TensorDataTypeTraits<TDataType>::is_float_type
        static void fill_xavier_uniform(
            Tensor<TDataType, TMemoryResource>& tensor,
            std::size_t input_size,
            std::size_t output_size,
            IExecutionContext* exec_context = nullptr )
        {
            std::size_t n = tensor.size();
            if ( n == 0 ) return;

            float* dst = reinterpret_cast<float*>(tensor.rawData());
            if ( !dst ) return;

            DeviceId device_id = tensor.getDeviceId();
            cudaStream_t stream = cudaStreamDefault;
            if ( exec_context )
            {
                auto* cuda_ctx = cast_context_<DeviceType::Cuda>( exec_context );
                stream = cuda_ctx->getStream();
            }
            int dev_index = device_id.index;
            Cuda::setCurrentDevice( dev_index );

            float limit = std::sqrt( 6.0f / static_cast<float>(input_size + output_size) );

            curandGenerator_t gen;
            curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT );


            // REVIEW: How to best seed the generator for reproducibility? Options:
            // - Use a fixed seed (e.g. 1234ULL) for deterministic behavior across runs (good for testing)
            // - Use a time-based seed for variability across runs (good for production)
            // - Use std::random_device{}() for non-deterministic seed (good for maximum randomness, but may have performance implications)
            // 
            curandSetPseudoRandomGeneratorSeed( gen, 1234ULL ); // FIXME: static_cast<unsigned long long>(std::time( nullptr )) );
            curandSetStream( gen, stream );

            // Fill with uniform [0, 1)
            curandStatus_t status = curandGenerateUniform( gen, dst, n );
            if ( status != CURAND_STATUS_SUCCESS )
            {
                curandDestroyGenerator( gen );
                throw std::runtime_error( "curandGenerateUniform failed" );
            }

            // Scale and shift to [-limit, limit]
            // FIXME: scale_shift_kernel <<<(n + 255) / 256, 256, 0, stream >>>( dst, n, limit );

            curandDestroyGenerator( gen );

            if ( !exec_context )
            {
                cudaError_t syncErr = cudaStreamSynchronize( stream );
                if ( syncErr != cudaSuccess )
                {
                    throw std::runtime_error( std::string( "cudaStreamSynchronize failed: " ) + cudaGetErrorString( syncErr ) );
                }
            }
        }

    private:
        // Kernel to scale and shift uniform [0,1) to [-limit, limit]
        /*static __global__ void scale_shift_kernel( float* data, std::size_t n, float limit )
        {
            std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if ( idx < n )
            {
                data[ idx ] = (data[ idx ] * 2.0f - 1.0f) * limit;
            }
        }*/
    };
}