/**
 * @file CudaTensorOps.Random.ixx
 * @brief CUDA random initialization partition for tensor buffers.
 *
 * Provides device-dispatched normal and uniform random initialization using cuRAND.
 * Reuses the cached cuRAND generator from CudaExecutionContext when provided,
 * avoiding per-call generator creation overhead.
 */

module;
#include <cuda_runtime.h>
#include <curand.h>
#include <stdexcept>
#include <string>
#include "Kernels/Random.h"

export module Compute.CudaTensorOps:Random;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.DeviceType;
import Compute.DeviceTraits;
import Compute.DeviceId;
import Core.RandomGenerator;
import Cuda.Helpers;
import Cuda.Error;

namespace Mila::Dnn::Compute::Cuda
{
    using namespace Mila::Dnn::Compute;

    export struct RandomOps
    {
        /**
         * @brief Fill a float tensor with values drawn from N(mean, stddev^2) using cuRAND.
         *
         * When exec_context is provided, the cached generator from CudaExecutionContext
         * is reused. Without a context, a temporary generator is created and destroyed
         * per call, seeded from Core::RandomGenerator.
         *
         * curandGenerateNormal requires an even element count (Box-Muller pairs). If the
         * tensor has an odd element count, a temporary device buffer of size n+1 is
         * allocated, the full even count is generated into it, and n elements are copied
         * to the tensor. This is zero-overhead for even-sized tensors.
         *
         * @tparam TDataType Floating-point tensor data type.
         * @tparam TMemoryResource CUDA memory resource type.
         * @param tensor Destination tensor (pre-allocated).
         * @param mean Mean of the normal distribution.
         * @param stddev Standard deviation of the normal distribution.
         * @param exec_context Optional execution context for stream and generator reuse (borrowed, not owned).
         *
         * @note Only FP32 native tensors are currently supported. FP16/BF16 require a
         *       temporary float buffer with a subsequent conversion pass (not yet implemented).
         * @throws std::runtime_error On cuRAND or CUDA failure.
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource> && TensorDataTypeTraits<TDataType>::is_float_type
        static void fill_normal(
            Tensor<TDataType, TMemoryResource>& tensor,
            float mean,
            float stddev,
            IExecutionContext* exec_context = nullptr )
        {
            size_t n = tensor.size();
            if ( n == 0 )
                return;

            void* raw_dst = static_cast<ITensor&>(tensor).rawData();
            if ( !raw_dst )
                return;

            cudaStream_t stream = nullptr;
            bool needs_sync = false;
            curandGenerator_t gen = nullptr;
            bool owns_gen = false;

            if ( exec_context )
            {
                auto* cuda_ctx = cast_context_<DeviceType::Cuda>( exec_context );
                stream = cuda_ctx->getStream();
                gen = cuda_ctx->getCurandGenerator();
            }
            else
            {
                DeviceId device_id = tensor.getDeviceId();
                Cuda::setCurrentDevice( device_id.index );
                needs_sync = true;
                gen = make_temp_generator_( nullptr );
                owns_gen = true;
            }

            // curandGenerateNormal requires an even count (Box-Muller pairs).
            size_t gen_count = n + (n & 1);
            float* gen_dst = static_cast<float*>(raw_dst);
            float* temp_buf = nullptr;

            if ( n & 1 )
            {
                cudaError_t err = cudaMalloc( &temp_buf, gen_count * sizeof(float) );

                if ( err != cudaSuccess )
                {
                    if ( owns_gen ) curandDestroyGenerator( gen );
                    throw std::runtime_error( "cudaMalloc failed for normal distribution padding buffer" );
                }

                gen_dst = temp_buf;
            }

            curandStatus_t status = curandGenerateNormal( gen, gen_dst, gen_count, mean, stddev );

            if ( status != CURAND_STATUS_SUCCESS )
            {
                if ( temp_buf ) cudaFree( temp_buf );
                if ( owns_gen ) curandDestroyGenerator( gen );
                throw std::runtime_error( "curandGenerateNormal failed" );
            }

            if ( temp_buf )
            {
                cudaMemcpyAsync( raw_dst, temp_buf, n * sizeof(float), cudaMemcpyDeviceToDevice, stream );
                cudaFree( temp_buf );
            }

            if ( owns_gen )
                curandDestroyGenerator( gen );

            if ( needs_sync )
            {
                cudaError_t err = cudaStreamSynchronize( stream );

                if ( err != cudaSuccess )
                    throw std::runtime_error( std::string( "cudaStreamSynchronize failed: " ) + cudaGetErrorString( err ) );
            }
        }

        /**
         * @brief Fill a float tensor with uniform values in [min_val, max_val) using cuRAND.
         *
         * cuRAND generates values in [0, 1), which are scaled and shifted to [min_val, max_val)
         * by a device kernel. When exec_context is provided, the cached generator is reused.
         * Without a context, a temporary generator is created and destroyed per call.
         *
         * @tparam TDataType Floating-point tensor data type.
         * @tparam TMemoryResource CUDA memory resource type.
         * @param tensor Destination tensor (pre-allocated).
         * @param min_val Lower bound of the uniform range (inclusive).
         * @param max_val Upper bound of the uniform range (exclusive).
         * @param exec_context Optional execution context for stream and generator reuse (borrowed, not owned).
         *
         * @note Only FP32 native tensors are currently supported.
         * @throws std::runtime_error On cuRAND or CUDA failure.
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource> && TensorDataTypeTraits<TDataType>::is_float_type
        static void fill_uniform(
            Tensor<TDataType, TMemoryResource>& tensor,
            float min_val,
            float max_val,
            IExecutionContext* exec_context = nullptr )
        {
            size_t n = tensor.size();
            if ( n == 0 )
                return;

            void* raw_dst = static_cast<ITensor&>(tensor).rawData();
            if ( !raw_dst )
                return;

            cudaStream_t stream = nullptr;
            bool needs_sync = false;
            curandGenerator_t gen = nullptr;
            bool owns_gen = false;

            if ( exec_context )
            {
                auto* cuda_ctx = cast_context_<DeviceType::Cuda>( exec_context );
                stream = cuda_ctx->getStream();
                gen = cuda_ctx->getCurandGenerator();
            }
            else
            {
                DeviceId device_id = tensor.getDeviceId();
                Cuda::setCurrentDevice( device_id.index );
                needs_sync = true;
                gen = make_temp_generator_( nullptr );
                owns_gen = true;
            }

            float* dst = static_cast<float*>( raw_dst );

            curandStatus_t status = curandGenerateUniform( gen, dst, n );

            if ( status != CURAND_STATUS_SUCCESS )
            {
                if ( owns_gen ) curandDestroyGenerator( gen );
                throw std::runtime_error( "curandGenerateUniform failed" );
            }

            // Scale generated [0, 1) values to [min_val, max_val).
            Cuda::launch_scale_shift( dst, n, min_val, max_val, stream );

            if ( owns_gen )
                curandDestroyGenerator( gen );

            if ( needs_sync )
            {
                cudaError_t err = cudaStreamSynchronize( stream );

                if ( err != cudaSuccess )
                    throw std::runtime_error( std::string( "cudaStreamSynchronize failed: " ) + cudaGetErrorString( err ) );
            }
        }

    private:
        // Creates and seeds a temporary cuRAND generator bound to the given stream.
        // Caller owns the returned generator and must call curandDestroyGenerator on it.
        static curandGenerator_t make_temp_generator_( cudaStream_t stream )
        {
            curandGenerator_t gen = nullptr;
            curandStatus_t status = curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT );

            if ( status != CURAND_STATUS_SUCCESS )
                throw std::runtime_error( "Failed to create cuRAND generator" );

            auto seed = static_cast<unsigned long long>( Core::RandomGenerator::getInstance().getSeed() );
            status = curandSetPseudoRandomGeneratorSeed( gen, seed );

            if ( status != CURAND_STATUS_SUCCESS )
            {
                curandDestroyGenerator( gen );
                throw std::runtime_error( "Failed to seed cuRAND generator" );
            }

            status = curandSetStream( gen, stream );

            if ( status != CURAND_STATUS_SUCCESS )
            {
                curandDestroyGenerator( gen );
                throw std::runtime_error( "Failed to bind cuRAND generator to stream" );
            }

            return gen;
        }
    };
}