/**
 * @file GatedResidual.cu
 * @brief Kernels and host launchers for the gated residual's elementwise steps, FP32 and BF16.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "CudaUtils.h"
#include "GatedResidual.cuh"

namespace Mila::Dnn::Compute::Cuda::Residual
{
    namespace
    {
        constexpr int kBlockSize = 256;

        __device__ __forceinline__ float load( float value ) { return value; }
        __device__ __forceinline__ float load( __nv_bfloat16 value ) { return __bfloat162float( value ); }

        template<typename T>
        __device__ __forceinline__ T store( float value );

        template<>
        __device__ __forceinline__ float store<float>( float value ) { return value; }

        template<>
        __device__ __forceinline__ __nv_bfloat16 store<__nv_bfloat16>( float value ) { return __float2bfloat16( value ); }

        template<typename T>
        __global__ void activate_down_kernel( T* __restrict__ activated, const T* __restrict__ down, int count, float inverse_streams )
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;

            if ( i < count )
            {
                const float scaled = load( down[ i ] ) * inverse_streams;
                activated[ i ] = store<T>( scaled / ( 1.0f + expf( -scaled ) ) );
            }
        }

        // One thread per (row, h); the streams are summed in index order, as the CPU reference does.
        template<typename T>
        __global__ void mix_kernel( T* __restrict__ mixed, const T* __restrict__ normed, const T* __restrict__ up,
            int rows, int streams, int model_dim, float inverse_streams )
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;

            if ( i >= rows * model_dim )
            {
                return;
            }

            const int row = i / model_dim;
            const int h = i % model_dim;
            const int base = row * streams * model_dim + h;

            float sum = 0.0f;

            for ( int stream_index = 0; stream_index < streams; ++stream_index )
            {
                const int at = base + stream_index * model_dim;
                sum += load( normed[ at ] ) / ( 1.0f + expf( -load( up[ at ] ) ) );
            }

            mixed[ i ] = store<T>( sum * inverse_streams );
        }

        template<typename T>
        __global__ void injection_weights_kernel( T* __restrict__ weights, const T* __restrict__ inject, int count, float inverse_streams )
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;

            if ( i < count )
            {
                weights[ i ] = store<T>( 2.0f / ( 1.0f + expf( -load( inject[ i ] ) * inverse_streams ) ) );
            }
        }

        template<typename T>
        __global__ void inject_kernel( T* __restrict__ output, const T* __restrict__ residual, const T* __restrict__ weights,
            const T* __restrict__ sublayer_output, int rows, int streams, int model_dim )
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;

            if ( i >= rows * streams * model_dim )
            {
                return;
            }

            const int h = i % model_dim;
            const int row_stream = i / model_dim;
            const int row = row_stream / streams;

            output[ i ] = store<T>( load( residual[ i ] ) + load( sublayer_output[ row * model_dim + h ] ) * load( weights[ row_stream ] ) );
        }

        template<typename T>
        void launchActivateDown( T* activated, const T* down, int count, int streams, cudaStream_t stream )
        {
            if ( count > 0 )
            {
                activate_down_kernel<T><<<ceil_div( count, kBlockSize ), kBlockSize, 0, stream>>>(
                    activated, down, count, 1.0f / static_cast<float>( streams ) );
            }

            cudaCheck( cudaGetLastError() );
        }

        template<typename T>
        void launchMix( T* mixed, const T* normed, const T* up, int rows, int streams, int model_dim, cudaStream_t stream )
        {
            const int count = rows * model_dim;

            if ( count > 0 )
            {
                mix_kernel<T><<<ceil_div( count, kBlockSize ), kBlockSize, 0, stream>>>(
                    mixed, normed, up, rows, streams, model_dim, 1.0f / static_cast<float>( streams ) );
            }

            cudaCheck( cudaGetLastError() );
        }

        template<typename T>
        void launchInjectionWeights( T* weights, const T* inject, int count, int streams, cudaStream_t stream )
        {
            if ( count > 0 )
            {
                injection_weights_kernel<T><<<ceil_div( count, kBlockSize ), kBlockSize, 0, stream>>>(
                    weights, inject, count, 1.0f / static_cast<float>( streams ) );
            }

            cudaCheck( cudaGetLastError() );
        }

        template<typename T>
        void launchInject( T* output, const T* residual, const T* weights, const T* sublayer_output,
            int rows, int streams, int model_dim, cudaStream_t stream )
        {
            const int count = rows * streams * model_dim;

            if ( count > 0 )
            {
                inject_kernel<T><<<ceil_div( count, kBlockSize ), kBlockSize, 0, stream>>>(
                    output, residual, weights, sublayer_output, rows, streams, model_dim );
            }

            cudaCheck( cudaGetLastError() );
        }
    }

    void cuda_gated_residual_activate_down( float* activated, const float* down, int count, int streams, cudaStream_t stream )
    {
        launchActivateDown( activated, down, count, streams, stream );
    }

    void cuda_gated_residual_activate_down( __nv_bfloat16* activated, const __nv_bfloat16* down, int count, int streams, cudaStream_t stream )
    {
        launchActivateDown( activated, down, count, streams, stream );
    }

    void cuda_gated_residual_mix( float* mixed, const float* normed, const float* up,
        int rows, int streams, int model_dim, cudaStream_t stream )
    {
        launchMix( mixed, normed, up, rows, streams, model_dim, stream );
    }

    void cuda_gated_residual_mix( __nv_bfloat16* mixed, const __nv_bfloat16* normed, const __nv_bfloat16* up,
        int rows, int streams, int model_dim, cudaStream_t stream )
    {
        launchMix( mixed, normed, up, rows, streams, model_dim, stream );
    }

    void cuda_gated_residual_injection_weights( float* weights, const float* inject, int count, int streams, cudaStream_t stream )
    {
        launchInjectionWeights( weights, inject, count, streams, stream );
    }

    void cuda_gated_residual_injection_weights( __nv_bfloat16* weights, const __nv_bfloat16* inject, int count, int streams, cudaStream_t stream )
    {
        launchInjectionWeights( weights, inject, count, streams, stream );
    }

    void cuda_gated_residual_inject( float* output, const float* residual, const float* weights, const float* sublayer_output,
        int rows, int streams, int model_dim, cudaStream_t stream )
    {
        launchInject( output, residual, weights, sublayer_output, rows, streams, model_dim, stream );
    }

    void cuda_gated_residual_inject( __nv_bfloat16* output, const __nv_bfloat16* residual, const __nv_bfloat16* weights,
        const __nv_bfloat16* sublayer_output, int rows, int streams, int model_dim, cudaStream_t stream )
    {
        launchInject( output, residual, weights, sublayer_output, rows, streams, model_dim, stream );
    }
}
