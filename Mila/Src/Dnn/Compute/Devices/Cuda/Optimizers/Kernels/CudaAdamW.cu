#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "../../Helpers/CudaUtils.h"

#ifndef NDEBUG
#  include <cassert>
#  define KERNEL_ASSERT(cond) assert(cond)
#else
#  define KERNEL_ASSERT(cond) ((void)0)
#endif


namespace Mila::Dnn::Compute
{
    // ============================================================================
    // Simple Type Conversion Helpers
    // ============================================================================

    __device__ __forceinline__ void stochastic_rounding( float in, float* out, unsigned int seed )
    {
        *out = in;  // FP32: direct assignment
    }

    __device__ __forceinline__ void stochastic_rounding( float in, __nv_bfloat16* out, unsigned int seed )
    {
        *out = __float2bfloat16( in );  // BF16: built-in conversion
    }

    __device__ __forceinline__ void stochastic_rounding( float in, __half* out, unsigned int seed )
    {
        *out = __float2half( in );  // FP16: built-in conversion
    }

    // ============================================================================
    // AdamW Implementation
    // ============================================================================

    // Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
    // Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
    __device__ float lerp( float start, float end, float weight )
    {
        return fma( weight, end, fma( -weight, start, start ) );
    }

    template <typename Tp, typename Tg>
    __device__ void adamw_update( Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
        float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
        float grad_scale, unsigned int seed )
    {
        // Sanity check thresholds
        constexpr float kGradAbsLimit = 500.0f;        // Gradients can spike temporarily
        constexpr float kMomentAbsLimit = 250.0f;      // Accumulated gradients
        constexpr float kAdaptiveLRAbsLimit = 100.0f;  // Normalized update magnitude
        constexpr float kParamAbsLimit = 10.0f;        // Actual parameter values
        constexpr float kParamChangeAbsLimit = 1.0f;   // Optional: limit |new - old|
        constexpr int kNumParamsToPrint = 2;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_parameters)
        {
            return;
        }

        // Load gradient
        float grad = grad_scale * (float)grads_memory[idx];

        // Check 1: Gradient sanity
        if ( !isfinite( grad ) || fabsf( grad ) > kGradAbsLimit ) {
            printf(
                "AdamW DEBUG gradient: block=%d thread=%d idx=%d grad_raw=%f grad_scaled=%f grad_scale=%f\n",
                blockIdx.x, threadIdx.x, idx, (float)grads_memory[ idx ], grad, grad_scale
            );
        }
        KERNEL_ASSERT( isfinite( grad ) );
        KERNEL_ASSERT( fabsf( grad ) <= kGradAbsLimit );

        float m = m_memory[idx];
        float v = v_memory[idx];

        // Update first moment (momentum)
        m = lerp( grad, m, beta1 );

        // Check 2: First moment sanity
        if ( !isfinite( m ) || fabsf( m ) > kMomentAbsLimit ) {
            printf(
                "AdamW DEBUG m: block=%d thread=%d idx=%d m=%f grad=%f beta1=%f m_old=%f\n",
                blockIdx.x, threadIdx.x, idx, m, grad, beta1, m_memory[ idx ]
            );
        }
        KERNEL_ASSERT( isfinite( m ) );
        KERNEL_ASSERT( fabsf( m ) <= kMomentAbsLimit );
        
        m_memory[idx] = m;

        // Update second moment (RMSprop)
        v = lerp( grad * grad, v, beta2 );

        // Check 3: Second moment sanity
        if ( !isfinite( v ) || fabsf( v ) > kMomentAbsLimit * kMomentAbsLimit ) {
            printf(
                "AdamW DEBUG v: block=%d thread=%d idx=%d v=%f grad²=%f beta2=%f v_old=%f\n",
                blockIdx.x, threadIdx.x, idx, v, grad * grad, beta2, v_memory[ idx ]
            );
        }
        KERNEL_ASSERT( isfinite( v ) );
        KERNEL_ASSERT( v >= 0.0f );  // Second moment must be non-negative
        KERNEL_ASSERT( fabsf( v ) <= kMomentAbsLimit * kMomentAbsLimit );

        v_memory[idx] = v;
        
        m /= beta1_correction;  // m_hat
        v /= beta2_correction;  // v_hat

        // fetch the old value of this parameter as a float, from either source
        float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];

        // Check 4: Old parameter sanity
        if ( !isfinite( old_param ) || fabsf( old_param ) > kParamAbsLimit ) {
            printf(
                "AdamW DEBUG old_param: block=%d thread=%d idx=%d old_param=%f\n",
                blockIdx.x, threadIdx.x, idx, old_param
            );
        }
        KERNEL_ASSERT( isfinite( old_param ) );
        KERNEL_ASSERT( fabsf( old_param ) <= kParamAbsLimit );

        // update this parameter
        float param = old_param - (learning_rate * (m / (sqrtf( v ) + eps) + weight_decay * old_param));

        // DEBUG: Print first N parameters
        /*if ( idx < kNumParamsToPrint ) {
            printf(
                "AdamW[%d]: old=%+9.6f grad=%+9.6f m=%+9.6f v=%+9.6f "
                "new=%+9.6f delta=%+9.6f\n",
                idx, old_param, grad, m, v, param, param - old_param
            );
        }*/

        // Check 6: Final parameter sanity
        if ( !isfinite( param ) || fabsf( param ) > kParamAbsLimit ) {
            printf(
                "AdamW DEBUG param: block=%d thread=%d idx=%d param=%f old_param=%f lr=%f wd=%f\n",
                blockIdx.x, threadIdx.x, idx, param, old_param, learning_rate, weight_decay
            );
        }
        KERNEL_ASSERT( isfinite( param ) );
        KERNEL_ASSERT( fabsf( param ) <= kParamAbsLimit );

        float param_change = param - old_param;

        if ( !isfinite( param_change ) || fabsf( param_change ) > kParamChangeAbsLimit ) {
            printf(
                "AdamW DEBUG param_change: block=%d thread=%d idx=%d change=%f old=%f new=%f\n",
                blockIdx.x, threadIdx.x, idx, param_change, old_param, param
            );
        }
        KERNEL_ASSERT( fabsf( param_change ) <= kParamChangeAbsLimit );

        // Update our low precision version of the parameters using stochastic rounding
        // this will be used in the next forward pass
        stochastic_rounding( param, &params_memory[idx], seed );

        // write the full, float version of the param into our master copy, if we maintain one
        // this will be used in the next update
        if (master_params_memory != NULL)
        {
            master_params_memory[idx] = param;
        }
    }

    __global__ void adamw_kernel2( 
        float* params_memory, float* grads_memory, 
        float* m_memory, float* v_memory, long num_parameters,
        float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay )
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i >= num_parameters)
            return;
        
        float grad = grads_memory[i];
        float m = m_memory[i];
        float v = v_memory[i];
        
        // update the first moment (momentum)
        m = lerp( grad, m, beta1 );
        m_memory[i] = m;
        
        // update the second moment (RMSprop)
        v = lerp( grad * grad, v, beta2 );
        v_memory[i] = v;
        m /= beta1_correction;  // m_hat
        v /= beta2_correction;  // v_hat
        
        params_memory[i] -= learning_rate * (m / (sqrtf( v ) + eps) + weight_decay * params_memory[i]);
    }

    template <typename Tp, typename Tg>
    __global__ void adamw_kernel3(
        Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
        ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,
        float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
        float grad_scale, unsigned int seed )
    {
        adamw_update( params_memory + blockIdx.y * w_stride,
            master_params_memory ? master_params_memory + blockIdx.y * s_stride : NULL,
            grads_memory + blockIdx.y * g_stride,
            m_memory + blockIdx.y * s_stride,
            v_memory + blockIdx.y * s_stride,
            num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale,
            seed
        );
    }

    template <typename Tp>
    __global__ void init_from_master_kernel(
        Tp* params_memory, float* master_params_memory, size_t num_parameters,
        ptrdiff_t w_stride, ptrdiff_t s_stride, unsigned int seed )
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_parameters)
        {
            return;
        }
        params_memory += blockIdx.y * w_stride; // adjust for layer offset
        master_params_memory += blockIdx.y * s_stride;
        stochastic_rounding( master_params_memory[idx], &params_memory[idx], seed );
    }

    template <typename Tp, typename Tg>
    void adamw_update( Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
        ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride, int num_slices, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
        float grad_scale, unsigned int seed, cudaStream_t stream )
    {
        // AdamW update
        int block_size = 512;
        int num_blocks = ceil_div( num_parameters, block_size );
        float beta1_correction = 1.0f - powf( beta1, static_cast<float>(t) );
        float beta2_correction = 1.0f - powf( beta2, static_cast<float>(t) );

        adamw_kernel3 <<<dim3( num_blocks, num_slices ), block_size, 0, stream >>> (
            params_memory, master_params_memory, grads_memory,
            m_memory, v_memory, num_parameters, w_stride, g_stride, s_stride,
            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,
            grad_scale, seed);

        cudaCheck( cudaGetLastError() );
    }

    template <typename Tp>
    void init_from_master( Tp* params_memory, float* master_params_memory, size_t num_parameters,
        ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices, unsigned int seed, cudaStream_t stream )
    {
        int block_size = 512; // must match block size of adamw_update so that RNG also matches
        int num_blocks = ceil_div( num_parameters, block_size );

        init_from_master_kernel << <dim3( num_blocks, num_slices ), block_size, 0, stream >> > (
            params_memory, master_params_memory, num_parameters, w_stride, s_stride, seed);

        cudaCheck( cudaGetLastError() );
    }

    // ============================================================================
    // Explicit Template Instantiations
    // ============================================================================

    // FP32 instantiations
    template void adamw_update<float, float>(
        float* params_memory, float* master_params_memory, float* grads_memory,
        float* m_memory, float* v_memory, size_t num_parameters,
        ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride, int num_slices,
        float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
        float grad_scale, unsigned int seed, cudaStream_t stream );

    template void init_from_master<float>(
        float* params_memory, float* master_params_memory, size_t num_parameters,
        ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices, unsigned int seed, cudaStream_t stream );

    // FP16 instantiations (if using half precision)
    template void adamw_update<__half, __half>(
        __half* params_memory, float* master_params_memory, __half* grads_memory,
        float* m_memory, float* v_memory, size_t num_parameters,
        ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride, int num_slices,
        float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
        float grad_scale, unsigned int seed, cudaStream_t stream );

    template void init_from_master<__half>(
        __half* params_memory, float* master_params_memory, size_t num_parameters,
        ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices, unsigned int seed, cudaStream_t stream );

    // Mixed precision: FP16 params, FP32 grads
    template void adamw_update<__half, float>(
        __half* params_memory, float* master_params_memory, float* grads_memory,
        float* m_memory, float* v_memory, size_t num_parameters,
        ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride, int num_slices,
        float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
        float grad_scale, unsigned int seed, cudaStream_t stream );
}