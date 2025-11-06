#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "device_launch_parameters.h"
#include "../../Helpers/CudaUtils.h"

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
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_parameters)
        {
            return;
        }

        // get the gradient, m, and v for this parameter
        float grad = grad_scale * (float)grads_memory[idx];
        float m = m_memory[idx];
        float v = v_memory[idx];

        // update the first moment (momentum)
        m = lerp( grad, m, beta1 );
        m_memory[idx] = m;

        // update the second moment (RMSprop)
        v = lerp( grad * grad, v, beta2 );
        v_memory[idx] = v;
        m /= beta1_correction;  // m_hat
        v /= beta2_correction;  // v_hat

        // fetch the old value of this parameter as a float, from either source
        float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];

        // update this parameter
        float param = old_param - (learning_rate * (m / (sqrtf( v ) + eps) + weight_decay * old_param));

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