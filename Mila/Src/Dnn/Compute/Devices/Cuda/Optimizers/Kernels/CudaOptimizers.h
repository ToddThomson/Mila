#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace Mila::Dnn::Compute
{
    template <typename Tp, typename Tg>
    void adamw_update(
        Tp* params_memory,
        float* master_params_memory,
        Tg* grads_memory,
        float* m_memory,
        float* v_memory,
        size_t num_parameters,
        ptrdiff_t w_stride,
        ptrdiff_t g_stride,
        ptrdiff_t s_stride,
        int num_slices,
        float learning_rate,
        float beta1,
        float beta2,
        int t,
        float eps,
        float weight_decay,
        float grad_scale,
        unsigned int seed,
        cudaStream_t stream );
}