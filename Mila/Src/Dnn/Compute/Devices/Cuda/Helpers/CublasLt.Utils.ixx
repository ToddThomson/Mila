// CublasLt.Utils.ixx

module;

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <type_traits>

export module CublasLt.Utils;

namespace Mila::Dnn::Compute::Cuda
{
    export template <typename NativeType>
    cudaDataType_t cublaslt_cuda_data_type()
    {
        if constexpr ( std::is_same_v<NativeType, float> )
            return CUDA_R_32F;
        else if constexpr ( std::is_same_v<NativeType, half> )
            return CUDA_R_16F;
        else if constexpr ( std::is_same_v<NativeType, nv_bfloat16> )
            return CUDA_R_16BF;
        else if constexpr ( std::is_same_v<NativeType, __nv_fp8_e4m3> )
            return CUDA_R_8F_E4M3;
        else if constexpr ( std::is_same_v<NativeType, __nv_fp8_e5m2> )
            return CUDA_R_8F_E5M2;
    }

    export template <typename NativeType>
    void cublaslt_compute_types(
        cublasComputeType_t& compute_type,
        cudaDataType_t& scale_type )
    {
        scale_type = CUDA_R_32F;

        if constexpr ( std::is_same_v<NativeType, half> )
            compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
        else if constexpr ( std::is_same_v<NativeType, nv_bfloat16> )
            compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
        else
            compute_type = CUBLAS_COMPUTE_32F;

    }
}