module;
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

export module Dnn.PrecisionConfig;

export class AMPConfig {
public:
    enum class OpPrecision {
        FP32,           // Use FP32 for everything
        FP16,           // Use FP16 for computation where possible
        BF16,           // Use BF16 for computation where possible
		FP8,            // Use FP8 for computation where possible
        Auto,           // Let the framework decide based on hardware
        Default
    };

    AMPConfig( OpPrecision precision = OpPrecision::Auto ) : precision_( precision ) {
        // Auto-detect hardware capabilities
        if ( precision_ == OpPrecision::Auto ) {
            int deviceId;
            cudaGetDevice( &deviceId );
            cudaDeviceProp prop;
            cudaGetDeviceProperties( &prop, deviceId );

            // Use BF16 on Ampere+ GPUs, FP16 on Volta/Turing, FP32 on older
            if ( prop.major >= 8 ) {
                precision_ = OpPrecision::BF16;
            }
            else if ( prop.major >= 7 ) {
                precision_ = OpPrecision::FP16;
            }
            else {
                precision_ = OpPrecision::FP32;
            }
        }
    }

    bool useReducedPrecision() const {
        return precision_ != OpPrecision::FP32;
    }

    bool useLossScaling() const {
        return precision_ == OpPrecision::FP16; // BF16 usually doesn't need loss scaling
    }

    template<typename T>
    cudaDataType_t getComputeType() const {
        if ( std::is_same_v<T, float> ) {
            return CUDA_R_32F;
        }
        else if ( std::is_same_v<T, __half> ) {
            return CUDA_R_16F;
        }
        else if ( std::is_same_v<T, __nv_bfloat16> ) {
            return CUDA_R_16BF;
        }
        return CUDA_R_32F;
    }

    // Add these methods to the AMPConfig class in PrecisionConfig.ixx

/**
 * @brief Get the appropriate cuBLASLt compute type based on storage and precision settings
 *
 * @tparam TPrecision The storage data type
 * @tparam TCompute The desired computation data type
 * @return cublasComputeType_t The recommended cuBLASLt compute type
 */
    template<typename TPrecision, typename TCompute = float>
    cublasComputeType_t getCublasComputeType() const {
        // Full precision configuration
        if ( precision_ == OpPrecision::FP32 ) {
            return CUBLAS_COMPUTE_32F;
        }

        // Mixed precision configurations
        //if ( precision_ == OpPrecision::Mixed_MatMul ) {
        //    if constexpr ( std::is_same_v<TPrecision, half> ) {
        //        return CUBLAS_COMPUTE_32F_FAST_16F;  // FP16 storage with FP32 compute using TensorCores
        //    }
        //    else if constexpr ( std::is_same_v<TPrecision, __nv_bfloat16> ) {
        //        return CUBLAS_COMPUTE_32F_FAST_16BF; // BF16 storage with FP32 compute using TensorCores
        //    }
        //    else {
        //        return CUBLAS_COMPUTE_32F;           // Default to FP32 compute
        //    }
        //}

        // Reduced precision configurations (pure FP16/BF16)
        if constexpr ( std::is_same_v<TPrecision, half> ) {
            return precision_ == OpPrecision::FP16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F_FAST_16F;
        }
        else if constexpr ( std::is_same_v<TPrecision, __nv_bfloat16> ) {
            return CUBLAS_COMPUTE_32F_FAST_16BF;
        }
    }

    /**
     * @brief Determine if TensorCore acceleration should be used
     *
     * @return bool True if TensorCore acceleration should be used
     */
    bool useTensorCores() const {
        // TensorCores are used in all modes except FP32
        return precision_ != OpPrecision::FP32;
    }


private:
    OpPrecision precision_;
};
