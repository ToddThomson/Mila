module;
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

export module Dnn.PrecisionConfig;

// Example of an AMP configuration class you could add
export class AMPConfig {
public:
    enum class OpPrecision {
        FP32,           // Use FP32 for everything
        FP16,           // Use FP16 for computation where possible
        BF16,           // Use BF16 for computation where possible
        Mixed_MatMul,   // Use FP16/BF16 for matrix multiply, FP32 elsewhere
        Auto            // Let the framework decide based on hardware
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

private:
    OpPrecision precision_;
};
