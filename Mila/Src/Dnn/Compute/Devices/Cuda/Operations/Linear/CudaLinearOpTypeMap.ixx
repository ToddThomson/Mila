/**
 * @file CudaLinearOpTypeMap.ixx
 * @brief LinearOpTypeMap specializations for CUDA device targets.
 *
 * Provides compile-time dispatch for (Cuda, TPrecision, TWeightQuant) combinations.
 * A missing specialization for any combination is a hard compile error.
 */
export module Compute.LinearOpTypeMap:Cuda;

import Compute.DeviceType;
import Compute.LinearOpTypeMap.Template;
import Compute.CudaLinearOp;
import Dnn.TensorDataType;
import Dnn.Quantization.Weight.Policies;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief Unquantized FP32 CUDA linear operation.
     *
     * Retained for validation/reference paths. BF16 is the standard inference precision.
     */
    export template<>
    struct LinearOpTypeMap<DeviceType::Cuda, TensorDataType::FP32, NoWeightQuant>
    {
        using op_type = Cuda::Linear::CudaLinearOp<TensorDataType::FP32, NoWeightQuant>;
    };

    /**
     * @brief Unquantized BF16 CUDA linear operation.
     *
     * Standard inference path. No weight scale tensor is allocated or used.
     */
    export template<>
    struct LinearOpTypeMap<DeviceType::Cuda, TensorDataType::BF16, NoWeightQuant>
    {
        using op_type = Cuda::Linear::CudaLinearOp<TensorDataType::BF16, NoWeightQuant>;
    };

    /**
     * @brief FP8 per-channel quantized BF16 CUDA linear operation.
     *
     * Weights stored as FP8_E4M3 with per-output-channel FP32 scales. Requires SM >= 8.9.
     * cuBLASLt executes the mixed-precision GEMM natively on the forward path.
     */
    export template<>
    struct LinearOpTypeMap<DeviceType::Cuda, TensorDataType::BF16, PerChannelFp8<>>
    {
        using op_type = Cuda::Linear::CudaLinearOp<TensorDataType::BF16, PerChannelFp8<>>;
    };
}