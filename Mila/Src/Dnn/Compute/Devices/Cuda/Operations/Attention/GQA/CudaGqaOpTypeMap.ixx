/**
 * @file CudaGpaOpMap.ixx
 * @brief GqaOpTypeMap specializations for CUDA FP32 and BF16 compute precisions.
 */
export module Compute.GqaOpTypeMap:Cuda;

import Dnn.TensorDataType;
import Compute.GqaOpTypeMap.Template;
import Compute.CudaGqaOp;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    export template<>
        struct GqaOpTypeMap<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>
    {
        using op_type = Cuda::Gqa::CudaGqaOp<TensorDataType::FP32>;
    };

    export template<>
        struct GqaOpTypeMap<DeviceType::Cuda, TensorDataType::BF16, TensorDataType::BF16>
    {
        // static constexpr TensorDataType data_type = Cuda::Linear::CudaLinearOp<TensorDataType::BF16>;
        using op_type = Cuda::Gqa::CudaGqaOp<TensorDataType::BF16>;
    };
}