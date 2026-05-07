/**
 * @file CudaLinearOpTraits.ixx
 * @brief LinearOpTraits specializations for CUDA / FP32 and CUDA / BF16.
 */
export module Compute.LinearOpTypeMap:Cuda;

import Dnn.TensorDataType;
import Compute.LinearOpTypeMap.Template;
import Compute.CudaLinearOp;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    export template<>
    struct LinearOpTypeMap<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>
    {
        using op_type = Cuda::Linear::CudaLinearOp<TensorDataType::FP32>;
    };

    export template<>
    struct LinearOpTypeMap<DeviceType::Cuda, TensorDataType::BF16, TensorDataType::BF16>
    {
        using op_type = Cuda::Linear::CudaLinearOp<TensorDataType::BF16>;
    };
}