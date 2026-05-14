/**
 * @file CpuLinearOpTypeMap.ixx
 * @brief LinearOpTraits specialization for CPU / FP32.
 */
export module Compute.LinearOpTypeMap:Cpu;

import Dnn.TensorDataType;
import Compute.LinearOpTypeMap.Template;
import Compute.CpuLinearOp;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    template<>
    struct LinearOpTypeMap<DeviceType::Cpu, TensorDataType::FP32>
    {
        using op_type = Mila::Dnn::Compute::CpuLinearOp;
    };
}