export module Compute.CpuTensorOps;

export import :Fill;
export import :Math;
export import :Transfer;

import Dnn.TensorOps.Base;
import Compute.DeviceType;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute::Cpu;

	export template<>
	struct TensorOps<Compute::DeviceType::Cpu> : FillOps, MathOps, TransferOps {};
}