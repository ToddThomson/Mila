export module Compute.CpuTensorOps;

export import :Zero;
export import :Fill;
//export import :Math;
export import :Transfer;
export import :Random;

import Dnn.TensorOps.Base;
import Compute.DeviceType;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute::Cpu;

	template<>
	struct TensorOps<Compute::DeviceType::Cpu> : ZeroOps, FillOps, /* MathOps, */ TransferOps, RandomOps {};
}