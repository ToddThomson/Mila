export module Compute.CpuTensorOps;

export import :Zero;
export import :Fill;
// :Math is disabled: importing it here triggers an MSVC C1116 ICE
// (<stop_token> / Compute.MemoryResource). CPU math is unavailable until that
// compiler bug is worked around. CUDA math is unaffected. See TensorOps.Math.ixx.
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