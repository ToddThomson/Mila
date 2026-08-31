export module Compute.CpuTensorOps;

export import :Zero;
export import :Fill;
export import :Math;
export import :Transfer;
export import :Random;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute::Cpu;

	export struct CpuTensorOps : ZeroOps, FillOps, MathOps, TransferOps, RandomOps {};
}