export module Compute.CudaTensorOps;

export import :Zero;
export import :Fill;
export import :Math;
export import :Transfer;
export import :Structural;
export import :Random;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute::Cuda;

	export struct CudaTensorOps : ZeroOps, FillOps, MathOps, TransferOps, StructuralOps, RandomOps {};
}
