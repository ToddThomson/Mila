export module Compute.CudaTensorOps;

export import :Fill;
export import :Math;
export import :Transfer;

import Dnn.TensorOps.Base;
import Compute.DeviceType;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute::Cuda;

	export template<>
	struct TensorOps<Compute::DeviceType::Cuda> : FillOps, MathOps, TransferOps {};
}
