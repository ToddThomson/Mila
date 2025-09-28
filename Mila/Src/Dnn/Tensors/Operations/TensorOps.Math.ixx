/**
 * @file TensorOps.Math.ixx
 * @brief Device-dispatching math helpers for tensor arithmetic operations
 *
 * This partition provides the high-level, device-agnostic entry points for
 * tensor math operations (e.g., element-wise addition). Each helper forwards
 * to the device-specific `TensorOps<ComputeDeviceTag>::...` implementation
 * (see CPU and CUDA specializations).
 *
 * The templates are constrained with `isValidTensor<TDataType, TMemoryResource>`
 * to ensure the tensor configuration is valid (memory resource compatibility,
 * type traits available, and device accessibility).
 *
 * Usage:
 *   - Call `add(a, b)` for element-wise addition of two tensors with the same
 *     abstract data type and memory resource. The call is automatically
 *     dispatched to the appropriate device implementation.
 *
 * Preconditions:
 *   - Both operands must satisfy `isValidTensor` and have matching shapes.
 *   - Device-specific implementations are expected to validate shapes and
 *     perform the operation efficiently.
 */

module;
#include <concepts>

export module Dnn.TensorOps:Math;
export import :Math.Cpu;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorTraits;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;

namespace Mila::Dnn
{
	/**
	 * @brief Element-wise addition (device-dispatched).
	 *
	 * Forwards an element-wise addition of two tensors to the device-specific
	 * implementation `TensorOps<Tag>::add`. The function is constrained by
	 * `isValidTensor` so the compiler will reject unsupported tensor
	 * configurations (e.g., device-only types with non-device memory).
	 *
	 * @tparam TDataType Abstract tensor data type from `TensorDataType` enum.
	 * @tparam TMemoryResource Memory resource type used for both operands.
	 * @param a Left-hand operand tensor.
	 * @param b Right-hand operand tensor.
	 * @return New tensor containing the element-wise sum of `a` and `b`.
	 *
	 * @note Device implementation is responsible for validating that shapes
	 *       match and for selecting efficient execution strategies.
	 */
	export template<TensorDataType TDataType, typename TMemoryResource>
		requires isValidTensor<TDataType, TMemoryResource>
	Tensor<TDataType, TMemoryResource> add( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
		using Tag = typename TMemoryResource::ComputeDeviceTag;

		return TensorOps<Tag>::add( a, b );
	}
}