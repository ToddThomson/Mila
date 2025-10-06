/**
 * @file TensorOps.Initializers.ixx
 * @brief High-level initializer helpers (device-dispatching) for tensors
 *
 * This partition provides the generic, device-agnostic entry points for tensor
 * initialization operations (copying host-provided values into tensors and
 * filling tensors with scalar values).  The implementation forwards to the
 * device-specific `TensorOps<...>` partitions (for example
 * `TensorOps<Compute::CpuComputeDeviceTag>::fill(...)`).
 *
 * The host representation for a logical tensor element is defined by
 * `TensorDataTypeTraits<TDataType>::host_type` and is exposed here via the
 * alias `host_value_t<TDataType>` so callers and implementations use a single,
 * authoritative host-side type for conversions.
 *
 * Usage:
 * - Call `fill(tensor, values)` or `fill(tensor, scalar)` from user
 *   code. The host value type is selected automatically from the tensor data
 *   type (float for floating-point tensors, int32_t for integer tensors).
 *
 * Preconditions:
 * - The `isValidTensor<TDataType,TMemoryResource>` concept must hold for the
 *   tensor types used here (ensures memory resource compatibility and trait
 *   availability).
 */

module;
#include <concepts>
#include <span>

export module Dnn.TensorOps:Fill;
export import :Fill.Cpu;
//export import :Fill.Cuda;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;

namespace Mila::Dnn
{
	/**
	 * @brief Host value type for given abstract tensor data type.
	 *
	 * Maps floating tensor types to `float` and integer tensor types to `int32_t`.
	 * Use this alias when declaring host-side buffers, spans or scalar arguments
	 * intended for conversion/transfer into tensors of `TDataType`.
	 *
	 * @tparam TDataType Abstract tensor data type from `TensorDataType` enum.
	 */
	template<TensorDataType TDataType>
	using host_value_t = std::conditional_t<TensorDataTypeTraits<TDataType>::is_integer_type, int32_t, float>;

	/**
	 * @brief Copy host values into a tensor with device dispatch.
	 *
	 * Forwards the host->tensor copy operation (span form) to the device-specific
	 * implementation `TensorOps<Tag>::fill`.
	 *
	 * The host element type is selected by `host_value_t<TDataType>` so callers
	 * must provide values in the expected host representation (float for
	 * floating-point tensor types, int32_t for integer tensor types).  The
	 * device implementation performs any necessary conversion/quantization.
	 *
	 * @tparam TDataType Abstract tensor data type.
	 * @tparam TMemoryResource Memory resource type backing the tensor.
	 * @param a Destination tensor to be filled. Must satisfy `isValidTensor`.
	 * @param host_values Span of host values in host representation (see host_value_t).
	 */
	export template<TensorDataType TDataType, typename TMemoryResource>
		requires isValidTensor<TDataType, TMemoryResource>
	void fill( Tensor<TDataType, TMemoryResource>& a, std::span<const host_value_t<TDataType>> host_values ) {
		using DeviceTag = typename TMemoryResource::ComputeDeviceTag;

		return TensorOps<DeviceTag>::fill( a, host_values );
	}

	/**
	 * @brief Fill a tensor with a scalar host value (device-dispatched).
	 *
	 * Forwards scalar fills to the device-specific `TensorOps<Tag>::fill`.
	 * The function signature enforces the expected host scalar representation
	 * for each abstract tensor data type via `host_value_t<TDataType>`.
	 *
	 * Example:
	 * - Floating tensor: `fill(tensor, 1.234f);`
	 * - Integer tensor:  `fill(tensor, int32_t{42});`
	 *
	 * @tparam TDataType Abstract tensor data type.
	 * @tparam TMemoryResource Memory resource type backing the tensor.
	 * @param a Destination tensor to be filled. Must satisfy `isValidTensor`.
	 * @param host_value Scalar value in host representation to broadcast to the tensor.
	 */
	export template<TensorDataType TDataType, typename TMemoryResource>
		requires isValidTensor<TDataType, TMemoryResource>
	void fill( Tensor<TDataType, TMemoryResource>& a, host_value_t<TDataType> host_value ) {
		using DeviceTag = typename TMemoryResource::ComputeDeviceTag;
		
		return TensorOps<DeviceTag>::fill( a, host_value );
	}
}