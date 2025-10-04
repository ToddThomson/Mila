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
//export import :Math.Cuda;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorTypeMap;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;

namespace Mila::Dnn
{
	/**
	 * @brief Element-wise addition (device-dispatched).
	 */
	export template<TensorDataType TDataType, typename TMemoryResource>
		requires isValidTensor<TDataType, TMemoryResource>
	Tensor<TDataType, TMemoryResource> add( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
		using DeviceTag = typename TMemoryResource::ComputeDeviceTag;
		return TensorOps<DeviceTag>::add( a, b );
	}

	/**
	 * @brief Element-wise subtraction (device-dispatched).
	 */
	export template<TensorDataType TDataType, typename TMemoryResource>
		requires isValidTensor<TDataType, TMemoryResource>
	Tensor<TDataType, TMemoryResource> subtract( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
		using DeviceTag = typename TMemoryResource::ComputeDeviceTag;
		return TensorOps<DeviceTag>::subtract( a, b );
	}

	/**
	 * @brief Element-wise multiplication (device-dispatched).
	 */
	export template<TensorDataType TDataType, typename TMemoryResource>
		requires isValidTensor<TDataType, TMemoryResource>
	Tensor<TDataType, TMemoryResource> multiply( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
		using DeviceTag = typename TMemoryResource::ComputeDeviceTag;
		return TensorOps<DeviceTag>::multiply( a, b );
	}

	/**
	 * @brief Element-wise division (device-dispatched).
	 */
	export template<TensorDataType TDataType, typename TMemoryResource>
		requires isValidTensor<TDataType, TMemoryResource>
	Tensor<TDataType, TMemoryResource> divide( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
		using DeviceTag = typename TMemoryResource::ComputeDeviceTag;
		return TensorOps<DeviceTag>::divide( a, b );
	}

	// --------------------------------------------------------------------
	// Thin operator syntactic sugar forwarding to the above helpers
	// --------------------------------------------------------------------

	export template<TensorDataType TDataType, typename TMemoryResource>
		requires isValidTensor<TDataType, TMemoryResource>
	inline Tensor<TDataType, TMemoryResource> operator+( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
		return add<TDataType, TMemoryResource>( a, b );
	}

	export template<TensorDataType TDataType, typename TMemoryResource>
		requires isValidTensor<TDataType, TMemoryResource>
	inline Tensor<TDataType, TMemoryResource> operator-( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
		return subtract<TDataType, TMemoryResource>( a, b );
	}

	export template<TensorDataType TDataType, typename TMemoryResource>
		requires isValidTensor<TDataType, TMemoryResource>
	inline Tensor<TDataType, TMemoryResource> operator*( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
		return multiply<TDataType, TMemoryResource>( a, b );
	}

	export template<TensorDataType TDataType, typename TMemoryResource>
		requires isValidTensor<TDataType, TMemoryResource>
	inline Tensor<TDataType, TMemoryResource> operator/( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
		return divide<TDataType, TMemoryResource>( a, b );
	}
}