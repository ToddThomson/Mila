/**
 * @file CpuTensorOps.Zero.ixx
 * @brief CPU fast zeroing partition for tensor buffers.
 *
 * Provides a device-dispatched fast zero() operation that uses an efficient
 * byte-wise memset for contiguous CPU buffers.
 */

module;
#include <cstring>
#include <type_traits>

export module Compute.CpuTensorOps:Zero;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.ExecutionContext;
import Compute.DeviceType;
import Compute.CpuTensorDataTypeTraits;

namespace Mila::Dnn::Compute::Cpu
{
	using namespace Mila::Dnn::Compute;

	export struct ZeroOps
	{
		/**
		 * @brief Zero the contents of a CPU tensor buffer.
		 *
		 * - No-op for empty tensors.
		 * - Uses std::memset for contiguous, trivially-copyable native types.
		 * - ExecutionContext parameter is accepted for API uniformity but ignored on CPU.
		 *
		 * Threading / sync: CPU path is synchronous.
		 */
		template<TensorDataType TDataType, typename TMemoryResource>
			requires isValidTensor<TDataType, TMemoryResource>
		static void zero( Tensor<TDataType, TMemoryResource>& tensor, [[maybe_unused]] IExecutionContext* exec_context = nullptr )
		{
			if ( tensor.size() == 0 )
			{
				return;
			}

			using NativeType = typename CpuTensorDataTypeTraits::template native_type<TDataType>;

			// Use memset for trivially-copyable native types (fast path).
			// All-bits-zero is a valid numeric zero for integers and IEEE floats.
			static_assert(std::is_trivially_copyable_v<NativeType>, "Zero path requires trivially copyable native type");

			void* dst = tensor.data();
			const std::size_t bytes = tensor.size() * sizeof( NativeType );

			std::memset( dst, 0, bytes );
		}
	};
}