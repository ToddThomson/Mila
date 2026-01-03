/**
 * @file CudaTensorOps.Zero.ixx
 * @brief CUDA fast zeroing partition for tensor buffers.
 *
 * Provides a device-dispatched fast `zero()` operation that uses `cudaMemsetAsync`
 * for contiguous CUDA buffers. The operation is allocation-free and accepts an
 * optional execution context to perform non-blocking zeroing on the caller's stream.
 */

module;
#include <cuda_runtime.h>
#include <cstring>
#include <type_traits>
#include <stdexcept>

export module Compute.CudaTensorOps:Zero;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorDataTypeMap;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.DeviceType;
import Compute.DeviceTraits;
import Compute.DeviceId;
import Cuda.Helpers;
import Cuda.Error;

namespace Mila::Dnn::Compute::Cuda
{
	using namespace Mila::Dnn::Compute;

	// Forward-declare CUDA-specific execution context type (concrete alias exists in ExecutionContext:Cuda).
	class CudaExecutionContext;

	export struct ZeroOps
	{
		/**
		 * @brief Zero the contents of a CUDA tensor buffer.
		 *
		 * - No-op for empty tensors.
		 * - Uses cudaMemsetAsync for contiguous device buffers.
		 * - If exec_context is provided and is a CUDA context, the context's
		 *   stream is used and the call is non-blocking. If exec_context is null the
		 *   tensor-provided device is used and the default stream is used synchronously.
		 *
		 * Preconditions:
		 * - The tensor buffer must be a single contiguous allocation (TensorBuffer by design).
		 *
		 * @tparam TDataType Abstract tensor data type
		 * @tparam TMemoryResource Memory resource type backing the tensor
		 * @param tensor Destination CUDA tensor to zero
		 * @param exec_context Optional execution context (borrowed). When provided and
		 *                     recognized as CUDA context, zero is scheduled on that stream.
		 */
		template<TensorDataType TDataType, typename TMemoryResource>
			requires isValidTensor<TDataType, TMemoryResource>
		static void zero( Dnn::Tensor<TDataType, TMemoryResource>& tensor, IExecutionContext* exec_context = nullptr )
		{
			if ( tensor.getStorageSize() == 0 )
			{
				return;
			}

			// Fast-path is only valid for numeric storage where all-bits-zero == numeric zero.
			static_assert(
				TensorDataTypeTraits<TDataType>::is_float_type || TensorDataTypeTraits<TDataType>::is_integer_type,
				"Zero fast-path only valid for numeric tensor data types (integer or IEEE floats)."
			);

			std::size_t bytes = tensor.getStorageSize();

			void* dst = tensor.rawData();
			if ( dst == nullptr )
			{
				// Moved-from or unallocated buffer; nothing to do
				return;
			}

			// Resolve device and stream. ExecutionContext (when provided) supplies the stream
			DeviceId device_id = tensor.getDeviceId();
			cudaStream_t stream = cudaStreamDefault;

			if ( exec_context )
			{
				auto* cuda_ctx = cast_context_<DeviceType::Cuda>( exec_context );
				stream = cuda_ctx->getStream();
			}

			int dev_index = device_id.index;

			Cuda::setCurrentDevice( dev_index );

			cudaError_t err = cudaMemsetAsync( dst, 0, bytes, stream );
			if ( err != cudaSuccess )
			{
				throw std::runtime_error( std::string( "cudaMemsetAsync failed: " ) + cudaGetErrorString( err ) );
			}

			// Synchronous semantics when caller did not provide an execution context.
			if ( exec_context == nullptr )
			{
				cudaError_t syncErr = cudaStreamSynchronize( stream );
				if ( syncErr != cudaSuccess )
				{
					throw std::runtime_error( std::string( "cudaStreamSynchronize failed: " ) + cudaGetErrorString( syncErr ) );
				}
			}
		}
	};
}