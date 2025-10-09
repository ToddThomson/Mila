#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace Mila::Dnn::Compute::Cuda
{
	/**
	 * @brief Launch type-converting copy kernel between tensors
	 *
	 * Copies data between tensors with automatic type conversion using
	 * specialized device conversion functions for optimal performance.
	 *
	 * @tparam SrcT Source tensor element type
	 * @tparam DstT Destination tensor element type
	 * @param d_src Source device memory pointer
	 * @param d_or_h_dst Destination memory pointer (device or mapped host)
	 * @param n Number of elements to copy
	 * @param stream CUDA stream for asynchronous execution (0 for default stream)
	 */
	template <typename SrcT, typename DstT>
	void launch_convert_copy_kernel( const SrcT* d_src, DstT* d_or_h_dst, size_t n, cudaStream_t stream );

	/**
	 * @brief Launch optimized same-type copy kernel
	 *
	 * Performs optimized copy between tensors of the same type using
	 * vectorized memory operations when possible.
	 *
	 * @tparam T Tensor element type (same for source and destination)
	 * @param d_src Source device memory pointer
	 * @param d_dst Destination device memory pointer
	 * @param n Number of elements to copy
	 * @param stream CUDA stream for asynchronous execution (0 for default stream)
	 */
	template <typename T>
	void launch_fast_copy_kernel( const T* d_src, T* d_dst, size_t n, cudaStream_t stream );
}