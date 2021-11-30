/**
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
  */

#ifndef _MILA_DNN_CUDA_STREAM_H_
#define _MILA_DNN_CUDA_STREAM_H_

#include <cuda_runtime.h>

#include "CudaUniqueHandle.h"

namespace Mila::Dnn::Cuda
{
	class CudaStream : public CudaUniqueHandle<cudaStream_t, CudaStream>
	{
	public:

		CudaUniqueHandle<cudaStream_t, CudaStream>::CudaUniqueHandle;
		CudaUniqueHandle<cudaStream_t, CudaStream>::operator=;

		/// @brief Creates a stream on specified device (or current device, if device_id < 0)
		static CudaStream Create( bool non_blocking, int device_id = -1 );

		/// @brief Creates a stream with given priority on specified device
		/// (or current device, if device_id < 0)
		static CudaStream CreateWithPriority( bool non_blocking, int priority, int device_id = -1 );

		/// @brief Calls cudaStreamDestroy on the handle.
		static void DestroyHandle( cudaStream_t stream );

	private:

		cudaStream_t stream_id_;
	};
}
#endif