/**
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
  */

#include <cuda_runtime.h>
#include "CudaStream.h"

namespace Mila::Dnn::Cuda
{
	CudaStream CudaStream::Create( bool non_blocking, int device_id )
	{
		cudaStream_t stream;
		int flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
		//DeviceGuard dg(device_id);
		cudaStreamCreateWithFlags( &stream, flags );

		return CudaStream( stream );
	}

	CudaStream CudaStream::CreateWithPriority( bool non_blocking, int priority, int device_id )
	{
		cudaStream_t stream;
		int flags = non_blocking ? cudaStreamNonBlocking : cudaStreamDefault;
		//DeviceGuard dg(device_id);
		cudaStreamCreateWithPriority( &stream, flags, priority );

		return CudaStream( stream );
	}

	void CudaStream::DestroyHandle( cudaStream_t stream_id )
	{
		cudaStreamDestroy( stream_id );
	}
}