/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef MILA_DNN_CUDA_PROFILER_H_
#define MILA_DNN_CUDA_PROFILER_H_

#include <cuda_runtime.h>

namespace Mila::Dnn::Cuda
{
	/// <summary>
	/// 
	/// </summary>
	class CudaTimer
	{
	public:

		CudaTimer()
		{
			cudaDeviceSynchronize();

			cudaEventCreate( &start_ );
			cudaEventCreate( &stop_ );
		}

		void Start()
		{
			cudaEventRecord( start_ );
		}

		float Stop()
		{
			cudaEventRecord( stop_ );
			cudaEventSynchronize( stop_ );
			
			cudaEventElapsedTime( &elapsedTime_, start_, stop_ );

			return elapsedTime_;
		}

	private:

		cudaEvent_t start_;
		cudaEvent_t stop_;

		float elapsedTime_ = 0.0f;
	};
}
#endif