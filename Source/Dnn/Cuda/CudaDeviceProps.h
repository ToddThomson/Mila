/**
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef MILA_DNN_CUDA_DEVICE_PROPS_H_
#define MILA_DNN_CUDA_DEVICE_PROPS_H_

#include <cuda_runtime.h>
#include <vector>

namespace Mila::Dnn::Cuda {

	class CudaDeviceProps
	{
	public:

		CudaDeviceProps();

		const cudaDeviceProp* get( int devId ) const;

	private:

		std::vector<cudaDeviceProp> props_;
	};
}
#endif