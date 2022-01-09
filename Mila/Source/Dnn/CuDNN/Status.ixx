/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

module;
#include <cudnn.h>

export module CuDnn.Status;

namespace Mila::Dnn::CuDNN
{
	export template <typename T>
	struct cudnnResult
	{
	public:

		cudnnResult() = default;

		cudnnResult( cudnnStatus_t status_, T value )
			: status_( status_ ), value_( value )
		{
		}

		bool IsSuccess()
		{
			return (status_ == CUDNN_STATUS_SUCCESS);
		}

		cudnnStatus_t getStatus()
		{
			return status_;
		}

		T getValue()
		{
			return value_;
		}

	private:

		cudnnStatus_t status_ = CUDNN_STATUS_SUCCESS;

		T value_;
	};
}