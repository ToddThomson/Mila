#ifndef MILA_DNN_CUDNN_RESULT_H_
#define MILA_DNN_CUDNN_RESULT_H_

#include <cudnn.h>

namespace Mila::Dnn::CuDNN
{
	template <typename T>
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
#endif