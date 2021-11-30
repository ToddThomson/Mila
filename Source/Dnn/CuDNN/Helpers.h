#ifndef MILA_DNN_CUDNN_HELPERS_H_
#define MILA_DNN_CUDNN_HELPERS_H_

#include <cudnn.h>
#include "Status.h"

namespace Mila::Dnn::CuDNN
{
	inline cudnnResult<int> getVersionPart( libraryPropertyType type )
	{
		int versionPart;
		auto status_ = cudnnGetProperty( type, &versionPart );

		return cudnnResult<int>( status_, versionPart );
	};
}
#endif