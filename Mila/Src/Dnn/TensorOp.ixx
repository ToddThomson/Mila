module;
#include <vector>
#include <stdexcept>

export module Dnn.TensorOps;

import Dnn.Tensor;

namespace Mila::Dnn
{
	template<typename T>
    class TensorOps {
    public:
                
        virtual ~TensorOps() = default;

		virtual Tensor<T>& layer_norm_forward( const Tensor<T>& input ) = 0;

    };
}

