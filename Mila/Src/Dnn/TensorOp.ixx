module;
#include <vector>
#include <stdexcept>

export module Dnn.TensorOperation;

import Dnn.Tensor;

namespace Mila::Dnn
{
    
    class TensorOperation {
    public:
                
        virtual ~TensorOperation() = default;

    };
}

