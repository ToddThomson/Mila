module;
#include <vector>
#include <string>

export module Dnn.Module;

import Dnn.Tensor;

namespace Mila::Dnn
{
    export template<typename T>
    class Module {
    public:
        virtual ~Module() = default;

        virtual Tensor<T> forward( const Tensor<T>& input ) = 0;

        virtual Tensor<T> backward( const Tensor<T>& gradient ) {
            // Default to no op
            return {};
        }

		virtual void print() = 0;

		//virtual void device( const std::string& device ) = 0;

		virtual size_t parameters() = 0;

		virtual std::string name() = 0;
    };
}