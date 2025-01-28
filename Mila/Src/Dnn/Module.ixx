module;
#include <vector>
#include <string>
#include <memory>
#include <thrust/host_vector.h>

export module Dnn.Module;

import Dnn.Tensor;

namespace Mila::Dnn
{
	// Forward declaration for setParent and getParent
	class Model; 

    export template<typename T>
    class Module {
    public:
        virtual ~Module() = default;

        virtual std::shared_ptr<Tensor<T>> forward( const std::shared_ptr<Tensor<T>>& input ) = 0;

        virtual Tensor<T> backward( const Tensor<T>& gradient ) {
            // Default to no op
            return {};
        }

		virtual size_t parameters() const = 0;

		virtual std::string name() const = 0;

        virtual void print() const = 0;

        void setParent( Model* parent ) {
            parent_ = parent;
        }

        Model* parent() const {
            return parent_;
        }

	private:
		Model* parent_{ nullptr };
    };
}