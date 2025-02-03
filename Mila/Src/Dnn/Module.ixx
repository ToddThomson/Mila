module;
#include <vector>
#include <string>
#include <memory>

export module Dnn.Module;

import Dnn.Tensor;

namespace Mila::Dnn
{
    export 
    template<typename T>
    class Module {
    public:
        virtual ~Module() = default;

        virtual std::shared_ptr<Tensor<T>> forward( const std::shared_ptr<Tensor<T>>& input ) = 0;

        virtual Tensor<T> backward( const Tensor<T>& gradient ) {
            // Default to no op
            return {};
        }

		void setTrainingMode( bool training ) {
			is_training_ = training;
		}

        bool isTraining() const {
            return is_training_;
        }

        virtual size_t parameters() const = 0;

        virtual std::string name() const = 0;

        virtual void print() const = 0;

    private:
        bool is_training_{ false };
    };
}