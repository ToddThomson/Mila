module;
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>

export module Compute.OperationBase;

import Dnn.Tensor;

export namespace Mila::Dnn::Compute
{
	template <typename T>
	class OperationBase {
	public:
		virtual ~OperationBase() = default;
		
		virtual std::string getName() const = 0;
		
		virtual void forward( 
			const std::shared_ptr<Tensor<T>>& input,
			const std::vector<std::shared_ptr<Tensor<T>>>& input_attributes,
			std::shared_ptr<Tensor<T>>& output,
			std::vector<std::shared_ptr<Tensor<T>>>& output_attributes ) const = 0;
		
		virtual void backward(
			const std::shared_ptr<Tensor<T>>& grad,
			const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
			std::vector<std::shared_ptr<Tensor<T>>>& outputGrads ) const {
			// Default implementation for backward pass
			throw std::runtime_error( "Operation does not support backward pass." );
		};
	};
}