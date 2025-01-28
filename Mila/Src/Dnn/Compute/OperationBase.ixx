module;
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

export module Compute.OperationBase;

import Dnn.Tensor;
import Compute.DeviceType;
import Compute.OperationType;

export namespace Mila::Dnn::Compute
{
	template <typename T>
	class OperationBase {
	public:

		OperationBase( DeviceType device_type, OperationType operation_type )
			: device_type_( device_type ), operation_type_( operation_type ) {}	

		virtual ~OperationBase() = default;
		
		virtual std::string getName() const = 0;

		constexpr DeviceType getDeviceType() const {
			return device_type_;
		}

		constexpr OperationType getOperationType() const {
			return operation_type_;
		}
		
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

	private:
		DeviceType device_type_;
		OperationType operation_type_;
	};
}