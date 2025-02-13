module;
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <type_traits>
#include <stdexcept>

export module Compute.CpuEncoderOp;

import Dnn.Tensor;
//import Compute.OperationBase;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.DeviceMemoryResource;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
	/**
	* @brief Base class for all compute operations.
	*
	* @tparam T The data type of the tensor elements.
	* @tparam MR The memory resource type, must be derived from MemoryResource.
	*/
	export
	template <typename TInput, typename TOutput, typename MR>
	requires std::is_same_v<MR, CpuMemoryResource> || std::is_same_v<MR, DeviceMemoryResource>
	class OperationBase {
	public:
		/**
		* @brief Constructs an OperationBase object.
		*
		* @param device_type The type of device on which the operation will be executed.
		* @param operation_type The type of the operation.
		*/
		OperationBase( DeviceType device_type, OperationType operation_type )
			: device_type_( device_type ), operation_type_( operation_type ) {}

		/**
		* @brief Virtual destructor for the OperationBase class.
		*/
		virtual ~OperationBase() = default;

		/**
		* @brief Gets the name of the operation.
		*
		* @return The name of the operation.
		*/
		virtual std::string getName() const = 0;

		/**
		* @brief Gets the device type.
		*
		* @return The device type.
		*/
		constexpr DeviceType getDeviceType() const {
			return device_type_;
		}

		/**
		* @brief Gets the operation type.
		*
		* @return The operation type.
		*/
		constexpr OperationType getOperationType() const {
			return operation_type_;
		}

		/**
		* @brief Executes the forward pass of the operation.
		*
		* @param input The input tensor.
		* @param parameters The parameters for the operation.
		* @param output The output tensor.
		* @param output_cache Cache for the output tensors.
		*/
		virtual void forward(
			const Tensor<TInput, MR>& input,
			const std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& parameters,
			Tensor<TOutput, MR>& output,
			std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& output_cache ) const = 0;

		/**
		* @brief Executes the backward pass of the operation.
		*
		* @param grad The gradient tensor.
		* @param inputs The input tensors.
		* @param outputGrads The gradients of the output tensors.
		*
		* @throws std::runtime_error If the operation does not support backward pass.
		*/
		virtual void backward(
			const Tensor<TInput, MR>& grad,
			const std::vector<std::shared_ptr<Tensor<TInput, MR>>>& parameters,
			std::vector<std::shared_ptr<Tensor<TOutput, MR>>>& outputGrads ) const {
			// Default implementation for backward pass
			throw std::runtime_error( "Operation does not support backward pass." );
		};

	private:
		DeviceType device_type_; ///< The device type.
		OperationType operation_type_; ///< The operation type.
	};

	export
	class CpuEncoderOp :public OperationBase<int, float, CpuMemoryResource> {
	public:

		CpuEncoderOp() : OperationBase<int, float, CpuMemoryResource>( DeviceType::Cpu, OperationType::EncoderOp ) {}

		void forward(
			const Tensor<int, CpuMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<float, CpuMemoryResource>>>& parameters,
			Tensor<float, CpuMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<float, CpuMemoryResource>>>& output_cache ) const override {
			auto X = input.data();
			auto Y =  output.data();

			auto wte = parameters[ 0 ];
			auto wpe = parameters[ 1 ];

			int B = input.shape()[ 0 ];
			int T = input.shape()[ 1 ];
			int C = wte->shape()[ 1 ];

			for ( int b = 0; b < B; b++ ) {
				for ( int t = 0; t < T; t++ ) {
					float* out_bt = Y + b * T * C + t * C;
					int ix = X[ b * T + t ];
					float* wte_ix = wte->data() + ix * C;
					float* wpe_t = wpe->data() + t * C;

					for ( int i = 0; i < C; i++ ) {
						out_bt[ i ] = wte_ix[ i ] + wpe_t[ i ];
					}
				}
			}
		}

		void backward( float* dwte, float* dwpe, float* dout, const Tensor<int,CpuMemoryResource>& inp, int B, int T, int C ) {
			for ( int b = 0; b < B; b++ ) {
				for ( int t = 0; t < T; t++ ) {
					float* dout_bt = dout + b * T * C + t * C;
					int ix = inp[ b * T + t ];
					float* dwte_ix = dwte + ix * C;
					float* dwpe_t = dwpe + t * C;
					for ( int i = 0; i < C; i++ ) {
						float d = dout_bt[ i ];
						dwte_ix[ i ] += d;
						dwpe_t[ i ] += d;
					}
				}
			}
		}

		static void registerOperation() {
			OperationRegistry<int, float, CpuMemoryResource>::instance().registerOperation( DeviceType::Cpu, "Cpu::EncoderOp", []() -> std::unique_ptr<OperationBase<int, float, CpuMemoryResource>> {
				return std::make_unique<CpuEncoderOp>();
			} );
		}

		std::string getName() const override {
			return "Cpu::EncoderOp";
		}
	};
}
