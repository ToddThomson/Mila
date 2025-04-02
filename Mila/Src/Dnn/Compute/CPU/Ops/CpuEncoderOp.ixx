module;
#include <vector>
#include <memory>
#include <string>
#include <type_traits>

export module Compute.CpuEncoderOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CpuDevice;
import Compute.CudaMemoryResource;

namespace Mila::Dnn::Compute
{
	using namespace Mila::Dnn;
	
	export template<typename TInput = uint16_t>
		requires (std::is_same_v<TInput, int> || std::is_same_v<TInput, uint16_t>)
		class CpuEncoderOp : public UnaryOperation<TInput, float, DeviceType::Cpu> {
		public:
			/**
			* @brief Constructs a CpuEncoderOp object
			*/
			CpuEncoderOp() : UnaryOperation<TInput, float, DeviceType::Cpu>( DeviceType::Cpu, OperationType::EncoderOp ) {}

			/**
			 * @brief Forward pass of the encoder operation
			 *
			 * @param input The input tensor with token indices
			 * @param parameters Vector of parameter tensors [wte (token embeddings), wpe (position embeddings)]
			 * @param properties Additional operation attributes
			 * @param output The output tensor to store results
			 * @param output_cache Cache for intermediate results
			 */
			void forward(
				const Tensor<TInput, HostMemoryResource>& input,
				const std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& parameters,
				const OperationAttributes& properties,
				Tensor<float, HostMemoryResource>& output,
				std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& output_cache ) const override {
				auto X = input.data();
				auto Y = output.data();

				auto wte = parameters[ 0 ];
				auto wpe = parameters[ 1 ];

				int B = input.shape()[ 0 ];
				int T = input.shape()[ 1 ];
				int C = wte->shape()[ 1 ];

			#pragma omp parallel for collapse(2)
				for ( int b = 0; b < B; b++ ) {
					for ( int t = 0; t < T; t++ ) {
						float* out_bt = Y + b * T * C + t * C;
						TInput ix = X[ b * T + t ];
						float* wte_ix = wte->data() + ix * C;
						float* wpe_t = wpe->data() + t * C;

						for ( int i = 0; i < C; i++ ) {
							out_bt[ i ] = wte_ix[ i ] + wpe_t[ i ];
						}
					}
				}
			}

			/**
			 * @brief Backward pass of the encoder operation
			 *
			 * @param dwte Gradient for token embeddings
			 * @param dwpe Gradient for position embeddings
			 * @param dout Output gradient
			 * @param input Input token indices tensor
			 * @param B Batch size
			 * @param TElementType Sequence length
			 * @param C Embedding dimension
			 */
			void backward( float* dwte, float* dwpe, float* dout, const Tensor<TInput, HostMemoryResource>& input, int B, int T, int C ) {
			#pragma omp parallel for collapse(2)
				for ( int b = 0; b < B; b++ ) {
					for ( int t = 0; t < T; t++ ) {
						float* dout_bt = dout + b * T * C + t * C;
						TInput ix = input[ b * T + t ];
						float* dwte_ix = dwte + ix * C;
						float* dwpe_t = dwpe + t * C;

						for ( int i = 0; i < C; i++ ) {
							float d = dout_bt[ i ];
						#pragma omp atomic
							dwte_ix[ i ] += d;
						#pragma omp atomic
							dwpe_t[ i ] += d;
						}
					}
				}
			}

			/**
			 * @brief Gets the name of the operation
			 *
			 * @return The name of the operation
			 */
			std::string getName() const override {
				return "Cpu::EncoderOp";
			}
	};

	/**
	* @brief Registers the encoder operations with the operation registry
	*/
	export class CpuEncoderOpRegistrar {
	public:
		static void registerOperations() {
			const std::string opName = "Cpu::EncoderOp";

			OperationRegistry::instance().registerOperation<int, float, DeviceType::Cpu>(
				opName,
				[]() -> std::shared_ptr<OperationBase<int, float, DeviceType::Cpu>> {
					return std::make_shared<CpuEncoderOp<int>>();
				}
			);

			OperationRegistry::instance().registerOperation<uint16_t, float, DeviceType::Cpu>(
				opName,
				[]() -> std::shared_ptr<OperationBase<uint16_t, float, DeviceType::Cpu>> {
					return std::make_shared<CpuEncoderOp<uint16_t>>();
				}
			);
		}

		static inline bool isRegistered = []() {
			registerOperations();
			return true;
			}();
	};
}