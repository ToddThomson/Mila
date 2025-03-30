module;
#include <memory>
#include <vector>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#ifdef USE_OMP
#include <omp.h>
#endif
#include <iostream>

export module Compute.CpuGeluOp;

import Dnn.Tensor;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuDevice;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
	const float GELU_SCALING_FACTOR = sqrtf( 2.0f / M_PI );
	// REVIEW: constexpr float GELU_SCALING_FACTOR = sqrtf( 2.0f / M_PI );

	export class CpuGeluOp : public UnaryOperation<float, float, DeviceType::Cpu> {
	public:
		CpuGeluOp() : UnaryOperation<float, float, DeviceType::Cpu>( DeviceType::Cpu, OperationType::GeluOp ) {}

		void forward(
			const Tensor<float, HostMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& parameters,
			const OperationAttributes& properties,
			Tensor<float, HostMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& output_cache ) const override {
			// (approximate) GeLU elementwise non-linearity in the MLP block of Transformer

			const float* X = input.data();
			float* Y = output.data();
			const int N = input.size();

			for ( int i = 0; i < N; i++ ) {
				float x = X[ i ];
				float cube = 0.044715f * x * x * x;
				Y[ i ] = 0.5f * x * (1.0f + tanhf( GELU_SCALING_FACTOR * (x + cube) ));
			}
		}

		// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
	#pragma float_control(precise, on, push)
	#if defined(__GNUC__) && !defined(__clang__)
		__attribute__( (optimize( "no-finite-math-only" )) )
		#endif

			void backward( float* dinp, float* inp, float* dout, int N ) {
			for ( int i = 0; i < N; i++ ) {
				float x = inp[ i ];
				float cube = 0.044715f * x * x * x;
				float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
				float tanh_out = tanhf( tanh_arg );
				float coshf_out = coshf( tanh_arg );
				float sech_out = 1.0f / (coshf_out * coshf_out);
				float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
				dinp[ i ] += local_grad * dout[ i ];
			#pragma float_control(pop)
			}
		}

		std::string getName() const override {
			return "Cpu::GeluOp";
		}

		static const std::string& className() {
			static std::string name = "Cpu::GeluOp";
			return name;
		}
	};

	/**
	* @brief Class responsible for registering the CpuGeluOp operation.
	*
	* The CpuGeluOpRegistrar class registers the CpuGeluOp operation with the OperationRegistry.
	* It associates the operation name "Cpu::GeluOp" with a factory function that creates instances of CpuGeluOp.
	*/
	export class CpuGeluOpRegistrar {
	public:
		/**
		* @brief Registers the CpuGeluOp operation with the OperationRegistry.
		*
		* This function registers the CpuGeluOp operation for the CPU device type
		* with the OperationRegistry. It associates the operation name "Cpu::GeluOp"
		* with a factory function that creates instances of CpuGeluOp.
		*/
		static void registerOperations() {
			const std::string opName = "Cpu::GeluOp";

			OperationRegistry::instance().registerOperation<float, float, DeviceType::Cpu>(
				opName,
				[]() -> std::shared_ptr<OperationBase<float, float, DeviceType::Cpu>> {
					return std::make_shared<CpuGeluOp>();
				}
			);
		}
	};
}