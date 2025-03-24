module;
#include <math.h>
#include <string>
#include <memory>
#include <vector>
#ifdef USE_OMP
#include <omp.h>
#endif
#include <cmath>

export module Compute.CpuCrossEntropyOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CpuDevice;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
	export
	class CpuCrossEntropyOp : public UnaryOperation<int, float, DeviceType::Cpu> {
	public:

		CpuCrossEntropyOp() : UnaryOperation<int, float, DeviceType::Cpu>( DeviceType::Cpu, OperationType::CrossEntropyOp ) {}

		void forward(
			const Tensor<int, CpuMemoryResource>& input,
			const std::vector<std::shared_ptr<Tensor<float, CpuMemoryResource>>>& parameters,
			const OperationAttributes& attributes,
			Tensor<float, CpuMemoryResource>& output,
			std::vector<std::shared_ptr<Tensor<float, CpuMemoryResource>>>& output_cache ) const override {
		//void forward( float* losses, float* probs, const Tensor<int, CpuMemoryResource>& targets, int B, int T, int Vp ) {
			// output: losses is (B,T) of the individual losses at each position
			// input: probs are (B,T,Vp) of the probabilities
			// input: targets is (B,T) of integers giving the correct index in logits
			auto B = input.shape()[ 0 ];
			auto T = input.shape()[ 1 ];
			auto Vp = parameters[ 0 ]->shape()[ 2 ];

			auto losses = output.data();
			auto probs = parameters[ 0 ]->data();
			auto targets = input.data();

			for ( int b = 0; b < B; b++ ) {
				for ( int t = 0; t < T; t++ ) {
					// loss = -log(probs[target])
					float* probs_bt = probs + b * T * Vp + t * Vp;
					int ix = targets[ b * T + t ];
					losses[ b * T + t ] = -logf( probs_bt[ ix ] );
				}
			}
		}

		void crossentropy_softmax_backward( float* dlogits, float* dlosses, float* probs, const Mila::Dnn::Tensor<int,CpuMemoryResource>& targets,
			int B, int T, int V, int Vp ) {
			// backwards through both softmax and crossentropy
			for ( int b = 0; b < B; b++ ) {
				for ( int t = 0; t < T; t++ ) {
					float* dlogits_bt = dlogits + b * T * Vp + t * Vp;
					float* probs_bt = probs + b * T * Vp + t * Vp;
					float dloss = dlosses[ b * T + t ];
					int ix = targets[ b * T + t ];
					// note we only loop to V, leaving the padded dimensions
					// of dlogits untouched, so gradient there stays at zero
					for ( int i = 0; i < V; i++ ) {
						float p = probs_bt[ i ];
						float indicator = i == ix ? 1.0f : 0.0f;
						dlogits_bt[ i ] += (p - indicator) * dloss;
					}
				}
			}
		}

		static void registerOperation() {
			OperationRegistry<int, float, DeviceType::Cpu>::instance().registerOperation( DeviceType::Cpu, "Cpu::CrossEntropyOp", []() -> std::unique_ptr<OperationBase<int, float, DeviceType::Cpu>> {
				return std::make_unique<CpuCrossEntropyOp>();
			} );
		}

		std::string getName() const override {
			return "Cpu::CrossEntropyOp";
		}
	};
}