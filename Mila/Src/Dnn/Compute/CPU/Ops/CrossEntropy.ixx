module;
#include <corecrt_math.h>

export module Compute.CpuCrossEntropy;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.CpuMemoryResource;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
	export
	class CpuCrossEntropyOp : public OperationBase<float, CpuMemoryResource> {
	public:

		CpuCrossEntropyOp() : OperationBase<float, CpuMemoryResource>( DeviceType::Cpu, OperationType::CrossEntropyOp ) {}

		void forward( float* losses, float* probs, const Tensor<int, CpuMemoryResource>& targets, int B, int T, int Vp ) {
			// output: losses is (B,T) of the individual losses at each position
			// input: probs are (B,T,Vp) of the probabilities
			// input: targets is (B,T) of integers giving the correct index in logits
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
	};
}