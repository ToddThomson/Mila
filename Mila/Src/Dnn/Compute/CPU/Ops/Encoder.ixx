module;
#include <corecrt_math.h>  
#include <corecrt_math_defines.h>
#include <thrust/host_vector.h>

export module Compute.CpuEncoder;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.CpuMemoryResource;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
	export
	class CpuEncoderOp :public OperationBase<float, CpuMemoryResource> {
	public:

		CpuEncoderOp() : OperationBase<float, CpuMemoryResource>( DeviceType::Cpu, OperationType::EncoderOp ) {}

		void forward( float* out, const Tensor<int,CpuMemoryResource>& inp, float* wte, float* wpe, int B, int T, int C ) {
			for ( int b = 0; b < B; b++ ) {
				for ( int t = 0; t < T; t++ ) {
					float* out_bt = out + b * T * C + t * C;
					int ix = inp[ b * T + t ];
					float* wte_ix = wte + ix * C;
					float* wpe_t = wpe + t * C;

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
	};
}
