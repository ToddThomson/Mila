module;
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <type_traits>
#include <stdexcept>

export module Compute.CpuEncoderOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CpuDevice;
import Compute.CudaMemoryResource;

namespace Mila::Dnn::Compute
{
	using namespace Mila::Dnn;
	
	export
	class CpuEncoderOp :public OperationBase<int, float, CpuDevice> {
	public:
		using MR = CpuDevice::MR;

		CpuEncoderOp() : OperationBase<int, float, CpuDevice>( DeviceType::Cpu, OperationType::EncoderOp ) {}

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
			OperationRegistry<int, float, CpuDevice>::instance().registerOperation( DeviceType::Cpu, "Cpu::EncoderOp", []() -> std::unique_ptr<OperationBase<int, float, CpuDevice>> {
				return std::make_unique<CpuEncoderOp>();
			} );
		}

		std::string getName() const override {
			return "Cpu::EncoderOp";
		}
	};
}
