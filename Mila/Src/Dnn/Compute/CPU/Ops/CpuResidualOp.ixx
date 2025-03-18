module;
#include <memory>
#include <vector>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuResidualOp;

import Dnn.Tensor;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.OperationBase;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuDevice;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    export
    template<typename T>
    class CpuResidualOp :public BinaryOperation<float, float, CpuDevice> {
    public:
        CpuResidualOp() : BinaryOperation<float, float , CpuDevice>( DeviceType::Cpu, OperationType::ResidualOp ) {}

        void forward(
            const Tensor<T, CpuMemoryResource>& input_a,
            const Tensor<T, CpuMemoryResource>& input_b,
            const std::vector<std::shared_ptr<Tensor<T, CpuMemoryResource>>>& parameters,
            Tensor<T, CpuMemoryResource>& output,
            std::vector<std::shared_ptr<Tensor<T, CpuMemoryResource>>>& output_cache ) const override {
			auto A = input_a.data();
            auto B = input_b.data();
			auto Y = output.data();
			auto N = input_a.size();
            
		#pragma omp parallel for
            for ( int i = 0; i < N; i++ ) {
				Y[ i ] = A[ i ] + B[ i ];
            }
		}

        void backward( float* dinp1, float* dinp2, float* dout, int N ) {
            for ( int i = 0; i < N; i++ ) {
                dinp1[ i ] += dout[ i ];
                dinp2[ i ] += dout[ i ];
            }
        }

        static void registerOperation() {
            OperationRegistry<float, float, CpuDevice>::instance().registerOperation( DeviceType::Cpu, "Cpu::ResidualOp", []() -> std::unique_ptr<OperationBase<float, float, CpuDevice>> {
                return std::make_unique<CpuResidualOp<float>>();
            } );
        }

        std::string getName() const override {
            return "Cpu::ResidualOp";
        }
    };
}