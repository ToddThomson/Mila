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
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    export
    template<typename T>
    class CpuResidualOp :public OperationBase<T, CpuMemoryResource> {
    public:
        CpuResidualOp() : OperationBase<T, CpuMemoryResource>( DeviceType::Cpu, OperationType::ResidualOp ) {}

        void forward(
            const std::shared_ptr<Tensor<T, CpuMemoryResource>> input,
            const std::vector<std::shared_ptr<Tensor<T, CpuMemoryResource>>>& parameters,
            std::shared_ptr<Tensor<T, CpuMemoryResource>> output,
            std::vector<std::shared_ptr<Tensor<T, CpuMemoryResource>>>& output_cache ) const override {
			auto N = input->size();
            
            for ( int i = 0; i < N; i++ ) {
                // FIXME: input 2 is required
                output->data()[ i ] = input->data()[ i ] + input->data()[ i ];
            }
		}

        void backward( float* dinp1, float* dinp2, float* dout, int N ) {
            for ( int i = 0; i < N; i++ ) {
                dinp1[ i ] += dout[ i ];
                dinp2[ i ] += dout[ i ];
            }
        }

        static void registerOperation() {
            OperationRegistry<float, CpuMemoryResource>::instance().registerOperation( DeviceType::Cpu, "Cpu::ResidualOp", []() -> std::unique_ptr<OperationBase<float, CpuMemoryResource>> {
                return std::make_unique<CpuResidualOp<float>>();
            } );
        }

        std::string getName() const override {
            return "Cpu::ResidualOp";
        }
    };
}