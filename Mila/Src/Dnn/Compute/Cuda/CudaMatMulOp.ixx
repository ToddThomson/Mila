module;
#include <math.h>
#include <iostream>
#include <thrust/host_vector.h>

#include "Kernels/Cuda.MatMul.h"

export module Compute.CudaMatMulOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.OperationRegistry;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    export
    template<typename T>
    class CudaMatMulOp :public OperationBase<T> {
    public:

		CudaMatMulOp() : OperationBase<T>( DeviceType::kCuda, OperationType::kMatMulOp ) {}

        void forward(
            const std::shared_ptr<Tensor<T>>& input,
            const std::vector<std::shared_ptr<Tensor<T>>>& parameters_,
            std::shared_ptr<Tensor<T>>& output,
            std::vector<std::shared_ptr<Tensor<T>>>& output_attributes ) const override {
            
            auto weight = parameters_[ 0 ];
            auto bias = parameters_[ 1 ];

            int B = input->shape()[ 0 ];
            int T = input->shape()[ 1 ];
            int C = input->shape()[ 2 ];
            int OC = weight->shape()[ 0 ];

			cuda_matmul_forward( input->data(), weight->data(), bias->data(), output->data(), B, T, C, OC );
        }

        static void registerOperation() {
            OperationRegistry<float>::instance().registerOperation( "CUDA", "Cuda::MatMulOp", []() -> std::shared_ptr<OperationBase<float>> {
                return std::make_shared<CudaMatMulOp<float>>();
                } );
        }

        std::string getName() const override {
            return "Cuda::MatMulOp";
        }
    };
}