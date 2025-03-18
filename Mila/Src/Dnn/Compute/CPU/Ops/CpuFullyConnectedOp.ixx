module;
#include <math.h>
#include <string>
#include <memory>
#include <vector>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuFullyConnectedOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CpuDevice;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    export
    template<typename TInput, typename TOutput = TInput>
    class CpuFullyConnectedOp : public UnaryOperation<TInput, TOutput, CpuDevice> {
    public:

        CpuFullyConnectedOp() : UnaryOperation<TInput, TOutput, CpuDevice>( DeviceType::Cpu, OperationType::FullyConnectedOp ) {}

        void forward(
            const Tensor<TInput, CpuMemoryResource>& input,
            const std::vector<std::shared_ptr<Tensor<TOutput, CpuMemoryResource>>>& parameters_,
            Tensor<TOutput, CpuMemoryResource>& output,
            std::vector<std::shared_ptr<Tensor<TOutput, CpuMemoryResource>>>& output_state ) const override {
            auto X = input.data();
            auto Y = output.data();

            auto weight = parameters_[ 0 ];
            std::shared_ptr<Tensor<TOutput, CpuMemoryResource>> bias = { nullptr };

            if ( parameters_.size() == 2 ) {
                bias = parameters_[ 1 ];
            }

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
			int C = input.shape()[ 2 ]; // input features
			int OC = output.shape()[ 2 ]; // output features

            const int LOOP_UNROLL = 8;
            if ( B * T % LOOP_UNROLL != 0 ) {
				// TJT: Write a unit test for this case
                forward_naive( input, weight, bias, output, B, T, C, OC );
                return;
            }

        #pragma omp parallel for
            for ( int obt = 0; obt < B * T; obt += LOOP_UNROLL ) {
                for ( int o = 0; o < OC; o++ ) {
                    float result[ LOOP_UNROLL ];
                    for ( int ibt = 0; ibt < LOOP_UNROLL; ibt++ ) {
                        result[ ibt ] = (bias ? bias->data()[ o ] : 0.0f);
                    }

                    for ( int i = 0; i < C; i++ ) {
                        float w = weight->data()[ i + o * C ];
                        for ( int ibt = 0; ibt < LOOP_UNROLL; ibt++ ) {
                            int bt = obt + ibt;
                            result[ ibt ] += X[ bt * C + i ] * w;
                        }
                    }

                    for ( int ibt = 0; ibt < LOOP_UNROLL; ibt++ ) {
                        int bt = obt + ibt;
                        Y[ bt * OC + o ] = result[ ibt ];
                    }
                }
            }
        }

        void backward( float* dinp, float* dweight, float* dbias, const float* dout, const float* inp, const float* weight,
            int B, int T, int C, int OC ) {
        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    const float* dout_bt = dout + b * T * OC + t * OC;
                    float* dinp_bt = dinp + b * T * C + t * C;
                    for ( int o = 0; o < OC; o++ ) {
                        const float* wrow = weight + o * C;
                        float d = dout_bt[ o ];
                        for ( int i = 0; i < C; i++ ) {
                            dinp_bt[ i ] += wrow[ i ] * d;
                        }
                    }
                }
            }
        #pragma omp parallel for
            for ( int o = 0; o < OC; o++ ) {
                for ( int b = 0; b < B; b++ ) {
                    for ( int t = 0; t < T; t++ ) {
                        const float* dout_bt = dout + b * T * OC + t * OC;
                        const float* inp_bt = inp + b * T * C + t * C;
                        float* dwrow = dweight + o * C;
                        float d = dout_bt[ o ];
                        if ( dbias != NULL ) { dbias[ o ] += d; }
                        for ( int i = 0; i < C; i++ ) {
                            dwrow[ i ] += inp_bt[ i ] * d;
                        }
                    }
                }
            }
        }

        static void registerOperation() {
            OperationRegistry<float, float, CpuDevice>::instance().registerOperation( DeviceType::Cpu, "Cpu::FullyConnectedOp", []() -> std::unique_ptr<OperationBase<float, float, CpuDevice>> {
                return std::make_unique<CpuFullyConnectedOp<float>>();
            } );
        }

        std::string getName() const override {
            return "Cpu::MatMulOp";
        }

    private:
        void forward_naive(
            const Tensor<float, CpuMemoryResource>& input,
            const std::shared_ptr<Tensor<float, CpuMemoryResource>>& weight, 
            const std::shared_ptr < Tensor<float, CpuMemoryResource>>& bias,
            Tensor<float, CpuMemoryResource>& output,
            int B, int T, int C, int OC ) const {
            
            // The most naive implementation of matrix multiplication
            // this serves as an algorithmic reference, and as a fallback for
            // unfriendly input shapes inside matmul_forward(), below.

        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    int bt = b * T + t;
                    for ( int o = 0; o < OC; o++ ) {
                        float val = (bias ? bias->data()[ o ] : 0.0f);
                        for ( int i = 0; i < C; i++ ) {
                            val += input.data()[ bt * C + i ] * weight->data()[ o * C + i ];
                        }
                        output.data()[ bt * OC + o ] = val;
                    }
                }
            }
        }
    };
}