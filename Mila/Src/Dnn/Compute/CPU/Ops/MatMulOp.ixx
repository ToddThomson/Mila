module;
#include <math.h>
#include <iostream>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuMatMulOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.OperationType;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    export
    template<typename T>
    class CpuMatMulOp :public OperationBase<T> {
    public:

		CpuMatMulOp() : OperationBase<T>( DeviceType::kCpu, OperationType::kMatMulOp ) {}

        void forward(
            const std::shared_ptr<Tensor<T>>& input,
            const std::vector<std::shared_ptr<Tensor<T>>>& input_attributes,
            std::shared_ptr<Tensor<T>>& output,
            std::vector<std::shared_ptr<Tensor<T>>>& output_attributes ) const override {
            auto weight = input_attributes[ 0 ];
            auto bias = input_attributes[ 1 ];

            int B = input->shape()[ 0 ];
            int T = input->shape()[ 1 ];
            int C = input->shape()[ 2 ];
			int OC = weight->shape()[ 0 ];

            const int LOOP_UNROLL = 8;
            if ( B * T % LOOP_UNROLL != 0 ) {
                forward_naive( input, weight, bias, output, B, T, C, OC );
                return;
            }

            #pragma omp parallel for
            for ( int obt = 0; obt < B * T; obt += LOOP_UNROLL ) {
                for ( int o = 0; o < OC; o++ ) {
                    float result[ LOOP_UNROLL ];
                    for ( int ibt = 0; ibt < LOOP_UNROLL; ibt++ ) {
                        result[ ibt ] = (bias->data() != nullptr ) ? bias->data()[ o ] : 0.0f;
                    }
                    
                    for ( int i = 0; i < C; i++ ) {
                        float w = weight->data()[ i + o * C ];
                        for ( int ibt = 0; ibt < LOOP_UNROLL; ibt++ ) {
                            int bt = obt + ibt;
                            result[ ibt ] += input->data()[ bt * C + i ] * w;
                        }
                    }
                    
                    for ( int ibt = 0; ibt < LOOP_UNROLL; ibt++ ) {
                        int bt = obt + ibt;
                        output->data()[ bt * OC + o ] = result[ ibt ];
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
            OperationRegistry<float>::instance().registerOperation( "CPU", "Cpu::MatMulOp", []() -> std::shared_ptr<OperationBase<float>> {
                return std::make_shared<CpuMatMulOp<float>>();
                } );
            }

        std::string getName() const override {
            return "Cpu::MatMulOp";
        }

    private:
        void forward_naive( 
            const std::shared_ptr<Tensor<float>>& input, 
            const std::shared_ptr<Tensor<float>>& weight, const std::shared_ptr<Tensor<float>>& bias,
			std::shared_ptr<Tensor<float>>& output,
            int B, int T, int C, int OC ) const {
            // the most naive implementation of matrix multiplication
            // this serves as an algorithmic reference, and as a fallback for
            // unfriendly input shapes inside matmul_forward(), below.
            #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    int bt = b * T + t;
                    for ( int o = 0; o < OC; o++ ) {
                        float val = (bias->data() != NULL) ? bias->data()[ o ] : 0.0f;
                        for ( int i = 0; i < C; i++ ) {
                            val += input->data()[ bt * C + i ] * weight->data()[ o * C + i ];
                        }
                        output->data()[ bt * OC + o ] = val;
                    }
                }
            }
        }
    };
}