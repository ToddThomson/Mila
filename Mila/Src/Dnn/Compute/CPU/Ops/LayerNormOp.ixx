module;
#include <math.h>
#include <iostream>

export module Compute.CpuLayerNormOp;

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
    class CpuLayerNormOp :public OperationBase<T> {
    public:
        CpuLayerNormOp() : OperationBase<T>( DeviceType::kCpu, OperationType::kLayerNorm ) {}

        void forward( 
            const std::shared_ptr<Tensor<T>>& input,
            const std::vector<std::shared_ptr<Tensor<T>>>& parameters, 
            std::shared_ptr<Tensor<T>>& output, 
            std::vector<std::shared_ptr<Tensor<T>>>& output_attributes ) const override {

	        auto weight = parameters[ 0 ];
			auto bias = parameters[ 1 ];

            auto mean = output_attributes[ 0 ];
			auto rstd = output_attributes[ 1 ];

			int B = input->shape()[ 0 ];
			int T = input->shape()[ 1 ];
			int C = input->shape()[ 2 ];

            float eps = 1e-5f;

            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    // seek to the input position inp[b,t,:]
                    // TJT: was float* x = inp + b * T * C_ + t * C_;
                    int input_offset = b * T * C + t * C;

                    // calculate the mean
                    float m = 0.0f;
                    for ( int i = 0; i < C; i++ ) {
                        m += input->data()[ input_offset + i ];
                    }
                    m = m / C;

                    // calculate the variance (without any bias correction)
                    float v = 0.0f;
                    for ( int i = 0; i < C; i++ ) {
                        float xshift = input->data()[ input_offset + i ] - m;
                        v += xshift * xshift;
                    }
                    v = v / C;

                    // calculate the rstd
                    float s = 1.0f / sqrtf( v + eps );

                    // seek to the output position in out[b,t,:]
                    // TJT: was float* out_bt = out + b * T_ * C_ + t * C_;
                    int out_offset = b * T * C + t * C;

                    for ( int i = 0; i < C; i++ ) {
                        float n = (s * (input->data()[ input_offset + i ] - m)); // normalized output
                        float o = n * weight->data()[ i ] + bias->data()[ i ]; // scale and shift it
                        output->data()[ out_offset + i ] = o; // write
                    }

                    // TJT: only if is_training_ cache the mean and rstd for the backward pass later
                    mean->data()[ b * T + t ] = m;
                    rstd->data()[ b * T + t ] = s;
                }
            }
        }

        void backward( 
            float* dinp, 
            float* dweight, float* dbias, 
            float* dout, 
            float* inp, float* weight, float* mean, float* rstd, 
            int B, int T, int C ) {
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    float* dout_bt = dout + b * T * C + t * C;
                    float* inp_bt = inp + b * T * C + t * C;
                    float* dinp_bt = dinp + b * T * C + t * C;
                    float mean_bt = mean[ b * T + t ];
                    float rstd_bt = rstd[ b * T + t ];

                    float dnorm_mean = 0.0f;
                    float dnorm_norm_mean = 0.0f;
                    for ( int i = 0; i < C; i++ ) {
                        float norm_bti = (inp_bt[ i ] - mean_bt) * rstd_bt;
                        float dnorm_i = weight[ i ] * dout_bt[ i ];
                        dnorm_mean += dnorm_i;
                        dnorm_norm_mean += dnorm_i * norm_bti;
                    }
                    dnorm_mean = dnorm_mean / C;
                    dnorm_norm_mean = dnorm_norm_mean / C;

                    for ( int i = 0; i < C; i++ ) {
                        float norm_bti = (inp_bt[ i ] - mean_bt) * rstd_bt;
                        float dnorm_i = weight[ i ] * dout_bt[ i ];
                        dbias[ i ] += dout_bt[ i ];
                        dweight[ i ] += norm_bti * dout_bt[ i ];
                        float dval = 0.0f;
                        dval += dnorm_i;
                        dval -= dnorm_mean;
                        dval -= norm_bti * dnorm_norm_mean;
                        dval *= rstd_bt;
                        dinp_bt[ i ] += dval;
                    }
                }
            }
        }

        static void registerOperation() {
            OperationRegistry<float>::instance().registerOperation( "CPU", "Cpu::LayerNormOp", []() -> std::shared_ptr<OperationBase<float>> {
                return std::make_shared<CpuLayerNormOp<float>>();
                } );
        }

		std::string getName() const override {
			return "Cpu::LayerNormOp";
		}
	};
}