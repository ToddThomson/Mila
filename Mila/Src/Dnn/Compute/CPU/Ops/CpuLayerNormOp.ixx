module;
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#ifdef USE_OMP
#include <omp.h>
#endif


export module Compute.CpuLayerNormOp;

import Dnn.Tensor;
import Compute.OperationBase;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
	export
	template<typename T>
    class CpuLayerNormOp : public OperationBase<T,CpuMemoryResource> {
    public:
        CpuLayerNormOp() : OperationBase<T,CpuMemoryResource>( DeviceType::Cpu, OperationType::LayerNormOp ) {}

        void forward( 
            const Tensor<T,CpuMemoryResource>& input,
            const std::vector<std::shared_ptr<Tensor<T,CpuMemoryResource>>>& parameters, 
            Tensor<T,CpuMemoryResource>& output, 
            std::vector<std::shared_ptr<Tensor<T, CpuMemoryResource>>>& output_cache ) const override {

			const T* X = input.data();
		    T* Y = output.data();

	        const T* weight = parameters[ 0 ]->data();
			const T* bias = parameters[ 1 ]->data();

            T* mean = output_cache[ 0 ]->data();
			T* rstd = output_cache[ 1 ]->data();

			// B: batch size, T: sequence length, C: number of channels
			int B = input.shape()[ 0 ];
			int T = input.shape()[ 1 ];
			int C = input.shape()[ 2 ];

			// TODO: make this a parameter
            float eps = 1e-5f;

            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    // seek to the input position inp[b,t,:]
                    // TJT: was float* x = inp + b * T * C_ + t * C_;
                    int input_offset = b * T * C + t * C;

                    // calculate the mean
                    float m = 0.0f;
                    for ( int i = 0; i < C; i++ ) {
                        m += input.data()[ input_offset + i ];
                    }
                    m = m / C;

                    // calculate the variance (without any bias correction)
                    float v = 0.0f;
                    for ( int i = 0; i < C; i++ ) {
                        float xshift = X[ input_offset + i ] - m;
                        v += xshift * xshift;
                    }
                    v = v / C;

                    // calculate the rstd
                    float s = 1.0f / sqrtf( v + eps );

                    // seek to the output position in out[b,t,:]
                    // TJT: was float* out_bt = out + b * T_ * C_ + t * C_;
                    int out_offset = b * T * C + t * C;

                    for ( int i = 0; i < C; i++ ) {
                        float n = (s * (X[ input_offset + i ] - m)); // normalized output
                        float o = n * weight[ i ] + bias[ i ]; // scale and shift it
                        Y[ out_offset + i ] = o;
                    }

                    // TJT: only if is_training_ cache the mean and rstd for the backward pass later
                    mean[ b * T + t ] = m;
                    rstd[ b * T + t ] = s;
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
            OperationRegistry<float,CpuMemoryResource>::instance().registerOperation( DeviceType::Cpu, "Cpu::LayerNormOp", []() -> std::unique_ptr<OperationBase<float,CpuMemoryResource>> {
                return std::make_unique<CpuLayerNormOp<float>>();
                } );
        }

		std::string getName() const override {
			return "Cpu::LayerNormOp";
		}
	};

	export bool registered_ = (CpuLayerNormOp<float>::registerOperation(), true);
}

