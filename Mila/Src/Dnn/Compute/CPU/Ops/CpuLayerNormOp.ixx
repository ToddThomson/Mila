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
import Compute.CpuDevice;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief CPU implementation of the Layer Normalization operation.
     */
    export
    class CpuLayerNormOp : public OperationBase<float, float, CpuDevice> {
    public:
        /**
         * @brief Constructor for CpuLayerNormOp.
         */
        CpuLayerNormOp() : OperationBase<float, float, CpuDevice>( DeviceType::Cpu, OperationType::LayerNormOp ) {}

        /**
         * @brief Forward pass for the Layer Normalization operation.
         * 
         * @param input Input tensor.
         * @param parameters Vector of parameters (weight and bias).
         * @param output Output tensor.
         * @param output_state Vector of output state tensors (mean and rstd).
         */
        void forward(
            const Tensor<float>& input,
            const std::vector<std::shared_ptr<Tensor<float>>>& parameters,
            Tensor<float>& output,
            std::vector<std::shared_ptr<Tensor<float>>>& output_state ) const override {

            const float* X = input.data();
            float* Y = output.data();

            const float* weight = parameters[ 0 ]->data();
            const float* bias = parameters[ 1 ]->data();

            float* mean = output_state[ 0 ]->data();
            float* rstd = output_state[ 1 ]->data();

            // B: batch size, T: sequence length, C: number of channels
            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];

            // TODO: make this a parameter
            float eps = 1e-5f;

            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    // seek to the input position inp[b,t,:]
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
                    int out_offset = b * T * C + t * C;

                    for ( int i = 0; i < C; i++ ) {
                        float n = (s * (X[ input_offset + i ] - m)); // normalized output
                        float o = n * weight[ i ] + bias[ i ]; // scale and shift it
                        Y[ out_offset + i ] = o;
                    }

                    // cache the mean and rstd for the backward pass later
                    mean[ b * T + t ] = m;
                    rstd[ b * T + t ] = s;
                }
            }
        }

        /**
         * @brief Backward pass for the Layer Normalization operation.
         * 
         * @param dinp Gradient of the input.
         * @param dweight Gradient of the weight.
         * @param dbias Gradient of the bias.
         * @param dout Gradient of the output.
         * @param inp Input tensor.
         * @param weight Weight tensor.
         * @param mean Mean tensor.
         * @param rstd Reciprocal of the standard deviation tensor.
         * @param B Batch size.
         * @param T Sequence length.
         * @param C Number of channels.
         */
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

        /**
         * @brief Registers the Layer Normalization operation in the operation registry.
         */
        static void registerOperation() {
            OperationRegistry<float, float, CpuDevice>::instance().registerOperation( DeviceType::Cpu, "Cpu::LayerNormOp", []() -> std::unique_ptr<OperationBase<float, float, CpuDevice>> {
                return std::make_unique<CpuLayerNormOp>();
                } );
        }

        /**
         * @brief Gets the name of the operation.
         * 
         * @return The name of the operation.
         */
        std::string getName() const override {
            return "Cpu::LayerNormOp";
        }
    };
}