/**
 * @file CpuLayerNormOp.ixx
 * @brief Implementation of the CPU-based Layer Normalization operation for neural networks.
 */

module;
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuLayerNormOp;

import Dnn.Modules.LayerNorm;
import Dnn.Tensor;
import Dnn.ConfigurationBase;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.CpuMemoryResource;
import Compute.MemoryResource;
import Compute.CpuDevice;
import Compute.Precision;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief CPU implementation of the Layer Normalization operation for neural networks.
     *
     * This class provides a CPU-based implementation of the Layer Normalization operation,
     * which normalizes inputs across the features dimension for each sample in a batch.
     * Layer normalization helps stabilize training by reducing internal covariate shift
     * and is commonly used in transformer architectures and other deep neural networks.
     *
     * The operation normalizes each input vector independently, unlike batch normalization
     * which normalizes across the batch dimension.
     *
     * CPU operations always use full precision regardless of policy settings.
     *
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TDataType The data type used for computation and output (defaults to the input type).
     */
    export class CpuLayerNormOp : public UnaryOperation<DeviceType::Cpu, float> {
    public:
        using MR = typename CpuDevice::MR;
        using OperationBase = UnaryOperation<DeviceType::Cpu, float>;

        /**
         * @brief Constructs a new CPU Layer Normalization operation with the default device context.
         *
         * CPU operations always use full precision regardless of policy settings.
         *
         * @param precision_policy Ignored for CPU operations, as they always use full precision.
         */
        CpuLayerNormOp( const LayerNormConfig& config )
            : OperationBase( OperationType::LayerNormOp ), config_( config ) {}

        /**
         * @brief Constructs a new CPU Layer Normalization operation with a specific device context.
         *
         * CPU operations always use full precision regardless of policy settings.
         *
         * @param context The device context to use for this operation.
         * @param precision_policy Ignored for CPU operations, as they always use full precision.
         * @throws std::runtime_error If the context is not for a CPU device.
         */
        CpuLayerNormOp( std::shared_ptr<DeviceContext> context, const LayerNormConfig& config )
            : OperationBase( OperationType::LayerNormOp, context ), config_( config ) {}

        /**
         * @brief Performs the forward pass of the Layer Normalization operation.
         *
         * Normalizes each input vector across the feature dimension, then applies
         * a learnable scaling factor and bias.
         *
         * @param input Input tensor of shape [B, TDataType, C] where B is batch size, TDataType is sequence length, and C is feature dimension.
         * @param parameters Vector of parameter tensors [weight, bias] where weight and bias are of shape [C].
         * @param attributes Additional attributes for the operation.
         * @param output Output tensor of the same shape as input, containing the normalized values.
         * @param output_state Cache for intermediate results [mean, rstd] used in the backward pass.
         */
        void forward(
            const Tensor<float, MR>& input,
            const std::vector<std::shared_ptr<ITensor>>& parameters,
            Tensor<float, MR>& output,
            std::vector<std::shared_ptr<Tensor<float, MR>>>& output_state ) const override {

            // Verify we're operating on CPU memory
            if ( this->getDeviceContext()->getDevice()->getDeviceType() != DeviceType::Cpu ) {
                throw std::runtime_error( "CpuLayerNormOp::forward can only be executed on CPU memory" );
            }

            const float* X = input.data();
            float* Y = output.data();

            const float* weight = static_cast<const float*>( parameters[ 0 ]->data() );
            const float* bias = (parameters.size() > 1 && parameters[ 1 ]) ? static_cast<const float*>(parameters[ 1 ]->data()) : nullptr;

            float* mean = output_state[ 0 ]->data();
            float* rstd = output_state[ 1 ]->data();

            // B: batch size, T: sequence length, C: number of channels
            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];

            float eps = config_.getEpsilon();

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
                        float o = n * weight[ i ];
                        
                        if ( bias ) {
							auto temp_bias = bias[ i ];
                            o += bias[ i ];
                        }
                        
                        Y[ out_offset + i ] = o;
                    }

                    // cache the mean and rstd for the backward pass later
                    mean[ b * T + t ] = m;
                    rstd[ b * T + t ] = s;
                }
            }
        }

        /**
         * @brief Performs the backward pass of the Layer Normalization operation.
         *
         * Computes gradients with respect to inputs, weights, and biases based
         * on the output gradient and the forward pass results.
         *
         * @param dinp Pointer to the gradient buffer for input.
         * @param dweight Pointer to the gradient buffer for weight parameters.
         * @param dbias Pointer to the gradient buffer for bias parameters.
         * @param dout Pointer to the gradient buffer from the output.
         * @param inp Pointer to the original input values.
         * @param weight Pointer to the weight parameters.
         * @param mean Pointer to the mean values computed during forward pass.
         * @param rstd Pointer to the reciprocal standard deviation values computed during forward pass.
         * @param B Batch size.
         * @param TDataType Sequence length.
         * @param C Number of features/channels.
         */
        void backward(
            float* dinp,
            float* dweight, float* dbias,
            float* dout,
            float* inp, float* weight, float* mean, float* rstd,
            int B, int T, int C ) {

            // Verify we're operating on CPU memory
            if ( this->getDeviceContext()->getDevice()->getDeviceType() != DeviceType::Cpu ) {
                throw std::runtime_error( "CpuLayerNormOp::backward can only be executed on CPU memory" );
            }

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
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cpu::LayerNormOp").
         */
        std::string getName() const override {
            return "Cpu::LayerNormOp";
        }

        private:
            LayerNormConfig config_; ///< Configuration for the LayerNorm operation.
    };

    /**
     * @brief Class responsible for registering the CpuLayerNormOp operation.
     *
     * The CpuLayerNormOpRegistrar class registers the CpuLayerNormOp operation with the OperationRegistry.
     * It associates the operation name "Cpu::LayerNormOp" with a factory function that creates
     * instances of CpuLayerNormOp.
     */
    export class CpuLayerNormOpRegistrar {
    public:
        /**
         * @brief Registers the CpuLayerNormOp operation with the OperationRegistry.
         *
         * This function registers the CpuLayerNormOp operation for the CPU device type
         * with the OperationRegistry. It associates the operation name "Cpu::LayerNormOp"
         * with a factory function that creates instances of CpuLayerNormOp.
         */
        static void registerOperations() {
            const std::string opName = "Cpu::LayerNormOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, float, float>(
                opName,
                []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, float, float>> {
                    const auto& layerNormConfig = dynamic_cast<const LayerNormConfig&>(config);
                    return context ? std::make_shared<CpuLayerNormOp>( context, layerNormConfig )
                        : std::make_shared<CpuLayerNormOp>( layerNormConfig );
                }
            );
        }

        /**
         * @brief Self-registration mechanism that registers the operation during startup.
         *
         * This static member ensures the operation is registered when the program starts
         * without requiring explicit registration calls.
         */
        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}