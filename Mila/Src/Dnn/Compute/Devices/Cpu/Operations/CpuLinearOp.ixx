/**
 * @file CpuLinearOp.ixx
 * @brief Implementation of the CPU-based Fully Connected operation for neural networks.
 */

module;
#include <math.h>
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <sstream>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuLinearOp;

import Dnn.Modules.Linear;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.ConfigurationBase;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CpuDevice;
import Compute.Precision;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief CPU implementation of the Fully Connected operation for neural networks.
     *
     * This class provides a CPU-based implementation of the Fully Connected operation,
     * which performs a matrix multiplication between the input and a weight matrix,
     * optionally adds a bias, and produces an output. This operation implements the
     * standard linear layer commonly used in neural networks.
     *
     * The implementation includes both a performance-optimized version with loop
     * unrolling and a naive fallback implementation for special cases.
     *
     * @tparam float The data type of the input tensor elements.
     * @tparam TDataType The data type used for computation and output (defaults to the input type).
     */
    export class CpuLinearOp : public UnaryOperation<DeviceType::Cpu, float> {
    public:
        using MR = typename CpuDevice::MR;
        using OperationBase = UnaryOperation<DeviceType::Cpu, float>;

        /**
         * @brief Constructs a new CPU Fully Connected operation with the default device context.
         *
         * CPU operations always use full precision regardless of policy settings.
         *
         * @param precision_policy Ignored for CPU operations, as they always use full precision.
         */
        CpuLinearOp( const LinearConfig& config )
            : OperationBase( OperationType::LinearOp ), config_( config ) {}

        /**
         * @brief Constructs a new CPU Fully Connected operation with a specific device context.
         *
         * CPU operations always use full precision regardless of policy settings.
         *
         * @param context The device context to use for this operation.
         * @param precision_policy Ignored for CPU operations, as they always use full precision.
         * @throws std::runtime_error If the context is not for a CPU device.
         */
        CpuLinearOp( std::shared_ptr<DeviceContext> context, const LinearConfig& config )
            : OperationBase( OperationType::LinearOp, context ), config_( config ) {}

        /**
         * @brief Performs the forward pass of the Linear operation.
         *
         * Computes the matrix multiplication between input and weights, adds bias if provided,
         * and stores the result in the output tensor. Uses loop unrolling for performance optimization
         * when possible, otherwise falls back to a naive implementation.
         *
         * @param input Input tensor of shape [B, TDataType, C] where B is batch size, TDataType is sequence length, and C is input feature dimension.
         * @param parameters Vector of parameter tensors [weight, bias] where weight is of shape [OC, C] and bias (optional) is of shape [OC].
         * @param properties Additional attributes for the operation.
         * @param output Output tensor of shape [B, TDataType, OC] where OC is output feature dimension.
         * @param output_state Cache for intermediate results (not used in this operation).
         */
        void forward(
            const Tensor<float, MR>& input,
            const std::vector<std::shared_ptr<ITensor>>& parameters,
            Tensor<float, MR>& output,
            std::vector<std::shared_ptr<Tensor<float, MR>>>& output_state ) const override {

            // Verify we're operating on CPU memory
            if ( this->getDeviceContext()->getDevice()->getDeviceType() != DeviceType::Cpu ) {
                throw std::runtime_error( "CpuLinearOp::forward can only be executed on CPU memory" );
            }

            // FIXME: Remove after testing
            auto t = input.toString( true );

            auto outer_dims = input.rank() - 1;

            if ( outer_dims <= 0 ) {
                throw std::runtime_error( "LinearOp requires input tensor with at least 2 dimensions" );
            }

            const float* X = input.data();
            float* Y = output.data();

            auto weight = std::static_pointer_cast<Tensor<float, MR>>(parameters[ 0 ]);
            std::shared_ptr<Tensor<float, MR>> bias = nullptr;

            if ( parameters.size() > 1 ) {
                bias = std::static_pointer_cast<Tensor<float, MR>>(parameters[ 1 ]);
            }

            int C = input.shape().back();
            int OC = output.shape().back();

			auto input_features = config_.getInputFeatures();
			auto output_features = config_.getOutputFeatures();

            size_t outer_size = 1;
            for ( size_t i = 0; i < outer_dims; i++ ) {
                outer_size *= input.shape()[ i ];
            }

            const int LOOP_UNROLL = 8;
            if ( outer_size % LOOP_UNROLL != 0 ) {
                // TODO: Write a unit test for this case
                forward_naive( input, weight, bias, output, outer_size, C, OC );
                return;
            }

            // Optimized implementation with loop unrolling
        #pragma omp parallel for
            for ( int out_idx = 0; out_idx < outer_size; out_idx += LOOP_UNROLL ) {
                for ( int o = 0; o < OC; o++ ) {
                    float result[ LOOP_UNROLL ];
                    for ( int i_idx = 0; i_idx < LOOP_UNROLL; i_idx++ ) {
                        result[ i_idx ] = (bias ? bias->data()[ o ] : 0.0f);
                    }

                    for ( int i = 0; i < C; i++ ) {
                        float w = weight->data()[ o * C + i ];
                        for ( int i_idx = 0; i_idx < LOOP_UNROLL; i_idx++ ) {
                            int idx = out_idx + i_idx;
                            result[ i_idx ] += X[ idx * C + i ] * w;
                        }
                    }

                    for ( int i_idx = 0; i_idx < LOOP_UNROLL; i_idx++ ) {
                        int idx = out_idx + i_idx;
                        Y[ idx * OC + o ] = result[ i_idx ];
                    }
                }
            }
        }

        /**
         * @brief Performs the backward pass of the Fully Connected operation.
         *
         * Computes gradients with respect to inputs, weights, and biases based
         * on the output gradient.
         *
         * @param dinp Pointer to the gradient buffer for input.
         * @param dweight Pointer to the gradient buffer for weight parameters.
         * @param dbias Pointer to the gradient buffer for bias parameters (can be NULL if no bias is used).
         * @param dout Pointer to the gradient buffer from the output.
         * @param inp Pointer to the original input values.
         * @param weight Pointer to the weight parameters.
         * @param B Batch size.
         * @param TDataType Sequence length.
         * @param C Input feature dimension.
         * @param OC Output feature dimension.
         */
        void backward(
            Tensor<float, MR>& input_grad,
            const std::vector<std::shared_ptr<Tensor<float, MR>>>& parameter_grads,
            const Tensor<float, MR>& output_grad,
            const Tensor<float, MR> input,
            const Tensor<float, MR> weight,
            int B, int T, int C, int OC ) {

            // Verify we're operating on CPU memory
            if ( this->getDeviceContext()->getDevice()->getDeviceType() != DeviceType::Cpu ) {
                throw std::runtime_error( "CpuLinearOp::backward can only be executed on CPU memory" );
            }

            //#pragma omp parallel for collapse(2)
            //    for ( int b = 0; b < B; b++ ) {
            //        for ( int t = 0; t < T; t++ ) {
            //            const float* dout_bt = output_grad + b * T * OC + t * OC;
            //            float* dinp_bt = input_grad + b * T * C + t * C;
            //            for ( int o = 0; o < OC; o++ ) {
            //                const float* wrow = weight + o * C;
            //                float d = dout_bt[ o ];
            //                for ( int i = 0; i < C; i++ ) {
            //                    dinp_bt[ i ] += wrow[ i ] * d;
            //                }
            //            }
            //        }
            //    }
            //#pragma omp parallel for
            //    for ( int o = 0; o < OC; o++ ) {
            //        for ( int b = 0; b < B; b++ ) {
            //            for ( int t = 0; t < T; t++ ) {
            //                const float* dout_bt = output_grad + b * T * OC + t * OC;
            //                const float* inp_bt = input + b * T * C + t * C;
            //                float* dwrow = weight_grad + o * C;
            //                float d = dout_bt[ o ];
            //                if ( dbias_grad != NULL ) { dbias[ o ] += d; }
            //                for ( int i = 0; i < C; i++ ) {
            //                    dwrow[ i ] += inp_bt[ i ] * d;
            //                }
            //            }
            //        }
            //    }
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cpu::LinearOp").
         */
        std::string getName() const override {
            return "Cpu::LinearOp";
        }

    private:
		const LinearConfig config_; ///< Configuration for the linear operation.

        /**
         * @brief Naive implementation of the forward pass for the Fully Connected operation.
         *
         * This is a simple implementation without optimizations that serves as a
         * fallback for cases where the optimized implementation cannot be used.
         *
         * @param input Input tensor.
         * @param weight Weight tensor.
         * @param bias Bias tensor (optional).
         * @param output Output tensor.
         * @param B Batch size.
         * @param TDataType Sequence length.
         * @param C Input feature dimension.
         * @param OC Output feature dimension.
         */
        void forward_naive(
            const Tensor<float, MR>& input,
            const std::shared_ptr<Tensor<float, MR>>& weight,
            const std::shared_ptr<Tensor<float, MR>>& bias,
            Tensor<float, MR>& output,
            int outer_size, int C, int OC ) const {

            // Simple implementation of matrix multiplication with bias
        #pragma omp parallel for
            for ( int idx = 0; idx < outer_size; idx++ ) {
                for ( int o = 0; o < OC; o++ ) {
                    float val = (bias ? bias->data()[ o ] : 0.0f);
                    for ( int i = 0; i < C; i++ ) {
                        val += input.data()[ idx * C + i ] * weight->data()[ o * C + i ];
                    }
                    output.data()[ idx * OC + o ] = val;
                }
            }
        }
    };

    /**
     * @brief Class responsible for registering the CpuLinearOp operation.
     *
     * The CpuLinearOpRegistrar class registers the CpuLinearOp operation with the OperationRegistry.
     * It associates the operation name "Cpu::LinearOp" with a factory function that creates
     * instances of CpuLinearOp.
     */
    export class CpuLinearOpRegistrar {
    public:
        /**
         * @brief Registers the CpuLinearOp operation with the OperationRegistry.
         *
         * This function registers the CpuLinearOp operation for the CPU device type
         * with the OperationRegistry. It associates the operation name "Cpu::LinearOp"
         * with a factory function that creates instances of CpuLinearOp.
         */
        static void registerOperations() {
            const std::string opName = "Cpu::LinearOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, float, float>(
                opName,
                []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, float, float>> {
                    const auto& linearConfig = dynamic_cast<const LinearConfig&>( config );
                    return context ? std::make_shared<CpuLinearOp>( context, linearConfig )
                        : std::make_shared<CpuLinearOp>( linearConfig );
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