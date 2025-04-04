/**
 * @file CpuFullyConnectedOp.ixx
 * @brief Implementation of the CPU-based Fully Connected operation for neural networks.
 */

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
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TPrecision The data type used for computation and output (defaults to the input type).
     */
    export
        template<typename TInput, typename TPrecision = TInput>
    class CpuFullyConnectedOp : public UnaryOperation<TInput, TPrecision, DeviceType::Cpu> {
    public:
        /**
         * @brief Constructs a new CPU Fully Connected operation.
         *
         * Initializes the operation with the CPU device type and FullyConnectedOp operation type.
         */
        CpuFullyConnectedOp()
            : UnaryOperation<TInput, TPrecision, DeviceType::Cpu>( DeviceType::Cpu, OperationType::FullyConnectedOp ) {}

        /**
         * @brief Performs the forward pass of the Fully Connected operation.
         *
         * Computes the matrix multiplication between input and weights, adds bias if provided,
         * and stores the result in the output tensor. Uses loop unrolling for performance optimization
         * when possible, otherwise falls back to a naive implementation.
         *
         * @param input Input tensor of shape [B, TElementType, C] where B is batch size, TElementType is sequence length, and C is input feature dimension.
         * @param parameters Vector of parameter tensors [weight, bias] where weight is of shape [OC, C] and bias (optional) is of shape [OC].
         * @param properties Additional attributes for the operation.
         * @param output Output tensor of shape [B, TElementType, OC] where OC is output feature dimension.
         * @param output_state Cache for intermediate results (not used in this operation).
         */
        void forward(
            const Tensor<TInput, HostMemoryResource>& input,
            const std::vector<std::shared_ptr<Tensor<TPrecision, HostMemoryResource>>>& parameters,
            const OperationAttributes& properties,
            Tensor<TPrecision, HostMemoryResource>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, HostMemoryResource>>>& output_state ) const override {
            auto X = input.raw_data();
            auto Y = output.raw_data();

            auto weight = parameters[ 0 ];
            std::shared_ptr<Tensor<TPrecision, HostMemoryResource>> bias = { nullptr };

            if ( parameters.size() == 2 ) {
                bias = parameters[ 1 ];
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
                        float w = weight->raw_data()[ i + o * C ];
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
         * @param TElementType Sequence length.
         * @param C Input feature dimension.
         * @param OC Output feature dimension.
         */
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

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cpu::MatMulOp").
         */
        std::string getName() const override {
            return "Cpu::MatMulOp";
        }

        /**
         * @brief Gets the class name of this operation.
         *
         * @return const std::string& The class name of the operation.
         */
        static const std::string& className() {
            static std::string name = "Cpu::FullyConnectedOp";
            return name;
        }

    private:
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
         * @param TElementType Sequence length.
         * @param C Input feature dimension.
         * @param OC Output feature dimension.
         */
        void forward_naive(
            const Tensor<float, HostMemoryResource>& input,
            const std::shared_ptr<Tensor<float, HostMemoryResource>>& weight,
            const std::shared_ptr<Tensor<float, HostMemoryResource>>& bias,
            Tensor<float, HostMemoryResource>& output,
            int B, int T, int C, int OC ) const {

            // The most naive implementation of matrix multiplication
            // this serves as an algorithmic reference, and as a fallback for
            // unfriendly input shapes inside matmul_forward(), below.

        #pragma omp parallel for collapse(2)
            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    int bt = b * T + t;
                    for ( int o = 0; o < OC; o++ ) {
                        float val = (bias ? bias->raw_data()[ o ] : 0.0f);
                        for ( int i = 0; i < C; i++ ) {
                            val += input.raw_data()[ bt * C + i ] * weight->raw_data()[ o * C + i ];
                        }
                        output.raw_data()[ bt * OC + o ] = val;
                    }
                }
            }
        }
    };

    /**
     * @brief Class responsible for registering the CpuFullyConnectedOp operation.
     *
     * The CpuFullyConnectedOpRegistrar class registers the CpuFullyConnectedOp operation with the OperationRegistry.
     * It associates the operation name "Cpu::FullyConnectedOp" with a factory function that creates
     * instances of CpuFullyConnectedOp.
     */
    export class CpuFullyConnectedOpRegistrar {
    public:
        /**
         * @brief Registers the CpuFullyConnectedOp operation with the OperationRegistry.
         *
         * This function registers the CpuFullyConnectedOp operation for the CPU device type
         * with the OperationRegistry. It associates the operation name "Cpu::FullyConnectedOp"
         * with a factory function that creates instances of CpuFullyConnectedOp.
         */
        static void registerOperations() {
            const std::string opName = "Cpu::FullyConnectedOp";

            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cpu>(
                opName,
                []() -> std::shared_ptr<OperationBase<float, float, DeviceType::Cpu>> {
                    return std::make_shared<CpuFullyConnectedOp<float, float>>();
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