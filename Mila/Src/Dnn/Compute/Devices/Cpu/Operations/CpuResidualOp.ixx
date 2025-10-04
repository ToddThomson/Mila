/**
 * @file CpuResidualOp.ixx
 * @brief Implementation of the CPU-based residual operation for neural networks.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#define _USE_MATH_DEFINES
#include <math.h>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuResidualOp;

import Dnn.Modules.Residual;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.ConfigurationBase;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.OperationBase;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CpuDevice;

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU implementation of the residual operation for neural networks.
     *
     * This class provides a CPU-based implementation of the residual operation,
     * which performs element-wise addition of two input tensors.
     * It is commonly used in residual connections in neural network architectures
     * such as ResNet and Transformers to help with gradient flow.
     *
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TDataType The data type used for computation and output (defaults to the input type).
     */
    export class CpuResidualOp : public BinaryOperation<DeviceType::Cpu, float, float> {
    public:
        using MR = typename CpuDevice::MR;
        using OperationBase = BinaryOperation<DeviceType::Cpu, float, float>;

        /**
         * @brief Constructs a new CPU Residual operation with the default device context.
         *
         * Initializes the operation with a CPU device context.
         */
        CpuResidualOp( const ResidualConfig& config )
            : OperationBase( OperationType::ResidualOp ), config_( config ) {}

        /**
         * @brief Constructs a new CPU Residual operation with a specific device context.
         *
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the context is not for a CPU device.
         */
        CpuResidualOp( std::shared_ptr<DeviceContext> context, const ResidualConfig& config )
            : OperationBase( OperationType::ResidualOp, context ), config_( config ) {
        }

        /**
         * @brief Performs the forward pass of the residual operation.
         *
         * Adds two input tensors element-wise and stores the result in the output tensor.
         *
         * @param input_a The first input tensor.
         * @param input_b The second input tensor.
         * @param parameters Additional parameters (not used in this operation).
         * @param attributes Additional attributes for the operation.
         * @param output The output tensor where the results will be stored.
         * @param output_state Cache for storing intermediate results (used in backward pass).
         */
        void forward(
            const Tensor<float, MR>& input_a,
            const Tensor<float, MR>& input_b,
            const std::vector<std::shared_ptr<ITensor>>& parameters,
            Tensor<float, MR>& output,
            std::vector<std::shared_ptr<Tensor<float, MR>>>& output_state ) const override {

            auto A = input_a.data();
            auto B = input_b.data();
            auto Y = output.data();
            auto N = input_a.size();

            #pragma omp parallel for
            for ( int i = 0; i < N; i++ ) {
                Y[ i ] = A[ i ] + B[ i ];
            }
        }

        /**
         * @brief Performs the backward pass of the residual operation.
         *
         * Computes the gradient with respect to each input by adding the output gradient
         * to each input gradient.
         *
         * @param input_a First input tensor from the forward pass.
         * @param input_b Second input tensor from the forward pass.
         * @param output Output tensor from the forward pass.
         * @param output_gradient Gradient of the loss with respect to the output.
         * @param parameters Parameters used in forward pass (not used in this operation).
         * @param parameter_gradients Gradients for parameters (not used in this operation).
         * @param input_a_gradient Gradient of the loss with respect to input_a.
         * @param input_b_gradient Gradient of the loss with respect to input_b.
         * @param attributes Additional attributes for the operation.
         * @param output_state Cache tensors from forward pass.
         */
        void backward(
            const Tensor<float, MR>& input_a,
            const Tensor<float, MR>& input_b,
            const Tensor<float, MR>& output,
            const Tensor<float, MR>& output_gradient,
            const std::vector<std::shared_ptr<ITensor>>& parameters,
            std::vector<std::shared_ptr<Tensor<float, MR>>>& parameter_gradients,
            Tensor<float, MR>& input_a_gradient,
            Tensor<float, MR>& input_b_gradient,
            const std::vector<std::shared_ptr<Tensor<float, MR>>>& output_state ) const {

            if ( !this->getDeviceContext()->isDeviceType( DeviceType::Cpu ) ) {
                throw std::runtime_error( "CpuResidualOp::backward can only be executed on CPU device." );
            }

            float* dinp1 = input_a_gradient.data();
            float* dinp2 = input_b_gradient.data();
            const float* dout = output_gradient.data();
            int N = input_a.size();

            backward_impl( dinp1, dinp2, dout, N );
        }

        /**
         * @brief Helper method for backward pass implementation.
         *
         * @param dinp1 Pointer to the gradient buffer for the first input.
         * @param dinp2 Pointer to the gradient buffer for the second input.
         * @param dout Pointer to the gradient buffer for the output.
         * @param N Size of the gradient buffers.
         */
        void backward_impl( float* dinp1, float* dinp2, const float* dout, int N ) const {
        #pragma omp parallel for
            for ( int i = 0; i < N; i++ ) {
                dinp1[ i ] += dout[ i ];
                dinp2[ i ] += dout[ i ];
            }
        }

        /**
         * @brief Gets the name of this operation.
         *
         * @return std::string The name of the operation ("Cpu::ResidualOp").
         */
        std::string getName() const override {
            return "Cpu::ResidualOp";
        }

        /**
         * @brief Gets the class name of this operation.
         *
         * @return const std::string& The class name of the operation.
         */
        static const std::string& className() {
            static std::string name = "Cpu::ResidualOp";
            return name;
        }
    };

    /**
     * @brief Class responsible for registering the CpuResidualOp operation.
     *
     * The CpuResidualOpRegistrar class registers the CpuResidualOp operation with the OperationRegistry.
     * It associates the operation name "Cpu::ResidualOp" with a factory function that creates instances of CpuResidualOp.
     */
    export class CpuResidualOpRegistrar {
    public:
        /**
         * @brief Registers the CpuResidualOp operation with the OperationRegistry.
         *
         * This function registers the CpuResidualOp operation for the CPU device type
         * with the OperationRegistry. It associates the operation name "Cpu::ResidualOp"
         * with a factory function that creates instances of CpuResidualOp.
         */
        static void registerOperations() {
            const std::string opName = "Cpu::ResidualOp";

            OperationRegistry::instance().registerBinaryOperation<DeviceType::Cpu, float, float>(
                opName,
                []( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config ) -> std::shared_ptr<BinaryOperation<DeviceType::Cpu, float, float>> {
                    const auto& residualConfig = dynamic_cast<const ResidualConfig&>( config );
                    return context ? std::make_shared<CpuResidualOp>( context, residualConfig )
                        : std::make_shared<CpuResidualOp>( residualConfig );
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
