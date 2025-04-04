/**
 * @file CpuResidualOp.ixx
 * @brief Implementation of the CPU-based residual operation for neural networks.
 */

module;
#include <memory>
#include <vector>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuResidualOp;

import Dnn.Tensor;
import Compute.DeviceType;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.OperationBase;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
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
     * @tparam TPrecision The data type used for computation and output (defaults to the input type).
     */
    export
        template<typename TInput = float, typename TPrecision = float>
    class CpuResidualOp :public BinaryOperation<float, float, DeviceType::Cpu> {
    public:
        /**
         * @brief Constructs a new CPU Residual operation.
         *
         * Initializes the operation with the CPU device type and ResidualOp operation type.
         */
        CpuResidualOp() : BinaryOperation<float, float, DeviceType::Cpu>( DeviceType::Cpu, OperationType::ResidualOp ) {}

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
         * @param output_cache Cache for storing intermediate results (used in backward pass).
         */
        void forward(
            const Tensor<float, HostMemoryResource>& input_a,
            const Tensor<float, HostMemoryResource>& input_b,
            const std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& parameters,
            const OperationAttributes& attributes,
            Tensor<float, HostMemoryResource>& output,
            std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>>& output_cache ) const override {
            auto A = input_a.raw_data();
            auto B = input_b.raw_data();
            auto Y = output.raw_data();
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
         * @param dinp1 Pointer to the gradient buffer for the first input.
         * @param dinp2 Pointer to the gradient buffer for the second input.
         * @param dout Pointer to the gradient buffer for the output.
         * @param N Size of the gradient buffers.
         */
        void backward( float* dinp1, float* dinp2, float* dout, int N ) {
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

            OperationRegistry::instance().registerOperation<float, float, DeviceType::Cpu>(
                opName,
                []() -> std::shared_ptr<OperationBase<float, float, DeviceType::Cpu>> {
                    return std::make_shared<CpuResidualOp<float, float>>();
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
