/**
 * @file Residual.ixx
 * @brief Implementation of the Residual connection module for neural networks.
 *
 * Provides a flexible implementation of residual connections which can be configured
 * with different connection types and scaling factors. Supports automatic dimension
 * matching via projection layers.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>

export module Dnn.Modules.Residual;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;

import Dnn.Modules.Linear;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
	using namespace Mila::Dnn::Serialization;

    /**
     * @brief A class implementing a residual connection module.
     *
     * Residual connections help deep neural networks avoid vanishing gradients by
     * providing shortcut connections. The basic formula is y = x + F(x), where F
     * is a differentiable function (usually a sequence of neural network layers).
     *
     * This implementation supports three types of residual connections:
     * 1. Addition: y = x + F(x)
     * 2. Scaled Addition: y = x + alpha*F(x), where alpha is a scaling factor
     * 3. Gated: y = g*x + (1-g)*F(x), where g is a learnable parameter
     *
     * When input and output dimensions don't match, an optional projection layer
     * can be automatically added to make the dimensions compatible.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidFloatTensorTypes<TInput, TOutput>
    class Residual : public Module<TDeviceType, TInput, TOutput> {
    public:
        /**
         * @brief Memory resource type used for tensors, selected based on device type.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;

        /**
         * @brief Alias for base module type.
         */
        using ModuleBase = Module<TDeviceType, TInput, TOutput>;

        /**
         * @brief Constructs a new Residual module with a device name.
         *
         * Creates a new DeviceContext internally using the provided device name.
         * This constructor is useful for creating standalone modules without
         * pre-existing device contexts.
         *
         * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
         * @param config Configuration parameters for the Residual module.
         * @throws std::invalid_argument If the device name is invalid or the configuration is invalid
         * @throws std::runtime_error If device type doesn't match template parameter TDeviceType or inner module type mismatch
         */
        explicit Residual( const std::string& device_name, const ResidualConfig& config )
            : ModuleBase( std::make_shared<DeviceContext>( device_name ), config ), config_( config ) {

            config.validate();
            createOperation();
        }

        /**
         * @brief Constructs a new Residual module with a provided device context.
         *
         * Uses a pre-existing DeviceContext instance. This constructor is useful when integrating
         * the module into a larger network that shares device contexts across modules.
         *
         * @param device_context The device context to use for this module.
         * @param config Configuration parameters for the Residual module.
         * @throws std::invalid_argument If device_context is null or configuration is invalid
         * @throws std::runtime_error If device context type doesn't match template parameter TDeviceType or inner module type mismatch
         */
        explicit Residual( std::shared_ptr<DeviceContext> device_context, const ResidualConfig& config )
            : ModuleBase( device_context, config ), config_( config ) {

            config.validate();
            createOperation();
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * Counts the total number of trainable parameters in the residual module,
         * including the inner module, projection layer (if present), and gating
         * parameters (if using gated connections).
         *
         * @return size_t The total number of parameters.
         */
        size_t parameterCount() const override {
            return 0;
        }

        /**
         * @brief Performs the forward pass of the Residual connection.
         *
         * Applies the residual transformation based on the configured connection type:
         * - Addition: y = x + F(x)
         * - Scaled Addition: y = x + alpha*F(x)
         * - Gated: y = g*x + (1-g)*F(x)
         *
         * Handles projection when input and inner module dimensions don't match.
         *
         * @param input The input tensor to be processed.
         * @param output The output tensor where the results will be stored.
         * @throws std::runtime_error If dimensions don't match and projection is disabled.
         */
        void forward( const Tensor<TInput, MR>& input, Tensor<TOutput, MR>& output ) {
			operation_->forward( input, parameters_, output, output_state_ );
        }

        /**
         * @brief Performs the backward pass of the Residual connection.
         *
         * Computes gradients for the input tensor and parameters based on the output gradients.
         * Handles backpropagation through the inner module and projection layer (if present).
         *
         * @param input The input tensor from the forward pass.
         * @param output_grad The gradient of loss with respect to the output.
         * @param input_grad The tensor to store gradients with respect to input.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TOutput, MR>& output_grad,
            Tensor<TInput, MR>& input_grad ) {
        }

        /**
         * @brief Serializes the module state to a ZIP archive.
         *
         * Saves the state of the inner module, projection layer (if present),
         * and gating parameters (if used) to the provided ZIP archive.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( ModelArchive& zip ) const override {
            // TODO:
        }

        /**
         * @brief Deserializes the module state from a ZIP archive.
         *
         * Loads the state of the inner module, projection layer (if present),
         * and gating parameters (if used) from the provided ZIP archive.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( ModelArchive& archive ) override {
        }

        /**
         * @brief Converts the module information to a human-readable string.
         *
         * Includes detailed information about the module configuration including:
         * - Module name
         * - Connection type
         * - Scaling factor (for scaled addition)
         * - Projection status
         * - Inner module information
         *
         * @return std::string A string representation of the module information.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Residual: " << this->getName() << std::endl;

            return oss.str();
        }

    private:
        /**
         * @brief Configuration for the Residual module.
         */
        ResidualConfig config_;


        /**
         * @brief Collection of trainable parameters.
         */
        std::vector<std::shared_ptr<ITensor>> parameters_;

        /**
         * @brief Gradients for trainable parameters.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameter_grads_;

        /**
         * @brief Output state tensors for backward pass.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> output_state_;

        /**
         * @brief Binary operation for standard and scaled residual connections.
         */
        std::shared_ptr<BinaryOperation<TDeviceType, TInput, TOutput>> operation_;


        /**
         * @brief Creates an appropriate operation based on the connection type.
         *
         * Instantiates the correct operation implementation based on the configured
         * connection type (Addition, ScaledAddition, or Gated) and device type.
         */
        void createOperation() {

            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createBinaryOperation<DeviceType::Cpu, TInput, TOutput, TOutput>(
                    "Cpu::ResidualOp",
                    this->getDeviceContext(),
                    config_ );

                operation_ = std::static_pointer_cast<BinaryOperation<DeviceType::Cpu, TInput, TOutput, TOutput>>(base_op);
            }
        }

    };

    /**
     * @brief Type alias for CPU-based residual module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CpuResidual = Residual<DeviceType::Cpu, TInput, TOutput>;

    /**
     * @brief Type alias for CUDA-based residual module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CudaResidual = Residual<DeviceType::Cuda, TInput, TOutput>;
}