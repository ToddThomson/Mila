/**
 * @file Linear.ixx
 * @brief Implementation of the Linear (fully connected) module for neural networks.
 */

module;
#include <miniz.h>
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>

export module Dnn.Modules.Linear;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief A class representing a linear transformation module.
     *
     * The linear module (also known as fully-connected or dense layer) performs
     * a linear transformation of the input data. The operation is defined as:
     * output = input * weight + bias
     *
     * This is a fundamental building block in neural networks that connects
     * every input neuron to every output neuron with learnable weights.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidFloatTensorTypes<TInput, TOutput>
    class Linear : public Module<TDeviceType, TInput, TOutput> {
    public:
        /**
         * @brief Memory resource type used for tensors, selected based on device type.
         *
         * Uses CudaMemoryResource for CUDA devices and CpuMemoryResource for CPU.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;

        /**
         * @brief Alias for base module type.
         */
        using ModuleBase = Module<TDeviceType, TInput, TOutput>;

        /**
         * @brief Constructs a new Linear module with a device name.
         *
         * Creates a new DeviceContext internally using the provided device name.
         * This constructor is useful for creating standalone modules without
         * pre-existing device contexts.
         *
         * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
         * @param config Configuration parameters for the Linear module.
         * @throws std::invalid_argument If the device name is invalid or the configuration is invalid
         * @throws std::runtime_error If device type doesn't match template parameter TDeviceType
         */
        explicit Linear( const std::string& device_name, const LinearConfig& config )
            : ModuleBase( std::make_shared<DeviceContext>( device_name ), config ), config_( config ) {

            config.validate();

            initializeParameters();

            if ( this->isTraining() ) {
                initializeParameterGradients();
            }

            createOperation();
        }

        /**
         * @brief Constructs a new Linear module with a provided device context.
         *
         * Uses a pre-existing DeviceContext instance. This constructor is useful when integrating
         * the module into a larger network that shares device contexts across modules.
         *
         * @param device_context The device context to use for this module.
         * @param config Configuration parameters for the Linear module.
         * @throws std::invalid_argument If device_context is null or configuration is invalid
         * @throws std::runtime_error If device context type doesn't match template parameter TDeviceType
         */
        explicit Linear( std::shared_ptr<DeviceContext> device_context, const LinearConfig& config )
            : ModuleBase( device_context, config ), config_( config ) {

            config.validate();

            initializeParameters();

            if ( this->isTraining() ) {
                initializeParameterGradients();
            }

            createOperation();
        }

        /**
         * @brief Performs the forward pass of the Linear operation.
         *
         * Applies the linear transformation to the input tensor:
         * output = input * weight + bias (if bias is enabled)
         *
         * @param input The input tensor to be transformed.
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TInput, MR>& input, Tensor<TOutput, MR>& output ) {
            operation_->forward( input, parameters_, properties_, output, output_state_ );
        }

        /**
         * @brief Performs the backward pass of the Linear operation.
         *
         * Computes gradients for the input tensor and parameters based on the output gradients.
         *
         * @param input The input tensor from the forward pass.
         * @param output_grad The gradient of loss with respect to the output.
         * @param input_grad The tensor to store gradients with respect to input.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TOutput, MR>& output_grad,
            Tensor<TInput, MR>& input_grad ) {
            operation_->backward(
                input,            // Input tensor
                output_grad,      // Gradient from next layer
                parameters_,      // Original parameters (weight, bias)
                parameter_grads_, // Gradient tensors for parameters
                input_grad,       // Gradient to propagate to previous layer
                properties_,      // Operation properties
                output_state_     // Cached tensors from forward pass
            );
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * Counts the total number of trainable parameters, which includes
         * the weight tensor and, if present, the bias tensor.
         *
         * @return size_t The total number of parameters.
         */
        size_t parameterCount() const override {
            size_t num_params = weight_->size();
            if ( config_.hasBias() ) {
                num_params += bias_->size();
            }
            return num_params;
        }

        /**
         * @brief Retrieves the weight tensor for this linear layer.
         *
         * The weight tensor has shape [output_features, input_features] and
         * is initialized with Xavier/Glorot uniform distribution.
         *
         * @return std::shared_ptr<Tensor<TOutput, MR>> The weight tensor used in the linear transformation.
         */
        std::shared_ptr<Tensor<TOutput, MR>> getWeight() {
            return weight_;
        }

        /**
         * @brief Retrieves the bias tensor if present.
         *
         * The bias tensor has shape [output_features] and is initialized to zeros
         * if bias is enabled in the layer configuration.
         *
         * @return std::optional<std::shared_ptr<Tensor<TOutput, MR>>> An optional containing the bias tensor if bias is enabled, otherwise std::nullopt.
         */
        std::optional<std::shared_ptr<Tensor<TOutput, MR>>> getBias() {
            return config_.hasBias() ? std::optional{ bias_ } : std::nullopt;
        }

        /**
         * @brief Checks whether the module has a bias tensor.
         *
         * @return bool True if the module has a bias tensor, false otherwise.
         */
        bool hasBias() const {
            return config_.hasBias();
        }

        /**
         * @brief Serializes the module state to a ZIP archive.
         *
         * Saves the trainable parameters (weight, bias) to the provided archive.
         * Note: This method is currently a placeholder and needs implementation.
         *
         * @param zip The ZIP archive to save the module state to.
         * @throws std::runtime_error If the serialization fails.
         */
        void save( mz_zip_archive& zip ) const override {
            // Save the state of the parameters
            for ( const auto& [name, tensor] : this->getParameterTensors() ) {
                // Save tensor data to zip archive
                // Implementation will depend on how tensors are serialized
            }
        }

        /**
         * @brief Deserializes the module state from a ZIP archive.
         *
         * Loads the trainable parameters (weight, bias) from the provided archive.
         * Note: This method is currently a placeholder and needs implementation.
         *
         * @param zip The ZIP archive to load the module state from.
         * @throws std::runtime_error If the deserialization fails.
         */
        void load( mz_zip_archive& zip ) override {
            for ( const auto& [name, tensor] : this->getParameterTensors() ) {
                // Load tensor data from zip archive
                // Implementation will depend on how tensors are deserialized
            }
        }

        /**
         * @brief Converts the module information to a human-readable string.
         *
         * Includes detailed information about the module configuration including:
         * - Module name
         * - Input/output features
         * - Device type
         * - Precision policy
         * - Parameter information
         *
         * @return std::string A string representation of the module information.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Linear: " << this->getName() << std::endl;
            oss << "Input features: " << config_.getInputFeatures();
            oss << ", Output features: " << config_.getOutputFeatures() << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

    private:
        /**
         * @brief Configuration for the Linear module.
         */
        LinearConfig config_;

        /**
         * @brief The weight tensor for the linear transformation.
         *
         * Shape is [output_features, input_features] to transform input features
         * to output features through matrix multiplication.
         */
        std::shared_ptr<Tensor<TOutput, MR>> weight_{ nullptr };

        /**
         * @brief The bias tensor added after the matrix multiplication.
         *
         * Shape is [output_features]. This tensor is only used if has_bias_ is true.
         */
        std::shared_ptr<Tensor<TOutput, MR>> bias_{ nullptr };

        /**
         * @brief Collection of trainable parameters for this module.
         *
         * Contains the weight tensor and optionally the bias tensor if has_bias_ is true.
         * These parameters are passed to the underlying operation during forward pass.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameters_;

        /**
         * @brief Gradients for the parameters of this module.
         *
         * Contains gradients for the weight tensor and optionally the bias tensor.
         * These are computed during the backward pass.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameter_grads_;

        /**
         * @brief Cache of intermediate tensors needed for backward pass.
         *
         * Stores tensors that are computed during the forward pass and
         * are needed for gradient computation during backpropagation.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> output_state_;

        /**
         * @brief Additional configuration options for the linear operation.
         *
         * These attributes can modify the behavior of the underlying operation
         * implementation without changing the API.
         */
        OperationAttributes properties_;

        /**
         * @brief The underlying operation that implements the Linear transformation.
         *
         * This operation performs the actual computation for the linear layer,
         * with different implementations for CPU and CUDA devices.
         */
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>> operation_{ nullptr };

        /**
         * @brief Initializes the tensors needed for the Linear operation.
         *
         * Creates and initializes:
         * - weight tensor (initialized with Xavier/Glorot uniform distribution)
         * - bias tensor (initialized to zeros if has_bias_ is true)
         *
         * The tensors are created on the appropriate device (CPU or CUDA)
         * based on the current device context.
         */
        void initializeParameters() {
            parameters_.clear();
            this->parameter_map_.clear();

            size_t input_features = config_.getInputFeatures();
            size_t output_features = config_.getOutputFeatures();
            bool has_bias = config_.hasBias();

            weight_ = std::make_shared<Tensor<TOutput, MR>>(
                std::vector<size_t>{output_features, input_features} );
            weight_->setName( this->getName() + ".weight" );

            xavier<TOutput, MR>( *weight_, input_features, output_features );

            if ( has_bias ) {
                bias_ = std::make_shared<Tensor<TOutput, MR>>(
                    std::vector<size_t>{output_features} );
                bias_->setName( this->getName() + ".bias" );
            }

            // Add tensors to parameters list and map
            parameters_.emplace_back( weight_ );
            this->parameter_map_[ "weight" ] = weight_;

            if ( has_bias ) {
                parameters_.emplace_back( bias_ );
                this->parameter_map_[ "bias" ] = bias_;
            }
        }

        /**
         * @brief Initializes gradient tensors for parameters.
         *
         * Creates tensors to store gradients for weights and biases (if present).
         * These tensors will be populated during backpropagation.
         */
        void initializeParameterGradients() {
            parameter_grads_.clear();

            size_t input_features = config_.getInputFeatures();
            size_t output_features = config_.getOutputFeatures();
            bool has_bias = config_.hasBias();

            auto weight_grad = std::make_shared<Tensor<TOutput, MR>>(
                std::vector<size_t>{output_features, input_features} );
            weight_grad->setName( this->getName() + ".weight_grad" );
            parameter_grads_.push_back( weight_grad );

            if ( has_bias ) {
                auto bias_grad = std::make_shared<Tensor<TOutput, MR>>(
                    std::vector<size_t>{output_features} );
                bias_grad->setName( this->getName() + ".bias_grad" );
                parameter_grads_.emplace_back( bias_grad );
            }
        }

        /**
         * @brief Creates the appropriate Linear operation based on the current device context.
         *
         * This method initializes the operation_ member with the appropriate implementation
         * of Linear for either CPU or CUDA, based on the current device context.
         * It also passes the compute precision policy to the operation.
         *
         * @throws std::runtime_error If the operation creation fails.
         */
        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TInput, TOutput>(
                    "Cpu::LinearOp",
                    this->getDeviceContext(),
                    config_ );
                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TInput, TOutput>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TInput, TOutput>(
                    "Cuda::LinearOp",
                    this->getDeviceContext(),
                    config_ );
                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TInput, TOutput>>(base_op);
            }
        }
    };

    /**
     * @brief Type alias for CPU-based linear module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CpuLinear = Linear<DeviceType::Cpu, TInput, TOutput>;

    /**
     * @brief Type alias for CUDA-based linear module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CudaLinear = Linear<DeviceType::Cuda, TInput, TOutput>;
}