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

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;

import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuDevice;
import Compute.CudaMemoryResource;
import Compute.CpuMemoryResource;

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
    * @tparam TPrecision The data type used for internal calculations, defaults to TOutput.
    */
    export template<DeviceType TDeviceType = DeviceType::Cuda,
        typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
        requires ValidFloatTensorTypes<TInput, TOutput> && ValidPrecisionType<TPrecision>
    class Linear : public Module<TDeviceType, TInput, TOutput, TPrecision> {
    public:
        /**
         * @brief Memory resource type used for tensors, selected based on device type.
         *
         * Uses CudaMemoryResource for CUDA devices and CpuMemoryResource for CPU.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;
        using ModuleBase = Module<TDeviceType, TInput, TOutput, TPrecision>; ///< Base class type for the module

        /**
         * @brief Constructs a new Linear module with the default device context.
         *
         * @param name The name of the module for identification purposes.
         * @param device_name The name of the device to use for this module.
         * @param input_features The number of input features.
         * @param output_features The number of output features.
         * @param has_bias Whether to include a bias term in the transformation (defaults to true).
         * @param is_training Whether the module is initially in training mode (defaults to false).
         */
        Linear(
            std::string name, std::string device_name, size_t input_features, size_t output_features,
            bool has_bias = true, bool is_training = false )
            : ModuleBase( device_name ), input_features_{ input_features }, output_features_{ output_features }, has_bias_{ has_bias } {
            this->setTraining( is_training );
            this->setName( name );
            initializeParameters();
            createOperation();
        }

        /**
         * @brief Constructs a new Linear module with a specific device context.
         *
         * @param name The name of the module for identification purposes.
         * @param context The device context to use for this module.
         * @param input_features The number of input features.
         * @param output_features The number of output features.
         * @param has_bias Whether to include a bias term in the transformation (defaults to true).
         * @param is_training Whether the module is initially in training mode (defaults to false).
         */
        Linear(
            std::string name, std::shared_ptr<DeviceContext> context, size_t input_features, size_t output_features,
            bool has_bias = true, bool is_training = false )
            : ModuleBase( context ), input_features_{ input_features }, output_features_{ output_features }, has_bias_{ has_bias } {
            this->setTraining( is_training );
            this->setName( name );

            initializeParameters();
            initializeParameterGradients();

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
            const Tensor<TPrecision, MR>& input,
            const Tensor<TPrecision, MR>& output_grad,
            Tensor<TPrecision, MR>& input_grad ) {
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
            if ( has_bias_ ) {
                num_params += bias_->size();
            }
            return num_params;
        }

        /**
        * @brief Retrieves the weight tensor for this linear layer.
        *
        * The weight tensor has shape [output_features, input_features] and
        * is initialized with Xavier/Glorot distribution.
        *
        * @return The weight tensor used in the linear transformation.
        */
        std::shared_ptr<Tensor<TPrecision, MR>> getWeight() {
            return weight_;
        }

        /**
         * @brief Retrieves the bias tensor if present.
         *
         * The bias tensor has shape [output_features] and is initialized to zeros
         * if bias is enabled in the layer configuration.
         *
         * @return An optional containing the bias tensor if bias is enabled, otherwise std::nullopt.
         */
        std::optional<std::shared_ptr<Tensor<TPrecision, MR>>> getBias() {
            return has_bias_ ? std::optional{ bias_ } : std::nullopt;
        }

        /**
         * @brief Checks whether the module has a bias tensor.
         *
         * @return bool True if the module has a bias tensor, false otherwise.
         */
        bool hasBias() const {
            return has_bias_;
        }

        /**
         * @brief Serializes the module state to a ZIP archive.
         *
         * Saves the trainable parameters (weight, bias) to the provided archive.
         * Note: This method is currently a placeholder and needs implementation.
         *
         * @param zip The ZIP archive to save the module state to.
         * @todo Implement proper serialization functionality
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
         * @todo Implement proper deserialization functionality
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
         * - Parameter information
         *
         * @return std::string A string representation of the module information.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Linear: " << this->getName();
            oss << ", Input features: " << input_features_;
            oss << ", Output features: " << output_features_;
            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->parametersToString();

            return oss.str();
        }

    private:
        /**
         * @brief The number of input features.
         *
         * This is the dimension of the input tensor's last axis that will be
         * transformed by the linear layer.
         */
        size_t input_features_{ 0 };

        /**
         * @brief The number of output features.
         *
         * This is the dimension of the output tensor's last axis after
         * the linear transformation is applied.
         */
        size_t output_features_{ 0 };

        /**
         * @brief Whether the module has a bias tensor.
         *
         * When true, the bias vector is added after the matrix multiplication.
         * When false, only the matrix multiplication is performed.
         */
        bool has_bias_{ true };

        /**
         * @brief The weight tensor for the linear transformation.
         *
         * Shape is [output_features_, input_features_] to transform input features
         * to output features through matrix multiplication.
         */
        std::shared_ptr<Tensor<TPrecision, MR>> weight_{ nullptr };

        /**
         * @brief The bias tensor added after the matrix multiplication.
         *
         * Shape is [output_features_]. This tensor is only used if has_bias_ is true.
         */
        std::shared_ptr<Tensor<TPrecision, MR>> bias_{ nullptr };

        /**
         * @brief Collection of trainable parameters for this module.
         *
         * Contains the weight tensor and optionally the bias tensor if has_bias_ is true.
         * These parameters are passed to the underlying operation during forward pass.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> parameters_;

        /**
         * @brief Gradients for the parameters of this module.
         *
         * Contains gradients for the weight tensor and optionally the bias tensor.
         * These are computed during the backward pass.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> parameter_grads_;

        /**
         * @brief Cache of intermediate tensors needed for backward pass.
         *
         * Stores tensors that are computed during the forward pass and
         * are needed for gradient computation during backpropagation.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> output_state_;

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
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput, TPrecision>> operation_{ nullptr };

        /**
         * @brief Initializes the tensors needed for the Linear operation.
         *
         * Creates and initializes:
         * - weight tensor (initialized with Xavier/Glorot distribution)
         * - bias tensor (initialized to zeros if has_bias_ is true)
         *
         * The tensors are created on the appropriate device (CPU or CUDA)
         * based on the current device context.
         */
        void initializeParameters() {
            parameters_.clear();
            this->parameter_map_.clear();

            weight_ = std::make_shared<Tensor<TPrecision, MR>>(
                std::vector<size_t>{output_features_, input_features_} );
            weight_->setName( this->getName() + ".weight" );

            xavier<TPrecision, MR>( *weight_, input_features_, output_features_ );

            if ( has_bias_ ) {
                bias_ = std::make_shared<Tensor<TPrecision, MR>>(
                    std::vector<size_t>{output_features_} );
                bias_->setName( this->getName() + ".bias" );
            }

            // Add tensors to parameters list and map
            parameters_.emplace_back( weight_ );
            this->parameter_map_[ "weight" ] = weight_;

            if ( has_bias_ ) {
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
            auto weight_grad = std::make_shared<Tensor<TPrecision, MR>>( std::vector<size_t>{output_features_, input_features_} );
            weight_grad->setName( this->getName() + ".weight_grad" );
            parameter_grads_.push_back( weight_grad );

            if ( has_bias_ ) {
                auto bias_grad = std::make_shared<Tensor<TPrecision, MR>>( std::vector<size_t>{output_features_} );
                bias_grad->setName( this->getName() + ".bias_grad" );
                parameter_grads_.emplace_back( bias_grad );
            }
        }

        /**
         * @brief Creates the appropriate Linear operation based on the current device context.
         *
         * This method initializes the operation_ member with the appropriate implementation
         * of Linear for either CPU or CUDA, based on the current device context.
         *
         * The operation is retrieved from the OperationRegistry which provides device-specific
         * implementations of the required operations.
         */
        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TInput, TOutput, TPrecision>(
                    "Cpu::FullyConnectedOp",
                    this->getDeviceContext() );
                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TInput, TOutput, TPrecision>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TInput, TOutput, TPrecision>(
                    "Cuda::FullyConnectedOp",
                    this->getDeviceContext() );
                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TInput, TOutput, TPrecision>>(base_op);
            }
        }
    };

    /**
     * @brief Type alias for CPU-based linear module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     * @tparam TPrecision Data type used for internal calculations, defaults to TOutput.
     */
    export template<typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
        using CpuLinear = Linear<DeviceType::Cpu, TInput, TOutput, TPrecision>;

    /**
     * @brief Type alias for CUDA-based linear module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     * @tparam TPrecision Data type used for internal calculations, defaults to TOutput.
     */
    export template<typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
        using CudaLinear = Linear<DeviceType::Cuda, TInput, TOutput, TPrecision>;
}