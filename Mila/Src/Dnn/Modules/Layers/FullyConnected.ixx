/**
 * @file FullyConnected.ixx
 * @brief Implementation of the Fully Connected (Linear) module for neural networks.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>

export module Dnn.Modules.FullyConnected;

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
    * @brief A class representing a fully-connected module.
    *
    * The fully-connected module (also known as a linear or dense layer) performs
    * a linear transformation of the input data. The operation is defined as:
    * output = input * weight + bias
    *
    * This is a fundamental building block in neural networks that connects
    * every input neuron to every output neuron with learnable weights.
    *
    * @tparam TPrecision The data type of the tensor elements (e.g., float, double).
    * @tparam TDeviceType The device type (CPU or CUDA) to run computations on.
    */
    export template<typename TPrecision, DeviceType TDeviceType = DeviceType::Cuda>
        requires ValidFloatTensorType<TPrecision>
    class FullyConnected : public Module<TPrecision, TPrecision, TDeviceType> {
    public:

        /**
         * @brief Memory resource type used for tensors, selected based on device type.
         *
         * Uses CudaMemoryResource for CUDA devices and HostMemoryResource for CPU.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;
		using ModuleBase = Module<TPrecision, TPrecision, TDeviceType>; ///< Base class type for the module

        /**
         * @brief Constructs a new FullyConnected module with the default device context.
         *
         * @param name The name of the module for identification purposes.
         * @param input_channels The number of input features/channels.
         * @param output_channels The number of output features/channels.
         * @param has_bias Whether to include a bias term in the transformation (defaults to true).
         * @param is_training Whether the module is initially in training mode (defaults to false).
         */
        FullyConnected(
            std::string name, std::string device_name, size_t input_channels, size_t output_channels,  bool has_bias = true,  bool is_training = false )
            : ModuleBase( device_name ), input_channels_{ input_channels }, output_channels_{ output_channels }, has_bias_{ has_bias } {
            this->setTraining( is_training );
            this->setName( name );
            initializeTensors();
            createOperation();
        }

        /**
         * @brief Constructs a new FullyConnected module with a specific device context.
         *
         * @param name The name of the module for identification purposes.
         * @param input_channels The number of input features/channels.
         * @param output_channels The number of output features/channels.
         * @param context The device context to use for this module.
         * @param has_bias Whether to include a bias term in the transformation (defaults to true).
         * @param is_training Whether the module is initially in training mode (defaults to false).
         */
        FullyConnected(
            std::string name, std::shared_ptr<DeviceContext> context, size_t input_channels, size_t output_channels,
            bool has_bias = true, bool is_training = false )
            : ModuleBase( context ), input_channels_{ input_channels }, output_channels_{ output_channels }, has_bias_{ has_bias } {
            this->setTraining( is_training );
            this->setName( name );
            initializeTensors();
            createOperation();
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
        * @brief Retrieves the weight tensor for this fully connected layer.
        *
        * The weight tensor has shape [output_channels, input_channels] and
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
         * The bias tensor has shape [output_channels] and is initialized to zeros
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
         * @brief Performs the forward pass of the FullyConnected operation.
         *
         * Applies the linear transformation to the input tensor:
         * output = input * weight + bias (if bias is enabled)
         *
         * @param input The input tensor to be transformed.
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TPrecision, MR>& input, Tensor<TPrecision, MR>& output ) {
            operation_->forward( input, parameters_, properties_, output, output_state_ );
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
         * - Input/output channels
         * - Device type
         * - Parameter information
         *
         * @return std::string A string representation of the module information.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "FullyConnected: " << this->getName();
            oss << ", Input channels: " << input_channels_;
            oss << ", Output channels: " << output_channels_;
            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->parametersToString();

            return oss.str();
        }

    private:
        /**
         * @brief The number of input features/channels.
         *
         * This is the dimension of the input tensor's last axis that will be
         * transformed by the fully connected layer.
         */
        size_t input_channels_{ 0 };

        /**
         * @brief The number of output features/channels.
         *
         * This is the dimension of the output tensor's last axis after
         * the linear transformation is applied.
         */
        size_t output_channels_{ 0 };

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
         * Shape is [output_channels_, input_channels_] to transform input features
         * to output features through matrix multiplication.
         */
        std::shared_ptr<Tensor<TPrecision, MR>> weight_{ nullptr };

        /**
         * @brief The bias tensor added after the matrix multiplication.
         *
         * Shape is [output_channels_]. This tensor is only used if has_bias_ is true.
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
         * @brief Cache of intermediate tensors needed for backward pass.
         *
         * Stores tensors that are computed during the forward pass and
         * are needed for gradient computation during backpropagation.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> output_state_;

        /**
         * @brief Additional configuration options for the fully connected operation.
         *
         * These attributes can modify the behavior of the underlying operation
         * implementation without changing the API.
         */
        OperationAttributes properties_;

        /**
         * @brief The underlying operation that implements the FullyConnected transformation.
         *
         * This operation performs the actual computation for the fully connected layer,
         * with different implementations for CPU and CUDA devices.
         */
        std::shared_ptr<UnaryOperation<TPrecision, TPrecision, TDeviceType>> operation_{ nullptr };

        /**
         * @brief Initializes the tensors needed for the FullyConnected operation.
         *
         * Creates and initializes:
         * - weight tensor (initialized with Xavier/Glorot distribution)
         * - bias tensor (initialized to zeros if has_bias_ is true)
         *
         * The tensors are created on the appropriate device (CPU or CUDA)
         * based on the current device context.
         */
        void initializeTensors() {
            parameters_.clear();
            this->parameter_map_.clear();

            weight_ = std::make_shared<Tensor<TPrecision, MR>>(
                std::vector<size_t>{output_channels_, input_channels_} );
            weight_->setName( this->getName() + ".weight" );

            xavier<TPrecision, MR>( *weight_, input_channels_, output_channels_ );

            if ( has_bias_ ) {
                bias_ = std::make_shared<Tensor<TPrecision, MR>>(
                    std::vector<size_t>{output_channels_} );
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
         * @brief Creates the appropriate FullyConnected operation based on the current device context.
         *
         * This method initializes the operation_ member with the appropriate implementation
         * of FullyConnected for either CPU or CUDA, based on the current device context.
         *
         * The operation is retrieved from the OperationRegistry which provides device-specific
         * implementations of the required operations.
         */
        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createOperation<TPrecision, TPrecision, DeviceType::Cpu>(
                    "Cpu::FullyConnectedOp",
                    this->getDeviceContext() );
                operation_ = std::static_pointer_cast<Dnn::Compute::UnaryOperation<TPrecision, TPrecision, DeviceType::Cpu>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createOperation<TPrecision, TPrecision, DeviceType::Cuda>(
                    "Cuda::FullyConnectedOp",
                    this->getDeviceContext() );
                operation_ = std::static_pointer_cast<Dnn::Compute::UnaryOperation<TPrecision, TPrecision, DeviceType::Cuda>>(base_op);
            }
        }
    };
}
