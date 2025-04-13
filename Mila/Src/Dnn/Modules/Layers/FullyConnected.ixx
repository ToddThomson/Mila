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

export namespace Mila::Dnn
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
    * @tparam TInput The data type of the input tensor elements.
    * @tparam TPrecision The data type used for computation and output (defaults to the input type).
    */
    export
        template<typename TInput, typename TPrecision = TInput>
        requires ValidTensorTypes<TInput, TPrecision>
    class FullyConnected : public Module<TInput, TPrecision> {
    public:
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
            std::string name,
            size_t input_channels,
            size_t output_channels,
            bool has_bias = true,
            bool is_training = false )
            : Module<TInput, TPrecision>(),
            input_channels_{ input_channels },
            output_channels_{ output_channels },
            has_bias_{ has_bias } {
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
            std::string name,
            size_t input_channels,
            size_t output_channels,
            std::shared_ptr<DeviceContext> context,
            bool has_bias = true,
            bool is_training = false )
            : Module<TInput, TPrecision>( context ),
            input_channels_{ input_channels },
            output_channels_{ output_channels },
            has_bias_{ has_bias } {
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
         * @brief Gets the weight tensor used for the linear transformation.
         *
         * The weight tensor is used in the matrix multiplication operation:
         * output = input * weight + bias
         *
         * @return std::shared_ptr<Tensor<TPrecision>> Shared pointer to the weight tensor.
         */
        template<typename TMR>
        std::shared_ptr<Tensor<TPrecision, TMR>> getWeight() {
            return std::static_pointer_cast<Tensor<TPrecision, TMR>>(weight_);
        }

        /**
         * @brief Gets the bias tensor used after the linear transformation.
         *
         * The bias tensor is added after the matrix multiplication with weights.
         * If the module was configured without bias, returns std::nullopt.
         *
         * @return std::optional<std::shared_ptr<Tensor<TPrecision>>> Optional containing
         *         the bias tensor if available, otherwise std::nullopt.
         */
        template<typename TMR>
        std::optional<std::shared_ptr<Tensor<TPrecision, TMR>>> getBias() {
            if ( !has_bias_ ) {
                return std::nullopt;
            }
            return std::static_pointer_cast<Tensor<TPrecision, TMR>>(bias_);
        }

        /**
         * @brief Gets whether the module has a bias tensor.
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
         * output = input * weight + bias
         *
         * @param input The input tensor to be transformed.
         * @param output The output tensor where the results will be stored.
         */
        template<typename TMR>
        void forward( const Tensor<TInput, TMR>& input, Tensor<TPrecision, TMR>& output ) {
            operation_->forward( input, parameters_, properties_, output, output_state_ );
        }

        /**
         * @brief Saves the module state to a ZIP archive.
         *
         * Serializes the trainable parameters (weight, bias) to the provided archive.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( mz_zip_archive& zip ) const override {
            // Save the state of the parameters
            for ( const auto& [name, tensor] : this->getParameterTensors() ) {
                // Save tensor data to zip archive
                // Implementation will depend on how tensors are serialized
            }
        }

        /**
         * @brief Loads the module state from a ZIP archive.
         *
         * Deserializes the trainable parameters (weight, bias) from the provided archive.
         *
         * @param zip The ZIP archive to load the module state from.
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
         * Includes detailed information about the module configuration and parameters.
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

    protected:
        /**
         * @brief Called when the device context changes.
         *
         * Recreates tensors and operations for the new device.
         */
        void onDeviceChanged() override {
            // Recreate tensors and operations for the new device
            initializeTensors();
            createOperation();
        }

    private:
        /**
         * @brief The number of input features/channels.
         */
        size_t input_channels_{ 0 };

        /**
         * @brief The number of output features/channels.
         */
        size_t output_channels_{ 0 };

        /**
         * @brief Whether the module has a bias tensor. Default is true.
         */
        bool has_bias_{ true };

        /**
         * @brief The weight tensor for the linear transformation.
         *
         * Shape is [output_channels_, input_channels_].
         */
        std::shared_ptr<Tensor<TPrecision, typename Module<TInput, TPrecision>::MR>> weight_{ nullptr };

        /**
         * @brief The bias tensor added after the matrix multiplication.
         *
         * Shape is [output_channels_].
         */
        std::shared_ptr<Tensor<TPrecision, typename Module<TInput, TPrecision>::MR>> bias_{ nullptr };

        /**
         * @brief The trainable parameters for this module (weight and bias).
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, typename Module<TInput, TPrecision>::MR>>> parameters_;

        /**
         * @brief Cache of intermediate tensors needed for backward pass.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, typename Module<TInput, TPrecision>::MR>>> output_state_;

        /**
         * @brief The operation attributes, may include additional configuration.
         */
        OperationAttributes properties_;

        /**
         * @brief The underlying operation that implements the FullyConnected transformation.
         */
        std::shared_ptr<UnaryOperation<TInput, TPrecision>> operation_{ nullptr };

        /**
         * @brief Initializes the tensors needed for the FullyConnected operation.
         *
         * Creates and initializes:
         * - weight tensor (initialized with Xavier/Glorot distribution)
         * - bias tensor (initialized to zeros if has_bias_ is true)
         */
        void initializeTensors() {
            // Clear existing parameters
            parameters_.clear();
            this->parameter_map_.clear();

            // Get device type for proper tensor creation
            auto device_type = this->getDeviceContext()->getDevice()->getDeviceType();

            // Initialize the weight tensor using xavier distribution
            if ( device_type == DeviceType::Cpu ) {
                weight_ = std::make_shared<Tensor<TPrecision, Compute::HostMemoryResource>>(
                    std::vector<size_t>{output_channels_, input_channels_} );
                weight_->setName( this->getName() + ".weight" );
                xavier<TPrecision, Compute::HostMemoryResource>( *std::static_pointer_cast<Tensor<TPrecision, Compute::HostMemoryResource>>(weight_),
                    input_channels_, output_channels_ );

                if ( has_bias_ ) {
                    // Initialize the bias tensor with zeros
                    bias_ = std::make_shared<Tensor<TPrecision, Compute::HostMemoryResource>>(
                        std::vector<size_t>{output_channels_} );
                    bias_->setName( this->getName() + ".bias" );
                }
            }
            else {
                weight_ = std::make_shared<Tensor<TPrecision, Compute::DeviceMemoryResource>>(
                    std::vector<size_t>{output_channels_, input_channels_} );
                weight_->setName( this->getName() + ".weight" );
                xavier<TPrecision, Compute::DeviceMemoryResource>( *std::static_pointer_cast<Tensor<TPrecision, Compute::DeviceMemoryResource>>(weight_),
                    input_channels_, output_channels_ );

                if ( has_bias_ ) {
                    // Initialize the bias tensor with zeros
                    bias_ = std::make_shared<Tensor<TPrecision, Compute::DeviceMemoryResource>>(
                        std::vector<size_t>{output_channels_} );
                    bias_->setName( this->getName() + ".bias" );
                }
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
         */
        void createOperation() {
            // Get the device type from the context
            auto device_type = this->getDeviceContext()->getDevice()->getDeviceType();

            if ( operation_ ) {
                operation_.reset(); // Clear existing operation
            }

            if ( device_type == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cpu>( "Cpu::FullyConnectedOp" );
                operation_ = std::static_pointer_cast<UnaryOperation<TInput, TPrecision>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cuda>( "Cuda::FullyConnectedOp" );
                operation_ = std::static_pointer_cast<UnaryOperation<TInput, TPrecision>>(base_op);
            }
        }
    };
}
