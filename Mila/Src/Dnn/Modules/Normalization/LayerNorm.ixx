/**
 * @file LayerNorm.ixx
 * @brief Implementation of Layer Normalization module for neural networks.
 */

module;
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>
#include <cstdint>
#include <stdexcept>

export module Dnn.Modules.LayerNorm;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
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
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
	using namespace Mila::Dnn::Serialization;

    /**
     * @brief Layer Normalization module.
     *
     * Layer Normalization is a technique used to normalize the inputs across features
     * for each data sample in a batch. It helps stabilize and accelerate deep neural
     * network training by reducing internal covariate shift.
     *
     * The operation can be expressed as:
     * y = ((x - mean) / sqrt(variance + epsilon)) * weight + bias
     *
     * Unlike Batch Normalization, Layer Normalization computes statistics independently
     * for each sample in a batch, making it well-suited for variable-length sequences
     * and recurrent neural networks.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidTensorTypes<TInput, TOutput>
    class LayerNorm : public Module<TDeviceType, TInput, TOutput> {
    public:
        /**
         * @brief Memory resource type used for tensors, selected based on device type.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;

        /**
         * @brief Alias for base module type.
         */
        using ModuleBase = Module<TDeviceType, TInput, TOutput>;

        /**
         * @brief Constructs a new LayerNorm module with a device name.
         *
         * Creates a new DeviceContext internally using the provided device name.
         * This constructor is useful for creating standalone modules without
         * pre-existing device contexts.
         *
         * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
         * @param config Configuration parameters for the LayerNorm module.
         * @throws std::invalid_argument If the device name is invalid or the configuration is invalid
         * @throws std::runtime_error If device type doesn't match template parameter TDeviceType
         */
        explicit LayerNorm( const std::string& device_name, const LayerNormConfig& config )
            : ModuleBase( std::make_shared<DeviceContext>( device_name ), config ), config_( config ) {

            config.validate();

            initializeTensors();
            createOperation();
        }

        /**
         * @brief Constructs a new LayerNorm module with a provided device context.
         *
         * Uses a pre-existing DeviceContext instance. This constructor is useful when integrating
         * the module into a larger network that shares device contexts across modules.
         *
         * @param device_context The device context to use for this module.
         * @param config Configuration parameters for the LayerNorm module.
         * @throws std::invalid_argument If device_context is null or configuration is invalid
         * @throws std::runtime_error If device context type doesn't match template parameter TDeviceType
         */
        explicit LayerNorm( std::shared_ptr<DeviceContext> device_context, const LayerNormConfig& config )
            : ModuleBase( device_context, config ), config_( config ) {

            config.validate();

            initializeTensors();
            createOperation();
        }

        /**
         * @brief Gets the weight tensor used for scaling after normalization.
         *
         * The weight tensor is applied as a scale factor to the normalized values.
         *
         * @return std::shared_ptr<Tensor<TInput, MR>> Shared pointer to the weight tensor.
         */
        std::shared_ptr<Tensor<TInput, MR>> getWeight() {
            return weight_;
        }

        /**
         * @brief Gets the bias tensor used after normalization and scaling.
         *
         * The bias tensor is added after normalization and scaling.
         *
         * @return std::shared_ptr<Tensor<TInput, MR>> Shared pointer to the bias tensor.
         */
        std::shared_ptr<Tensor<TInput, MR>> getBias() {
            return bias_;
        }

        /**
         * @brief Gets whether the module has a bias tensor.
         *
         * @return bool True if the module has a bias tensor, false otherwise.
         */
        bool hasBias() const {
            return config_.hasBias();
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
            size_t count = weight_->size();
            if ( config_.hasBias() ) {
                count += bias_->size();
            }
            return count;
        }

        /**
         * @brief Performs the forward pass of the Layer Normalization operation.
         *
         * Normalizes the input tensor across the specified axis, then scales and shifts
         * the result using the weight and bias tensors.
         *
         * @param input The input tensor to be normalized.
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TInput, MR>& input, Tensor<TOutput, MR>& output ) {
            operation_->forward( input, parameters_, properties_, output, output_state_ );
        }

        /**
         * @brief Performs the backward pass of the Layer Normalization operation.
         *
         * Computes gradients with respect to input and parameters.
         * Layer Normalization's backward pass requires computing:
         * 1. Gradient with respect to weight and bias (if present)
         * 2. Gradient with respect to input, which is more complex due to the
         *    normalization operation's chain rule derivatives
         *
         * @param input The input tensor from the forward pass.
         * @param output_grad The gradient of the loss with respect to the output.
         * @param input_grad The tensor to store the gradient with respect to input.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TOutput, MR>& output_grad,
            Tensor<TInput, MR>& input_grad ) {

            std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameter_grads;
            parameter_grads.resize( parameters_.size() );

            for ( size_t i = 0; i < parameters_.size(); ++i ) {
                parameter_grads[ i ] = std::make_shared<Tensor<TOutput, MR>>( parameters_[ i ]->getShape() );
            }

            operation_->backward(
                input,           // Input tensor
                output_grad,     // Gradient from next layer
                parameters_,     // Parameters (weight and bias)
                parameter_grads, // Parameter gradients
                input_grad,      // Gradient to propagate to previous layer
                properties_,     // Operation properties
                output_state_    // Cached tensors from forward pass
            );
        }

        /**
         * @brief Serializes the module state to a ZIP archive.
         *
         * Saves the trainable parameters (weight, bias) to the provided archive.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( ModelArchive& archive ) const override {
            // Save the state of the parameters
            for ( const auto& [name, tensor] : this->getParameterTensors() ) {
                // Save tensor data to zip archive
            }
        }

        /**
         * @brief Deserializes the module state from a ZIP archive.
         *
         * Loads the trainable parameters (weight, bias) from the provided archive.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( ModelArchive& archive ) override {
            for ( const auto& [name, tensor] : this->getParameterTensors() ) {
                // Load tensor data from zip archive
            }
        }

        /**
         * @brief Generates a string representation of this module's configuration.
         *
         * @return std::string A formatted string with module information
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "LayerNorm: " << this->getName() << std::endl;
            oss << "Normalization Axis: " << config_.getAxis() << std::endl;

            const auto& input_shape = config_.getInputShape();
            oss << "Input shape: (";
            for ( size_t i = 0; i < input_shape.size(); ++i ) {
                oss << input_shape[ i ];
                if ( i != input_shape.size() - 1 ) {
                    oss << ",";
                }
            }
            oss << ")" << std::endl;

            oss << "Epsilon: " << config_.getEpsilon() << std::endl;
            oss << "Has Bias: " << (config_.hasBias() ? "Yes" : "No") << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

    private:
        /**
         * @brief Configuration for the LayerNorm module.
         */
        LayerNormConfig config_;

        /**
         * @brief The weight tensor for scaling after normalization.
         */
        std::shared_ptr<Tensor<TOutput, MR>> weight_{ nullptr };

        /**
         * @brief The bias tensor added after normalization and scaling.
         */
        std::shared_ptr<Tensor<TOutput, MR>> bias_{ nullptr };

        /**
         * @brief The mean tensor used for normalization.
         *
         * Stores the mean values computed during the forward pass.
         */
        std::shared_ptr<Tensor<TOutput, MR>> mean_{ nullptr };

        /**
         * @brief The reciprocal standard deviation tensor.
         *
         * Stores the reciprocal of the standard deviation values (1/sqrt(variance + epsilon))
         * computed during the forward pass.
         */
        std::shared_ptr<Tensor<TOutput, MR>> rstd_{ nullptr };

        /**
         * @brief Collection of trainable parameters for this module.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameters_;

        /**
         * @brief Collection of output state tensors for caching.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> output_state_;

        /**
         * @brief Operation attributes and configuration.
         */
        OperationAttributes properties_;

        /**
         * @brief The underlying operation that implements Layer Normalization.
         */
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>> operation_{ nullptr };

        /**
         * @brief Initializes the tensors needed for the Layer Normalization operation.
         *
         * Creates and initializes:
         * - weight tensor (initialized to ones)
         * - bias tensor (initialized to zeros)
         * - mean tensor (for storing means during forward pass)
         * - reciprocal standard deviation tensor (for storing 1/std during forward pass)
         */
        void initializeTensors() {
            // Clear existing parameters and state
            parameters_.clear();
            output_state_.clear();
            this->parameter_map_.clear();
            this->state_map_.clear();

            const auto& input_shape = config_.getInputShape();
            if ( input_shape.empty() || input_shape.size() < 3 ) {
                throw std::invalid_argument( "Input shape must have at least 3 dimensions [batch_size, seq_len, features]" );
            }

            auto batch_size = input_shape[ 0 ];
            auto sequence_length = input_shape[ 1 ];
            auto channels = input_shape[ 2 ];

            weight_ = std::make_shared<Tensor<TOutput, MR>>(
                std::vector<size_t>{channels}, static_cast<TInput>(1.0f) );
            weight_->setName( this->getName() + ".weight" );

            if ( config_.hasBias() ) {
                bias_ = std::make_shared<Tensor<TOutput, MR>>(
                    std::vector<size_t>{channels} );
                bias_->setName( this->getName() + ".bias" );
            }

            mean_ = std::make_shared<Tensor<TOutput, MR>>(
                std::vector<size_t>{batch_size, sequence_length} );
            mean_->setName( this->getName() + ".mean" );

            rstd_ = std::make_shared<Tensor<TOutput, MR>>(
                std::vector<size_t>{batch_size, sequence_length} );
            rstd_->setName( this->getName() + ".rstd" );

            parameters_.emplace_back( weight_ );
            this->parameter_map_[ "weight" ] = weight_;

            if ( config_.hasBias() ) {
                parameters_.emplace_back( bias_ );
                this->parameter_map_[ "bias" ] = bias_;
            }

            // Add state tensors
            output_state_.emplace_back( mean_ );
            output_state_.emplace_back( rstd_ );

            this->state_map_[ "mean" ] = mean_;
            this->state_map_[ "rstd" ] = rstd_;

            // Set properties
            properties_.set( "epsilon", config_.getEpsilon() );
            properties_.set( "axis", config_.getAxis() );
        }

        /**
         * @brief Creates the appropriate Layer Normalization operation based on the current device context.
         *
         * This method initializes the operation_ member with the appropriate implementation
         * of Layer Normalization for either CPU or CUDA, based on the current device context.
         * It also passes the config object to the operation.
         */
        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TInput, TOutput>(
                    "Cpu::LayerNormOp",
                    this->getDeviceContext(),
                    config_ );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TInput, TOutput>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TInput, TOutput>(
                    "Cuda::LayerNormOp",
                    this->getDeviceContext(),
                    config_ );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TInput, TOutput>>(base_op);
            }
        }
    };

    /**
     * @brief Type alias for CPU-based layer normalization module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CpuLayerNorm = LayerNorm<DeviceType::Cpu, TInput, TOutput>;

    /**
     * @brief Type alias for CUDA-based layer normalization module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CudaLayerNorm = LayerNorm<DeviceType::Cuda, TInput, TOutput>;
}