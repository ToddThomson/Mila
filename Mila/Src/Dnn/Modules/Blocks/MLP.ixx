/**
 * @file MLP.ixx
 * @brief Implementation of Multi-Layer Perceptron (MLP) block for neural networks.
 */

module;
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <miniz.h>

export module Dnn.Blocks.MLP;
export import :Config;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.CompositeModule;
import Dnn.ActivationType;
import Compute.Precision;
import Compute.MemoryResource;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Compute.OperationRegistry;
import Dnn.Modules.Linear;
import Dnn.Modules.Gelu;
// FUTURE: import Dnn.Modules.ReLU;
// FUTURE: import Dnn.Modules.Swish;
import Dnn.Modules.Dropout;
import Dnn.Modules.LayerNorm;
import Dnn.Modules.Residual;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Multi-Layer Perceptron (MLP) block for neural networks.
     *
     * This module implements a two-layer MLP with an activation function in between:
     * input -> Linear -> Activation -> Linear -> output
     *
     * Optionally includes:
     * - Dropout after each layer
     * - Layer normalization
     * - Residual connection
     *
     * MLP blocks are fundamental components in many network architectures, including
     * transformers where they typically follow attention layers and process token
     * representations.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TDataType The data type used for tensor elements throughout the network.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TDataType = float>
        requires ValidFloatTensorType<TDataType>
    class MLP : public CompositeModule<TDeviceType, TDataType> {
    public:
        /**
         * @brief Memory resource type used for tensors, selected based on device type.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;

        /**
         * @brief Alias for base module type.
         */
        using CompositeModuleBase = CompositeModule<TDeviceType, TDataType>;

        /**
         * @brief Constructs a new MLP module with a device name.
         *
         * Creates a new DeviceContext internally using the provided device name.
         * This constructor is useful for creating standalone modules without
         * pre-existing device contexts.
         *
         * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
         * @param config Configuration parameters for the MLP module.
         * @throws std::invalid_argument If the device name is invalid or the configuration is invalid
         * @throws std::runtime_error If device type doesn't match template parameter TDeviceType
         */
        explicit MLP( const std::string& device_name, const MLPConfig& config )
            : CompositeModuleBase( std::make_shared<DeviceContext>( device_name ), config ), config_( config ) {

            config.validate();

            initializeModules();
        }

        /**
         * @brief Constructs a new MLP module with a provided device context.
         *
         * Uses a pre-existing DeviceContext instance. This constructor is useful when integrating
         * the module into a larger network that shares device contexts across modules.
         *
         * @param device_context The device context to use for this module.
         * @param config Configuration parameters for the MLP module.
         * @throws std::invalid_argument If device_context is null or configuration is invalid
         * @throws std::runtime_error If device context type doesn't match template parameter TDeviceType
         */
        explicit MLP( std::shared_ptr<DeviceContext> device_context, const MLPConfig& config )
            : CompositeModuleBase( device_context, config ), config_( config ) {

            config.validate();

            initializeModules();
        }

        /**
         * @brief Performs the forward pass of the MLP block.
         *
         * Processes the input through the full network:
         * Linear -> (LayerNorm) -> Activation -> (Dropout) -> Linear -> (Residual)
         *
         * When in inference mode with fused operations enabled, uses optimized execution.
         *
         * @param input The input tensor to be processed.
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TDataType, MR>& input, Tensor<TDataType, MR>& output ) {
            if ( this->isTraining() || !config_.useFusedOperations() ) {
                if ( config_.useResidual() ) {
                    input.copyTo( residual_input_ );
                }

                fc1_->forward( input, fc1_output_ );

                if ( config_.useLayerNorm() ) {
                    norm1_->forward( fc1_output_, norm1_output_ );
                    activation_->forward( norm1_output_, act_output_ );
                }
                else {
                    activation_->forward( fc1_output_, act_output_ );
                }

                if ( config_.getDropout() > 0.0f ) {
                    dropout1_->forward( act_output_, dropout1_output_ );
                    fc2_->forward( dropout1_output_, fc2_output_ );
                }
                else {
                    fc2_->forward( act_output_, fc2_output_ );
                }

                if ( config_.useResidual() ) {
                    // Add residual connection
                    for ( size_t i = 0; i < fc2_output_.size(); ++i ) {
                        output.data()[ i ] = fc2_output_.data()[ i ] + residual_input_.data()[ i ];
                    }
                }
                else {
                    fc2_output_.copyTo( output );
                }

                return;
            }

            // Inference mode with fused operations
            // Simplified implementation - would normally use dedicated fused operations
            fc1_->forward( input, fc1_output_ );

            if ( config_.useLayerNorm() ) {
                norm1_->forward( fc1_output_, norm1_output_ );
                activation_->forward( norm1_output_, act_output_ );
            }
            else {
                activation_->forward( fc1_output_, act_output_ );
            }

            fc2_->forward( act_output_, fc2_output_ );

            if ( config_.useResidual() ) {
                // Add residual connection
                for ( size_t i = 0; i < fc2_output_.size(); ++i ) {
                    output.data()[ i ] = fc2_output_.data()[ i ] + input.data()[ i ];
                }
            }
            else {
                fc2_output_.copyTo( output );
            }
        }

        /**
         * @brief Performs the backward pass of the MLP block.
         *
         * Computes gradients for all components in the network by working
         * backwards from the output gradient. Handles residual connections,
         * dropout, layer normalization, and activation functions.
         *
         * @param input The input tensor from the forward pass.
         * @param output_grad The gradient of loss with respect to the output.
         * @param input_grad The tensor to store gradients with respect to input.
         */
        void backward(
            const Tensor<TDataType, MR>& input,
            const Tensor<TDataType, MR>& output_grad,
            Tensor<TDataType, MR>& input_grad ) {

            if ( config_.useResidual() ) {
                // Copy output gradients to input_grad for the residual connection
                output_grad.copyTo( input_grad );

                // Compute gradients for fc2
                Tensor<TDataType, MR> fc2_grad( fc2_output_.getShape() );
                fc2_->backward(
                    config_.getDropout() > 0.0f ? dropout1_output_ : act_output_,
                    output_grad,
                    fc2_grad );

                // Compute gradients for dropout1 if needed
                if ( config_.getDropout() > 0.0f ) {
                    Tensor<TDataType, MR> dropout1_grad( act_output_.getShape() );
                    dropout1_->backward( act_output_, fc2_grad, dropout1_grad );

                    // Compute gradients for activation
                    Tensor<TDataType, MR> act_grad( config_.useLayerNorm() ? norm1_output_.getShape() : fc1_output_.getShape() );
                    activation_->backward(
                        config_.useLayerNorm() ? norm1_output_ : fc1_output_,
                        dropout1_grad,
                        act_grad );

                    // Process layer norm if used
                    if ( config_.useLayerNorm() ) {
                        Tensor<TDataType, MR> norm1_grad( fc1_output_.getShape() );
                        norm1_->backward( fc1_output_, act_grad, norm1_grad );

                        // Compute gradients for fc1
                        Tensor<TDataType, MR> fc1_grad( input.getShape() );
                        fc1_->backward( input, norm1_grad, fc1_grad );

                        // Add fc1 gradients to input_grad (already containing residual gradients)
                        for ( size_t i = 0; i < input_grad.size(); ++i ) {
                            input_grad.data()[ i ] += fc1_grad.data()[ i ];
                        }
                    }
                    else {
                        // Compute gradients for fc1 directly
                        Tensor<TDataType, MR> fc1_grad( input.getShape() );
                        fc1_->backward( input, act_grad, fc1_grad );

                        // Add fc1 gradients to input_grad (already containing residual gradients)
                        for ( size_t i = 0; i < input_grad.size(); ++i ) {
                            input_grad.data()[ i ] += fc1_grad.data()[ i ];
                        }
                    }
                }
                else {
                    // Similar flow but without dropout
                    Tensor<TDataType, MR> act_grad( config_.useLayerNorm() ? norm1_output_.getShape() : fc1_output_.getShape() );
                    activation_->backward(
                        config_.useLayerNorm() ? norm1_output_ : fc1_output_,
                        fc2_grad,
                        act_grad );

                    if ( config_.useLayerNorm() ) {
                        Tensor<TDataType, MR> norm1_grad( fc1_output_.getShape() );
                        norm1_->backward( fc1_output_, act_grad, norm1_grad );

                        Tensor<TDataType, MR> fc1_grad( input.getShape() );
                        fc1_->backward( input, norm1_grad, fc1_grad );

                        for ( size_t i = 0; i < input_grad.size(); ++i ) {
                            input_grad.data()[ i ] += fc1_grad.data()[ i ];
                        }
                    }
                    else {
                        Tensor<TDataType, MR> fc1_grad( input.getShape() );
                        fc1_->backward( input, act_grad, fc1_grad );

                        for ( size_t i = 0; i < input_grad.size(); ++i ) {
                            input_grad.data()[ i ] += fc1_grad.data()[ i ];
                        }
                    }
                }
            }
            else {
                // No residual connection - standard backward pass
                fc2_->backward(
                    config_.getDropout() > 0.0f ? dropout1_output_ : act_output_,
                    output_grad,
                    input_grad );

                // Remaining backward pass logic follows same pattern as above
                // but without adding to residual gradients
                if ( config_.getDropout() > 0.0f ) {
                    Tensor<TDataType, MR> dropout1_grad( act_output_.getShape() );
                    dropout1_->backward( act_output_, input_grad, dropout1_grad );

                    Tensor<TDataType, MR> act_grad( config_.useLayerNorm() ? norm1_output_.getShape() : fc1_output_.getShape() );
                    activation_->backward(
                        config_.useLayerNorm() ? norm1_output_ : fc1_output_,
                        dropout1_grad,
                        act_grad );

                    if ( config_.useLayerNorm() ) {
                        Tensor<TDataType, MR> norm1_grad( fc1_output_.getShape() );
                        norm1_->backward( fc1_output_, act_grad, norm1_grad );

                        fc1_->backward( input, norm1_grad, input_grad );
                    }
                    else {
                        fc1_->backward( input, act_grad, input_grad );
                    }
                }
                else {
                    Tensor<TDataType, MR> act_grad( config_.useLayerNorm() ? norm1_output_.getShape() : fc1_output_.getShape() );
                    activation_->backward(
                        config_.useLayerNorm() ? norm1_output_ : fc1_output_,
                        input_grad,
                        act_grad );

                    if ( config_.useLayerNorm() ) {
                        Tensor<TDataType, MR> norm1_grad( fc1_output_.getShape() );
                        norm1_->backward( fc1_output_, act_grad, norm1_grad );

                        fc1_->backward( input, norm1_grad, input_grad );
                    }
                    else {
                        fc1_->backward( input, act_grad, input_grad );
                    }
                }
            }
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * Counts the total number of parameters across all submodules.
         *
         * @return size_t The total number of parameters.
         */
        size_t parameterCount() const override {
            size_t total_parameters = 0;
            for ( const auto& module : this->getModules() ) {
                total_parameters += module->parameterCount();
            }
            return total_parameters;
        }

        /**
         * @brief Serializes the module state to a ZIP archive.
         *
         * Saves the state of all submodules to the provided ZIP archive.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( mz_zip_archive& zip ) const override {
            for ( const auto& module : this->getModules() ) {
                module->save( zip );
            }
        }

        /**
         * @brief Deserializes the module state from a ZIP archive.
         *
         * Loads the state of all submodules from the provided ZIP archive.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( mz_zip_archive& zip ) override {
            for ( const auto& module : this->getModules() ) {
                module->load( zip );
            }
        }

        /**
         * @brief Generates a string representation of this module's configuration.
         *
         * @return std::string A formatted string with module information
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "====================" << std::endl;
            oss << "MLP: " << this->getName() << std::endl;

            const auto& input_shape = config_.getInputShape();
            oss << "Input shape: (";
            for ( size_t i = 0; i < input_shape.size(); ++i ) {
                oss << input_shape[ i ];
                if ( i != input_shape.size() - 1 ) {
                    oss << ",";
                }
            }
            oss << ")" << std::endl;

            oss << "Input features: " << config_.getInputFeatures() << std::endl;
            oss << "Hidden size: " << config_.getHiddenSize() << std::endl;
            oss << "Bias: " << (config_.hasBias() ? "enabled" : "disabled") << std::endl;
            oss << "Activation: " << activationTypeToString( config_.getActivationType() ) << std::endl;

            if ( config_.getDropout() > 0.0f ) {
                oss << "Dropout: " << config_.getDropout() << std::endl;
            }

            if ( config_.useLayerNorm() ) {
                oss << "Layer Norm: enabled" << std::endl;
            }

            if ( config_.useResidual() ) {
                oss << "Residual: enabled" << std::endl;
            }

            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;
            oss << "Sub-Modules..." << std::endl;

            for ( const auto& [name, module] : this->getNamedModules() ) {
                oss << module->toString();
            }

            return oss.str();
        }

    private:
        /**
         * @brief Configuration for the MLP module.
         */
        MLPConfig config_;

        /**
         * @brief First linear layer (input_features -> hidden_size).
         */
        std::shared_ptr<Linear<TDeviceType, TDataType>> fc1_{ nullptr };

        /**
         * @brief Activation function module.
         */
        std::shared_ptr<Module<TDeviceType, TDataType>> activation_{ nullptr };

        /**
         * @brief Second linear layer (hidden_size -> input_features).
         */
        std::shared_ptr<Linear<TDeviceType, TDataType>> fc2_{ nullptr };

        /**
         * @brief Optional layer normalization module.
         */
        std::shared_ptr<LayerNorm<TDeviceType, TDataType>> norm1_{ nullptr };

        /**
         * @brief Optional dropout module.
         */
        std::shared_ptr<Dropout<TDeviceType, TDataType>> dropout1_{ nullptr };

        /**
         * @brief Output tensor from first linear layer.
         */
        Tensor<TDataType, MR> fc1_output_;

        /**
         * @brief Output tensor from layer normalization.
         */
        Tensor<TDataType, MR> norm1_output_;

        /**
         * @brief Output tensor from activation function.
         */
        Tensor<TDataType, MR> act_output_;

        /**
         * @brief Output tensor from dropout.
         */
        Tensor<TDataType, MR> dropout1_output_;

        /**
         * @brief Output tensor from second linear layer.
         */
        Tensor<TDataType, MR> fc2_output_;

        /**
         * @brief Cached input tensor for residual connection.
         */
        Tensor<TDataType, MR> residual_input_;

        /**
         * @brief Initializes all submodules for the MLP.
         *
         * Creates and configures:
         * 1. Two linear layers with configurable hidden size
         * 2. Activation function (GELU or others in the future)
         * 3. Optional layer normalization
         * 4. Optional dropout
         * Also prepares intermediate tensors for computation.
         */
        void initializeModules() {
            // Clear any existing modules
            for ( const auto& [name, _] : this->getNamedModules() ) {
                this->removeModule( name );
            }

            // First linear layer: input_features -> hidden_size
            auto fc1_config = LinearConfig( config_.getInputFeatures(), config_.getHiddenSize() )
                .withName( this->getName() + ".fc1" )
                .withBias( config_.hasBias() )
                .withTraining( this->isTraining() );

            fc1_ = std::make_shared<Linear<TDeviceType, TDataType>>( this->getDeviceContext(), fc1_config );
            this->addModule( "fc1", fc1_ );

            // Optional layer normalization
            if ( config_.useLayerNorm() ) {
                auto norm1_config = LayerNormConfig( config_.getHiddenSize() )
                    .withName( this->getName() + ".norm1" )
                    .withTraining( this->isTraining() );

                norm1_ = std::make_shared<LayerNorm<TDeviceType, TDataType>>( this->getDeviceContext(), norm1_config );
                this->addModule( "norm1", norm1_ );
            }

            // Activation function based on configuration
            switch ( config_.getActivationType() ) {
                case ActivationType::Gelu:
                {
                    auto gelu_config = GeluConfig()
                        .withName( this->getName() + ".gelu" )
                        .withTraining( this->isTraining() );

                    activation_ = std::make_shared<Gelu<TDeviceType, TDataType>>( this->getDeviceContext(), gelu_config );
                    break;
                }
                // FUTURE: Add more activation types here
            }

            this->addModule( "activation", activation_ );

            // Optional dropout
            if ( config_.getDropout() > 0.0f ) {
                auto dropout_config = DropoutConfig( config_.getDropout() )
                    .withName( this->getName() + ".dropout1" )
                    .withTraining( this->isTraining() );

                dropout1_ = std::make_shared<Dropout<TDeviceType, TDataType>>( this->getDeviceContext(), dropout_config );
                this->addModule( "dropout1", dropout1_ );
            }

            // Second linear layer: hidden_size -> input_features
            auto fc2_config = LinearConfig( config_.getHiddenSize(), config_.getInputFeatures() )
                .withName( this->getName() + ".fc2" )
                .withBias( config_.hasBias() )
                .withTraining( this->isTraining() );

            fc2_ = std::make_shared<Linear<TDeviceType, TDataType>>( this->getDeviceContext(), fc2_config );
            this->addModule( "fc2", fc2_ );

            // Create intermediate tensors
            const auto& input_shape = config_.getInputShape();
            std::vector<size_t> hidden_shape = input_shape;
            if ( !hidden_shape.empty() ) {
                hidden_shape.back() = config_.getHiddenSize();
            }
            else {
                hidden_shape = { config_.getHiddenSize() };
            }

            fc1_output_ = Tensor<TDataType, MR>( hidden_shape );

            if ( config_.useLayerNorm() ) {
                norm1_output_ = Tensor<TDataType, MR>( hidden_shape );
            }

            act_output_ = Tensor<TDataType, MR>( hidden_shape );

            if ( config_.getDropout() > 0.0f ) {
                dropout1_output_ = Tensor<TDataType, MR>( hidden_shape );
            }

            fc2_output_ = Tensor<TDataType, MR>( input_shape );

            if ( config_.useResidual() ) {
                residual_input_ = Tensor<TDataType, MR>( input_shape );
            }
        }
    };

    /**
     * @brief Type alias for CPU-based MLP module with customizable tensor type.
     *
     * @tparam TDataType Data type of the tensor elements.
     */
    export template<typename TDataType = float>
        using CpuMLP = MLP<DeviceType::Cpu, TDataType>;

    /**
     * @brief Type alias for CUDA-based MLP module with customizable tensor type.
     *
     * @tparam TDataType Data type of the tensor elements.
     */
    export template<typename TDataType = float>
        using CudaMLP = MLP<DeviceType::Cuda, TDataType>;
}