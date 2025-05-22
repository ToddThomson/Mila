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
import Compute.CpuDevice;
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
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TDataType The data type used for tensor elements throughout the network.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TDataType = float>
        requires ValidFloatTensorType<TDataType>
    class MLP : public CompositeModule<TDeviceType, TDataType> {
    public:
        using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::CudaMemoryResource, Compute::CpuMemoryResource>;
        using CompositeModuleBase = CompositeModule<TDeviceType, TDataType>;

        /**
         * @brief Construct a new MLP module from configuration.
         *
         * @param config The configuration for this module
         */
        explicit MLP( const MLPConfig& config )
            : CompositeModuleBase(
                config.getContext() ? config.getContext() : std::make_shared<DeviceContext>( config.getDeviceName() ),
                TDeviceType == DeviceType::Cpu ? ComputePrecision::Policy::Disabled : config.getPrecision() ),
            input_shape_( config.getInputShape() ),
            input_features_( config.getInputFeatures() ),
            hidden_size_( config.getHiddenSize() ),
            has_bias_( config.hasBias() ),
            activation_type_( config.getActivationType() ),
            dropout_prob_( config.getDropout() ),
            use_layer_norm_( config.useLayerNorm() ),
            use_residual_( config.useResidual() ),
            fuse_operations_( config.useFusedOperations() ) {

            config.validate();

            this->setName( config.getName() );
            this->setTraining( config.isTraining() );

            initializeModules();
        }


        /**
         * @brief Performs the forward pass of the MLP block.
         */
        void forward( const Tensor<TDataType, MR>& input, Tensor<TDataType, MR>& output ) {
            if ( this->isTraining() || !fuse_operations_ ) {
                if ( use_residual_ ) {
                    input.copyTo( residual_input_ );
                }

                fc1_->forward( input, fc1_output_ );

                if ( use_layer_norm_ ) {
                    norm1_->forward( fc1_output_, norm1_output_ );
                    activation_->forward( norm1_output_, act_output_ );
                }
                else {
                    activation_->forward( fc1_output_, act_output_ );
                }

                if ( dropout_prob_ > 0.0f ) {
                    dropout1_->forward( act_output_, dropout1_output_ );
                    fc2_->forward( dropout1_output_, fc2_output_ );
                }
                else {
                    fc2_->forward( act_output_, fc2_output_ );
                }

                if ( use_residual_ ) {
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

            if ( use_layer_norm_ ) {
                norm1_->forward( fc1_output_, norm1_output_ );
                activation_->forward( norm1_output_, act_output_ );
            }
            else {
                activation_->forward( fc1_output_, act_output_ );
            }

            fc2_->forward( act_output_, fc2_output_ );

            if ( use_residual_ ) {
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
         */
        void backward(
            const Tensor<TDataType, MR>& input,
            const Tensor<TDataType, MR>& output_grad,
            Tensor<TDataType, MR>& input_grad ) {

            if ( use_residual_ ) {
                // Copy output gradients to input_grad for the residual connection
                output_grad.copyTo( input_grad );

                // Compute gradients for fc2
                Tensor<TDataType, MR> fc2_grad( fc2_output_.getShape() );
                fc2_->backward(
                    dropout_prob_ > 0.0f ? dropout1_output_ : act_output_,
                    output_grad,
                    fc2_grad );

                // Compute gradients for dropout1 if needed
                if ( dropout_prob_ > 0.0f ) {
                    Tensor<TDataType, MR> dropout1_grad( act_output_.getShape() );
                    dropout1_->backward( act_output_, fc2_grad, dropout1_grad );

                    // Compute gradients for activation
                    Tensor<TDataType, MR> act_grad( use_layer_norm_ ? norm1_output_.getShape() : fc1_output_.getShape() );
                    activation_->backward(
                        use_layer_norm_ ? norm1_output_ : fc1_output_,
                        dropout1_grad,
                        act_grad );

                    // Process layer norm if used
                    if ( use_layer_norm_ ) {
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
                    Tensor<TDataType, MR> act_grad( use_layer_norm_ ? norm1_output_.getShape() : fc1_output_.getShape() );
                    activation_->backward(
                        use_layer_norm_ ? norm1_output_ : fc1_output_,
                        fc2_grad,
                        act_grad );

                    if ( use_layer_norm_ ) {
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
                    dropout_prob_ > 0.0f ? dropout1_output_ : act_output_,
                    output_grad,
                    input_grad );

                // Remaining backward pass logic follows same pattern as above
                // but without adding to residual gradients
                if ( dropout_prob_ > 0.0f ) {
                    Tensor<TDataType, MR> dropout1_grad( act_output_.getShape() );
                    dropout1_->backward( act_output_, input_grad, dropout1_grad );

                    Tensor<TDataType, MR> act_grad( use_layer_norm_ ? norm1_output_.getShape() : fc1_output_.getShape() );
                    activation_->backward(
                        use_layer_norm_ ? norm1_output_ : fc1_output_,
                        dropout1_grad,
                        act_grad );

                    if ( use_layer_norm_ ) {
                        Tensor<TDataType, MR> norm1_grad( fc1_output_.getShape() );
                        norm1_->backward( fc1_output_, act_grad, norm1_grad );

                        fc1_->backward( input, norm1_grad, input_grad );
                    }
                    else {
                        fc1_->backward( input, act_grad, input_grad );
                    }
                }
                else {
                    Tensor<TDataType, MR> act_grad( use_layer_norm_ ? norm1_output_.getShape() : fc1_output_.getShape() );
                    activation_->backward(
                        use_layer_norm_ ? norm1_output_ : fc1_output_,
                        input_grad,
                        act_grad );

                    if ( use_layer_norm_ ) {
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
         */
        size_t parameterCount() const override {
            size_t total_parameters = 0;
            for ( const auto& module : this->getModules() ) {
                total_parameters += module->parameterCount();
            }
            return total_parameters;
        }

        /**
         * @brief Saves the module state to a ZIP archive.
         */
        void save( mz_zip_archive& zip ) const override {
            for ( const auto& module : this->getModules() ) {
                module->save( zip );
            }
        }

        /**
         * @brief Loads the module state from a ZIP archive.
         */
        void load( mz_zip_archive& zip ) override {
            for ( const auto& module : this->getModules() ) {
                module->load( zip );
            }
        }

        /**
         * @brief Converts the module information to a human-readable string.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "====================" << std::endl;
            oss << "MLP: " << this->getName();
            oss << ", Input shape: (";
            for ( size_t i = 0; i < input_shape_.size(); ++i ) {
                oss << input_shape_[ i ];
                if ( i != input_shape_.size() - 1 ) {
                    oss << ",";
                }
            }
            oss << ")";
            oss << ", Input features: " << input_features_;
            oss << ", Hidden size: " << hidden_size_;
            oss << ", Bias: " << (has_bias_ ? "enabled" : "disabled");
            oss << ", Activation: " << activationTypeToString( activation_type_ );

            if ( dropout_prob_ > 0.0f ) {
                oss << ", Dropout: " << dropout_prob_;
            }

            if ( use_layer_norm_ ) {
                oss << ", Layer Norm: enabled";
            }

            if ( use_residual_ ) {
                oss << ", Residual: enabled";
            }

            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;
            oss << "Sub-Modules..." << std::endl;

            for ( const auto& [name, module] : this->getNamedModules() ) {
                oss << *module;
            }

            return oss.str();
        }

    private:
        std::vector<size_t> input_shape_;
        size_t input_features_{ 0 };
        size_t hidden_size_{ 0 };
        bool has_bias_{ true };
        ActivationType activation_type_{ ActivationType::Gelu };
        float dropout_prob_{ 0.0f };
        bool use_layer_norm_{ false };
        bool use_residual_{ false };
        bool fuse_operations_{ false };

        std::shared_ptr<Linear<TDeviceType, TDataType>> fc1_{ nullptr };
        std::shared_ptr<Module<TDeviceType, TDataType>> activation_{ nullptr };
        std::shared_ptr<Linear<TDeviceType, TDataType>> fc2_{ nullptr };
        std::shared_ptr<LayerNorm<TDeviceType, TDataType>> norm1_{ nullptr };
        std::shared_ptr<Dropout<TDeviceType, TDataType>> dropout1_{ nullptr };

        Tensor<TDataType, MR> fc1_output_;
        Tensor<TDataType, MR> norm1_output_;
        Tensor<TDataType, MR> act_output_;
        Tensor<TDataType, MR> dropout1_output_;
        Tensor<TDataType, MR> fc2_output_;
        Tensor<TDataType, MR> residual_input_;

        void initializeModules() {
            for ( const auto& [name, _] : this->getNamedModules() ) {
                this->removeModule( name );
            }

            auto precision = this->getComputePrecision().getPolicy();

            // First linear layer: input_features -> hidden_size
            auto fc1_config = LinearConfig( input_features_, hidden_size_ )
                .withName( this->getName() + ".fc1" )
                .withDeviceContext( this->getDeviceContext() )
                .withPrecision( precision )
                .withBias( has_bias_ )
                .training( this->isTraining() );

            fc1_ = std::make_shared<Linear<TDeviceType, TDataType>>( fc1_config );
            this->addModule( "fc1", fc1_ );

            // Optional layer normalization
            if ( use_layer_norm_ ) {
                auto norm1_config = LayerNormConfig( hidden_size_ )
                    .withName( this->getName() + ".norm1" )
                    .withDeviceContext( this->getDeviceContext() )
                    .withPrecision( precision )
                    .training( this->isTraining() );

                norm1_ = std::make_shared<LayerNorm<TDeviceType, TDataType>>( norm1_config );
                this->addModule( "norm1", norm1_ );
            }

            // Activation function based on configuration
            switch ( activation_type_ ) {
                case ActivationType::Gelu:
                {
                    auto gelu_config = GeluConfig()
                        .withName( this->getName() + ".gelu" )
                        .withDeviceContext( this->getDeviceContext() )
                        .withPrecision( precision )
                        .training( this->isTraining() );

                    activation_ = std::make_shared<Gelu<TDeviceType, TDataType>>( gelu_config );
                    break;
                }
            }

            this->addModule( "activation", activation_ );

            // Optional dropout
            if ( dropout_prob_ > 0.0f ) {
                auto dropout_config = DropoutConfig( dropout_prob_ )
                    .withName( this->getName() + ".dropout1" )
                    .withDeviceContext( this->getDeviceContext() )
                    .withPrecision( precision )
                    .training( this->isTraining() );

                dropout1_ = std::make_shared<Dropout<TDeviceType, TDataType>>( dropout_config );
                this->addModule( "dropout1", dropout1_ );
            }

            // Second linear layer: hidden_size -> input_features
            auto fc2_config = LinearConfig( hidden_size_, input_features_ )
                .withName( this->getName() + ".fc2" )
                .withDeviceContext( this->getDeviceContext() )
                .withPrecision( precision )
                .withBias( has_bias_ )
                .training( this->isTraining() );

            fc2_ = std::make_shared<Linear<TDeviceType, TDataType>>( fc2_config );
            this->addModule( "fc2", fc2_ );

            // Create intermediate tensors
            std::vector<size_t> hidden_shape = input_shape_;
            if ( !hidden_shape.empty() ) {
                hidden_shape.back() = hidden_size_;
            }
            else {
                hidden_shape = { hidden_size_ };
            }

            fc1_output_ = Tensor<TDataType, MR>( hidden_shape );

            if ( use_layer_norm_ ) {
                norm1_output_ = Tensor<TDataType, MR>( hidden_shape );
            }

            act_output_ = Tensor<TDataType, MR>( hidden_shape );

            if ( dropout_prob_ > 0.0f ) {
                dropout1_output_ = Tensor<TDataType, MR>( hidden_shape );
            }

            fc2_output_ = Tensor<TDataType, MR>( input_shape_ );

            if ( use_residual_ ) {
                residual_input_ = Tensor<TDataType, MR>( input_shape_ );
            }
        }
    };

    export template<typename TDataType = float>
        using CpuMLP = MLP<DeviceType::Cpu, TDataType>;

    export template<typename TDataType = float>
        using CudaMLP = MLP<DeviceType::Cuda, TDataType>;
}