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

export module Dnn.Blocks.MLP;
export import :Config;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.Module;
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
import Dnn.Modules.LayerNorm;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Multi-Layer Perceptron (MLP) block for neural networks.
     *
     * This module implements a two-layer MLP with an activation function in between:
     * input -> Linear -> Activation -> Linear -> output
     *
     * Optionally includes layer normalization after the first linear layer.
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
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;
        using CompositeModuleBase = CompositeModule<TDeviceType, TDataType>;

        explicit MLP( const std::string& device_name, const MLPConfig& config )
            : CompositeModuleBase( std::make_shared<DeviceContext>( device_name ), config ), config_( config ) {
            config.validate();
            initializeModules();
        }

        explicit MLP( std::shared_ptr<DeviceContext> device_context, const MLPConfig& config )
            : CompositeModuleBase( device_context, config ), config_( config ) {
            config.validate();
            initializeModules();
        }

        void forward( const Tensor<TDataType, MR>& input, Tensor<TDataType, MR>& output ) {
            fc1_->forward( input, fc1_output_ );

            if ( config_.useLayerNorm() ) {
                norm1_->forward( fc1_output_, norm1_output_ );
                activation_->forward( norm1_output_, act_output_ );
            }
            else {
                activation_->forward( fc1_output_, act_output_ );
            }

            fc2_->forward( act_output_, output );
        }

        void backward(
            const Tensor<TDataType, MR>& input,
            const Tensor<TDataType, MR>& output_grad,
            Tensor<TDataType, MR>& input_grad ) {

            Tensor<TDataType, MR> fc2_grad( act_output_.shape() );
            fc2_->backward( act_output_, output_grad, fc2_grad );

            Tensor<TDataType, MR> act_grad( config_.useLayerNorm() ? norm1_output_.shape() : fc1_output_.shape() );
            activation_->backward(
                config_.useLayerNorm() ? norm1_output_ : fc1_output_,
                fc2_grad,
                act_grad );

            if ( config_.useLayerNorm() ) {
                Tensor<TDataType, MR> norm1_grad( fc1_output_.shape() );
                norm1_->backward( fc1_output_, act_grad, norm1_grad );
                fc1_->backward( input, norm1_grad, input_grad );
            }
            else {
                fc1_->backward( input, act_grad, input_grad );
            }
        }

        size_t parameterCount() const override {
            size_t total_parameters = 0;
            for ( const auto& module : this->getModules() ) {
                total_parameters += module->parameterCount();
            }
            return total_parameters;
        }

        void save( ModelArchive& archive ) const override {
            for ( const auto& module : this->getModules() ) {
                module->save( archive );
            }
        }

        void load( ModelArchive& archive ) override {
            for ( const auto& module : this->getModules() ) {
                module->load( archive );
            }
        }

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

            if ( config_.useLayerNorm() ) {
                oss << "Layer Norm: enabled" << std::endl;
            }

            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;
            oss << "Sub-Modules..." << std::endl;

            for ( const auto& [name, module] : this->getNamedModules() ) {
                oss << module->toString();
            }

            return oss.str();
        }

    private:
        MLPConfig config_;

        std::shared_ptr<Linear<TDeviceType, TDataType>> fc1_{ nullptr };
        std::shared_ptr<Module<TDeviceType, TDataType>> activation_{ nullptr };
        std::shared_ptr<Linear<TDeviceType, TDataType>> fc2_{ nullptr };
        std::shared_ptr<LayerNorm<TDeviceType, TDataType>> norm1_{ nullptr };

        Tensor<TDataType, MR> fc1_output_;
        Tensor<TDataType, MR> norm1_output_;
        Tensor<TDataType, MR> act_output_;

        void initializeModules() {
            for ( const auto& [name, _] : this->getNamedModules() ) {
                this->removeModule( name );
            }

            auto fc1_config = LinearConfig( config_.getInputFeatures(), config_.getHiddenSize() )
                .withName( this->getName() + ".fc1" )
                .withBias( config_.hasBias() )
                .withTraining( this->isTraining() );

            fc1_ = std::make_shared<Linear<TDeviceType, TDataType>>( this->getDeviceContext(), fc1_config );
            this->addModule( "fc1", fc1_ );

            if ( config_.useLayerNorm() ) {
                auto norm1_config = LayerNormConfig( config_.getHiddenSize() )
                    .withName( this->getName() + ".norm1" )
                    .withTraining( this->isTraining() );

                norm1_ = std::make_shared<LayerNorm<TDeviceType, TDataType>>( this->getDeviceContext(), norm1_config );
                this->addModule( "norm1", norm1_ );
            }

            switch ( config_.getActivationType() ) {
                case ActivationType::Gelu:
                {
                    auto gelu_config = GeluConfig()
                        .withName( this->getName() + ".gelu" )
                        .withTraining( this->isTraining() );

                    activation_ = std::make_shared<Gelu<TDeviceType, TDataType>>( this->getDeviceContext(), gelu_config );
                    break;
                }
            }

            this->addModule( "activation", activation_ );

            auto fc2_config = LinearConfig( config_.getHiddenSize(), config_.getInputFeatures() )
                .withName( this->getName() + ".fc2" )
                .withBias( config_.hasBias() )
                .withTraining( this->isTraining() );

            fc2_ = std::make_shared<Linear<TDeviceType, TDataType>>( this->getDeviceContext(), fc2_config );
            this->addModule( "fc2", fc2_ );

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
        }
    };

    export template<typename TDataType = float>
        using CpuMLP = MLP<DeviceType::Cpu, TDataType>;

    export template<typename TDataType = float>
        using CudaMLP = MLP<DeviceType::Cuda, TDataType>;
}