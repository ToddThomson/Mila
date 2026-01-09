/**
 * @file MLP.ixx
 * @brief Multi-Layer Perceptron (MLP) block for neural networks.
 */

module;
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <optional>

export module Dnn.Blocks.MLP;
export import :Config;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Component;
import Dnn.CompositeComponent;
import Dnn.ActivationType;
import Compute.Precision;
import Compute.MemoryResource;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.OperationRegistry;
import Dnn.Components.Linear;
import Dnn.Components.Gelu;
import Dnn.Components.LayerNorm;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Multi-Layer Perceptron (MLP) block for neural networks.
     *
     * Device-templated composite component that implements a standard MLP structure:
     *   Input -> Linear(in_features, hidden_size) -> [LayerNorm] -> Activation -> Linear(hidden_size, in_features) -> Output
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class MLP : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using ComponentPtr = typename CompositeComponentBase::ComponentPtr;
        using TensorType = Tensor<TPrecision, MR>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using GeluType = Gelu<TDeviceType, TPrecision>;
        using LayerNormType = LayerNorm<TDeviceType, TPrecision>;

        explicit MLP( const std::string& name, const MLPConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : CompositeComponentBase( name ), config_( config )
        {
            config_.validate();

            createGraph();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "MLP: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );

                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~MLP() override = default;

        /**
         * @brief Forward pass - returns component-owned output tensor.
         *
         * Chains child component forward calls using their new API (returns ITensor*).
         *
         * @param input Input tensor bound to the component device.
         * @return Pointer to ITensor containing output (owned by final Linear child).
         *
         * @throws std::runtime_error if component is not built.
         */
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "MLP component must be built before calling forward." );
            }

            last_fc1_out_ = &fc1_->forward( input );

            last_norm_out_ = nullptr;
            last_act_out_ = nullptr;

            if ( config_.useLayerNorm() )
            {
                last_norm_out_ = &norm_->forward( *last_fc1_out_ );
                last_act_out_ = &activation_->forward( *last_norm_out_ );
            }
            else
            {
                last_act_out_ = &activation_->forward( *last_fc1_out_ );
            }

            last_final_out_ = &fc2_->forward( *last_act_out_ );

            return *last_final_out_;
        }

        /**
         * @brief Backward pass - returns pointer to input gradient tensor.
         *
         * Chains child component backward calls using their new API (returns ITensor*).
         *
         * The backward implementation *uses* the tensors produced by the most recent
         * forward() call (captured above). This avoids recomputing forward during backward.
         *
         * @param input Original forward input.
         * @param output_grad Gradient w.r.t. MLP output.
         * @return Pointer to ITensor containing gradient w.r.t. input (owned by fc1 child).
         *
         * @throws std::runtime_error if component is not built or forward() was not called.
         */
        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "MLP component must be built before calling backward." );
            }

            // Require that forward() was called so we have the child-owned intermediate tensors.
            if ( last_fc1_out_ == nullptr || last_act_out_ == nullptr )
            {
                throw std::runtime_error( "MLP::backward: forward() must be called before backward() to capture intermediates." );
            }

            // Backprop through fc2 using the activation output captured during forward.
            auto& fc2_grad = fc2_->backward( *last_act_out_, output_grad );

            // Backprop through activation and optional norm using captured forward tensors.
            if ( config_.useLayerNorm() )
            {
                if ( last_norm_out_ == nullptr )
                {
                    throw std::runtime_error( "MLP::backward: missing stored norm output for backward chaining" );
                }

                auto& act_grad = activation_->backward( *last_norm_out_, fc2_grad );

                auto& norm_grad = norm_->backward( *last_fc1_out_, act_grad );

                auto& input_grad = fc1_->backward( input, norm_grad );

                // Clear cached forward pointers to avoid accidental reuse across calls.
                clearForwardCache();

                return input_grad;
            }
            else
            {
                auto& act_grad = activation_->backward( *last_fc1_out_, fc2_grad );

                auto& input_grad = fc1_->backward( input, act_grad );

                clearForwardCache();

                return input_grad;
            }
        }

        void zeroGradients() override
        {
            // Zero gradients in child components and preallocated buffers if present.
            fc1_->zeroGradients();

            if ( norm_ )
            {
                norm_->zeroGradients();
            }

            activation_->zeroGradients();

            fc2_->zeroGradients();
        }

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            fc1_->save_( archive, mode );

            if ( norm_ )
            {
                norm_->save_( archive, mode );
            }

            activation_->save_( archive, mode );
            fc2_->save_( archive, mode );
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "====================" << std::endl;
            oss << "MLP: " << this->getName() << std::endl;
            oss << "Input features: " << config_.getInputFeatures() << std::endl;
            oss << "Hidden size: " << config_.getHiddenSize() << std::endl;
            oss << "Bias: " << ( config_.hasBias() ? "enabled" : "disabled" ) << std::endl;
            oss << "Activation: " << activationTypeToString( config_.getActivationType() ) << std::endl;
            oss << "Layer Norm: " << ( config_.useLayerNorm() ? "enabled" : "disabled" ) << std::endl;

            if ( this->hasExecutionContext() )
            {
                oss << "Device: " << this->getDeviceId().toString() << std::endl;
            }
            else
            {
                oss << "Device: (context not set)" << std::endl;
            }

            if ( this->isBuilt() )
            {
                oss << "Parameter count: " << this->parameterCount() << std::endl;

                oss << "Input shape: ("; 
                for ( size_t i = 0; i < cached_input_shape_.size(); ++i )
                {
                    oss << cached_input_shape_[ i ];
                    if ( i != cached_input_shape_.size() - 1 )
                        oss << ", ";
                }
                oss << ")" << std::endl;

                oss << "Hidden shape: ("; 
                for ( size_t i = 0; i < cached_hidden_shape_.size(); ++i )
                {
                    oss << cached_hidden_shape_[ i ];
                    if ( i != cached_hidden_shape_.size() - 1 )
                        oss << ", ";
                }
                oss << ")" << std::endl;
            }

            oss << "Sub-Components:" << std::endl;

            if ( fc1_ )
            {
                oss << "  - fc1: " << fc1_->getName() << std::endl;
            }

            if ( norm_ )
            {
                oss << "  - norm: " << norm_->getName() << std::endl;
            }

            if ( activation_ )
            {
                oss << "  - activation: " << activation_->getName() << std::endl;
            }

            if ( fc2_ )
            {
                oss << "  - fc2: " << fc2_->getName() << std::endl;
            }

            return oss.str();
        }

    protected:

        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            cached_input_shape_ = input_shape;

            cached_hidden_shape_ = input_shape;
            cached_hidden_shape_.back() = config_.getHiddenSize();

            fc1_ = this->template getComponentAs<LinearType>( this->getName() + ".fc1" );
            fc1_->build( input_shape );

            if ( config_.useLayerNorm() )
            {
                norm_ = this->template getComponentAs<LayerNormType>( this->getName() + ".norm" );
                norm_->build( cached_hidden_shape_ );
            }

            activation_ = this->template getComponentAs<GeluType>( this->getName() + ".act" );
            activation_->build( cached_hidden_shape_ );

            fc2_ = this->template getComponentAs<LinearType>( this->getName() + ".fc2" );
            fc2_->build( cached_hidden_shape_ );

            // Clear any cached forward pointers on new build.
            clearForwardCache();

            // Preallocate nothing for child-owned forward/backward buffers; children own their buffers.
            // Keep cached shapes for validation and introspection.
        }

        void onTrainingChanging( bool is_training ) override
        {
            if ( fc1_ )
            {
                fc1_->setTraining( is_training );
            }

            if ( norm_ )
            {
                norm_->setTraining( is_training );
            }

            if ( activation_ )
            {
                activation_->setTraining( is_training );
            }

            if ( fc2_ )
            {
                fc2_->setTraining( is_training );
            }
        }

    private:

        MLPConfig config_;

        shape_t cached_input_shape_;
        shape_t cached_hidden_shape_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        std::shared_ptr<LinearType> fc1_{ nullptr };
        std::shared_ptr<GeluType> activation_{ nullptr };
        std::shared_ptr<LinearType> fc2_{ nullptr };
        std::shared_ptr<LayerNormType> norm_{ nullptr };

        // Captured child-owned tensors from the most recent forward() call.
        // These are non-owning raw pointers to ITensor objects owned by the child components.
        TensorType* last_fc1_out_{ nullptr };
        TensorType* last_norm_out_{ nullptr };
        TensorType* last_act_out_{ nullptr };
        TensorType* last_final_out_{ nullptr };

        void createGraph()
        {
            addLinear( "fc1", config_.getInputFeatures(), config_.getHiddenSize() );

            if ( config_.useLayerNorm() )
            {
                addLayerNorm( "norm" );
            }

            addActivation( "act" );
            addLinear( "fc2", config_.getHiddenSize(), config_.getInputFeatures() );
        }

        void addLinear( const std::string& suffix, dim_t in_features, dim_t out_features )
        {
            auto cfg = LinearConfig( in_features, out_features )
                .withBias( config_.hasBias() );

            auto component = std::make_shared<LinearType>( this->getName() + "." + suffix, cfg, std::nullopt );

            this->addComponent( component );
        }

        void addLayerNorm( const std::string& suffix )
        {
            auto cfg = LayerNormConfig().withAxis( -1 );

            auto component = std::make_shared<LayerNormType>( this->getName() + "." + suffix, cfg, std::nullopt );

            this->addComponent( component );
        }

        void addActivation( const std::string& suffix )
        {
            switch ( config_.getActivationType() )
            {
                case ActivationType::Gelu:
                {
                    auto cfg = GeluConfig();
                    auto component = std::make_shared<GeluType>( this->getName() + "." + suffix, cfg, std::nullopt );
                    this->addComponent( component );
                    break;
                }
                default:
                    throw std::invalid_argument( "MLP: unsupported activation type" );
            }
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.empty() )
            {
                throw std::invalid_argument( "MLP: input must have rank >= 1" );
            }

            int64_t input_features = input_shape.back();

            if ( input_features != config_.getInputFeatures() )
            {
                std::ostringstream oss;
                oss << "MLP: input feature dimension mismatch. Expected "
                    << config_.getInputFeatures() << ", got " << input_features;
                throw std::invalid_argument( oss.str() );
            }
        }

        void clearForwardCache() noexcept
        {
            last_fc1_out_ = nullptr;
            last_norm_out_ = nullptr;
            last_act_out_ = nullptr;
            last_final_out_ = nullptr;
        }
    };
}