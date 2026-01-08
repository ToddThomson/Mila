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
     *
     * Design philosophy:
     * - Three-phase initialization: constructor creates architecture graph; onExecutionContextSet() 
     *   propagates context to children; onBuilding() builds children with shapes and allocates buffers
     * - Composite component pattern: manages child components (Linear, activation, LayerNorm)
     * - Context-independent architecture: component graph defined without device knowledge
     * - Shape-agnostic configuration: input_features and hidden_size define architecture
     * - Runtime shape determined at onBuilding() time from actual input tensor
     * - Child components stored as concrete types for type safety and direct access
     *
     * Construction Modes:
     * - **Standalone mode**: Construct with DeviceId to create and own an ExecutionContext.
     *   The component manages the context lifetime and uses it for operation execution.
     * - **Shared mode**: Construct without DeviceId; parent (Network/CompositeComponent) 
     *   provides ExecutionContext via setExecutionContext() after construction.
     *
     * Ownership:
     * - Standalone mode: Component owns its ExecutionContext (stored in owned_exec_context_).
     * - Shared mode: Component borrows ExecutionContext from parent; lifecycle managed externally.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
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

        /**
         * @brief Construct MLP with optional ExecutionContext ownership.
         *
         * Architecture is created immediately in the constructor (context-independent).
         * ExecutionContext binding happens either immediately (standalone mode) or 
         * later when parent calls setExecutionContext() (shared mode).
         *
         * Construction sequence:
         * 1. Validate configuration
         * 2. Create architecture graph (no device required)
         * 3. [Standalone only] Create and bind ExecutionContext
         *
         * **Standalone mode (device_id provided)**:
         * - Creates and owns an ExecutionContext for the specified device.
         * - Registers the owned context with the base Component class via setExecutionContext().
         * - Context is propagated to children via onExecutionContextSet() hook.
         * - Use case: Unit tests, standalone component usage.
         *
         * **Shared mode (device_id not provided)**:
         * - Does not create ExecutionContext; expects parent to provide one.
         * - Parent (Network/CompositeComponent) calls setExecutionContext() after construction.
         * - Context propagated to children when parent sets it.
         * - Use case: Components added to Network.
         *
         * @param config MLP configuration (input_features, hidden_size, activation, etc.).
         * @param device_id Optional device identifier. If provided, creates owned ExecutionContext
         *                  for standalone mode. If nullopt, expects shared context from parent.
         *
         * @throws std::invalid_argument if configuration is invalid.
         * @throws std::invalid_argument if device_id.type does not match TDeviceType.
         * @throws std::runtime_error if ExecutionContext creation fails (standalone mode).
         *
         * @note Architecture is inspectable immediately after construction, even without
         *       an ExecutionContext. Use hasExecutionContext() to check if context is available.
         *
         * @example
         * // Standalone mode (owns context)
         * MLPConfig config(768, 3072);
         * MLP<DeviceType::Cpu, TensorDataType::FP32> mlp(config, Device::Cpu());
         *
         * @example
         * // Shared mode (borrows parent's context)
         * Network<DeviceType::Cpu, TensorDataType::FP32> net(Device::Cpu(), "my_net");
         * auto mlp = std::make_shared<MLP<DeviceType::Cpu, TensorDataType::FP32>>(
         *     MLPConfig(768, 3072), std::nullopt);
         * mlp->setName("mlp");
         * net.addComponent(mlp);
         */
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
         * @brief Forward pass - HOT PATH, pure dispatch to child components.
         *
         * All setup and validation was done in onBuilding(). This method chains
         * forward calls through the MLP structure using pre-allocated buffers.
         */
        void forward( const ITensor& input, ITensor& output )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "MLP component must be built before calling forward." );
            }

            fc1_->forward( input, *fc1_output_ );

            if ( config_.useLayerNorm() )
            {
                norm_->forward( *fc1_output_, *norm_output_ );
                activation_->forward( *norm_output_, *act_output_ );
            }
            else
            {
                activation_->forward( *fc1_output_, *act_output_ );
            }

            fc2_->forward( *act_output_, output );
        }

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to child components.
         *
         * Chains backward calls through the MLP structure in reverse order,
         * using pre-allocated gradient buffers.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "MLP component must be built before calling backward." );
            }

            fc2_->backward( *act_output_, output_grad, *fc2_grad_ );

            if ( config_.useLayerNorm() )
            {
                activation_->backward( *norm_output_, *fc2_grad_, *act_grad_ );

                norm_->backward( *fc1_output_, *act_grad_, *norm_grad_ );

                fc1_->backward( input, *norm_grad_, input_grad );
            }
            else
            {
                activation_->backward( *fc1_output_, *fc2_grad_, *act_grad_ );

                fc1_->backward( input, *act_grad_, input_grad );
            }
        }

        void zeroGradients() override
        {
            // Zero gradients in preallocated buffers
            zero( *fc2_grad_ /* this->getExecutionContext() */);
            
            if ( config_.useLayerNorm() )
            {
                zero( *norm_grad_ /* this->getExecutionContext() */);
            }
            
            zero( *act_grad_ /* this->getExecutionContext() */);

            // Zero gradients in all child components
            fc1_->zeroGradients();
            
            if ( norm_ )
            {
                norm_->zeroGradients();
            }
            
            activation_->zeroGradients();
            
            fc2_->zeroGradients();
        }

        // ====================================================================
        // Serialization
        // ====================================================================

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

        // ====================================================================
        // Component interface
        // ====================================================================

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

        /**
         * @brief Hook invoked during build() to initialize component with input shape.
         *
         * Validates input shape, computes hidden shape, caches typed pointers to children,
         * builds all child components with appropriate shapes, and allocates intermediate buffers.
         *
         * All children have ExecutionContext at this point (propagated via onExecutionContextSet).
         */
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

            auto device = this->getDeviceId();

            fc1_output_ = std::make_shared<TensorType>( device, cached_hidden_shape_ );
            fc1_output_->setName( this->getName() + ".fc1_output" );

            if ( config_.useLayerNorm() )
            {
                norm_output_ = std::make_shared<TensorType>( device, cached_hidden_shape_ );
                norm_output_->setName( this->getName() + ".norm_output" );
            }

            act_output_ = std::make_shared<TensorType>( device, cached_hidden_shape_ );
            act_output_->setName( this->getName() + ".act_output" );

            // Preallocate gradient buffers used in backward() to avoid per-call allocation
            fc2_grad_ = std::make_shared<TensorType>( device, cached_hidden_shape_ );
            fc2_grad_->setName( this->getName() + ".fc2_grad" );
            zero( *fc2_grad_ );

            act_grad_ = std::make_shared<TensorType>( device, cached_hidden_shape_ );
            act_grad_->setName( this->getName() + ".act_grad" );
            zero( *act_grad_ );

            if ( config_.useLayerNorm() )
            {
                norm_grad_ = std::make_shared<TensorType>( device, cached_hidden_shape_ );
                norm_grad_->setName( this->getName() + ".norm_grad" );
                zero( *norm_grad_ );
            }
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagates training mode to all child components (fc1, fc2, activation, norm).
         * Called by Component::setTraining() with the training mutex held.
         *
         * @param is_training New training mode (true = training, false = eval)
         *
         * @note Do not call setTraining() from this hook (reentrancy prohibited).
         */
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

        std::shared_ptr<TensorType> fc1_output_{ nullptr };
        std::shared_ptr<TensorType> norm_output_{ nullptr };
        std::shared_ptr<TensorType> act_output_{ nullptr };

        // Preallocated gradient buffers (allocated in onBuilding)
        std::shared_ptr<TensorType> fc2_grad_{ nullptr };
        std::shared_ptr<TensorType> act_grad_{ nullptr };
        std::shared_ptr<TensorType> norm_grad_{ nullptr };

        /**
         * @brief Create the MLP block architecture (context-independent).
         *
         * Defines the computational graph based on configuration:
         *   fc1 -> [norm] -> activation -> fc2
         *
         * Components are created in shared mode (no ExecutionContext).
         * Context binding happens later via setExecutionContext() which
         * triggers onExecutionContextSet() hook for context propagation.
         *
         * Called from constructor before any device resources are bound.
         * This enables architecture introspection without requiring a device.
         */
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

        /**
         * @brief Helper to create and register a linear layer child component.
         *
         * @param suffix Component name suffix (will be prefixed with MLP name)
         * @param in_features Input feature dimension
         * @param out_features Output feature dimension
         */
        void addLinear( const std::string& suffix, dim_t in_features, dim_t out_features )
        {
            auto cfg = LinearConfig( in_features, out_features )
                .withBias( config_.hasBias() );

            auto component = std::make_shared<LinearType>( this->getName() + "." + suffix, cfg, std::nullopt );

            this->addComponent( component );
        }

        /**
         * @brief Helper to create and register a normalization layer child component.
         *
         * @param suffix Component name suffix
         */
        void addLayerNorm( const std::string& suffix )
        {
            auto cfg = LayerNormConfig().withAxis( -1 );

            auto component = std::make_shared<LayerNormType>( this->getName() + "." + suffix, cfg, std::nullopt );

            this->addComponent( component );
        }

        /**
         * @brief Helper to create and register an activation layer child component.
         *
         * @param suffix Component name suffix
         */
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

        /**
         * @brief Validate input shape for MLP operation.
         */
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
    };
}