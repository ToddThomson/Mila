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
     * @brief Multi-Layer Perceptron (MLP) composite component.
     *
     * Device-templated composite component that implements a standard MLP structure:
     *   Input -> Linear(in_features, hidden_size) -> [LayerNorm] -> Activation -> Linear(hidden_size, in_features) -> Output
     *
     * The component composes child components (Linear, optional LayerNorm, Activation) and
     * delegates forward/backward calls to them. Child components own intermediate tensors;
     * MLP stores non-owning pointers to those tensors after forward() to chain backward().
     *
     * Threading: call sites must ensure that forward/backward/zeroGradients are invoked
     * in a thread-safe manner relative to one another; this class does not provide internal
     * synchronization.
     *
     * @tparam TDeviceType Device type for execution (CPU, CUDA, ...).
     * @tparam TPrecision  Tensor data precision (Fp32, Fp16, etc.). Must be supported on the device.
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
         * @brief Construct an MLP component.
         *
         * The constructor validates the provided `config`, constructs the internal
         * child component graph, and optionally creates and assigns an execution context
         * when `device_id` is provided.
         *
         * @param name      Component name used to name child subcomponents.
         * @param config    MLP configuration (input features, hidden size, activation, bias, layer-norm flag).
         * @param device_id Optional device identifier; when present the MLP creates an owned execution context
         *                  bound to that device and sets it on the component. If the provided `device_id`
         *                  type does not match the template `TDeviceType`, an exception is thrown.
         *
         * @throws std::invalid_argument if `device_id` is present but has a mismatched device type.
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

        /**
         * @brief Default destructor.
         *
         * Child components are stored as shared_ptr and will be destroyed automatically.
         */
        ~MLP() override = default;

        /**
         * @brief Forward pass.
         *
         * Chains child component forward calls:
         *   - fc1_->forward(input)
         *   - optional norm_->forward(...)
         *   - activation_->forward(...)
         *   - fc2_->forward(...)
         *
         * The function stores non-owning pointers to child-owned intermediate tensors produced
         * during the forward call; these pointers are used by `backward()` to chain gradients.
         *
         * Preconditions:
         *   - Component must be built (onBuilding called).
         *   - Input tensor must be bound to the same device/context as the component.
         *
         * @param input Input tensor bound to this component's device/context.
         * @return Reference to the output tensor produced by the final Linear child (owned by that child).
         *
         * @throws std::runtime_error if the component is not built prior to calling forward.
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
         * @brief Backward pass using captured forward intermediates.
         *
         * Uses the child-owned tensors captured by the most recent `forward()` invocation
         * to chain backward calls without recomputing forward:
         *   - fc2_->backward(captured_activation_output, output_grad)
         *   - activation_->backward(...)
         *   - optional norm_->backward(...)
         *   - fc1_->backward(input, ...)
         *
         * The method clears the cached forward pointers before returning to avoid accidental reuse.
         *
         * Preconditions:
         *   - Component must be built.
         *   - `forward()` must have been called previously to populate internal forward caches.
         *
         * @param input       The original input tensor passed to `forward()`; required by fc1_->backward.
         * @param output_grad Gradient tensor w.r.t. the MLP output.
         * @return Reference to the input-gradient tensor (owned by the `fc1` child).
         *
         * @throws std::runtime_error if the component is not built or if `forward()` was not called.
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

        /**
         * @brief Zero gradients for all child components.
         *
         * Recursively zeroes optimizer/parameter gradients in children. Safe to call
         * regardless of build state; child pointers are checked before use.
         */
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

        /**
         * @brief Serialize parameters to a model archive.
         *
         * Saves child component parameters and state into the provided archive in a deterministic order.
         *
         * @param archive Serialization archive to write to.
         * @param mode    Serialization mode (enum driven by Serialization::Mode).
         */
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

        /**
         * @brief Human-readable status and configuration summary.
         *
         * Produces a multi-line string describing the component name, shapes, parameter counts,
         * activation and layer-norm usage, device assignment (if set), and child component names.
         *
         * @return String containing component introspection information suitable for logging.
         */
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
         * @brief Build-time callback invoked by the CompositeComponent framework.
         *
         * Validates the provided `input_shape`, computes the hidden shape, and builds
         * each child component with the appropriate shape. After building, any cached
         * forward pointers are cleared to guarantee a clean state for subsequent forward calls.
         *
         * @param input_shape Shape of the input tensor. The last dimension must equal config_.getInputFeatures().
         *
         * @throws std::invalid_argument if `input_shape` rank < 1 or last dimension mismatches config.
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

            // Clear any cached forward pointers on new build.
            clearForwardCache();
        }

        /**
         * @brief Called when the training/inference mode changes.
         *
         * Propagates the training flag to child components so they can adjust behavior
         * (dropout, batch/statistics, etc.) as needed.
         *
         * @param is_training True if switching to training mode; false for evaluation mode.
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

        // Captured child-owned tensors from the most recent forward() call.
        // These are non-owning raw pointers to ITensor objects owned by the child components.
        TensorType* last_fc1_out_{ nullptr };
        TensorType* last_norm_out_{ nullptr };
        TensorType* last_act_out_{ nullptr };
        TensorType* last_final_out_{ nullptr };

        /**
         * @brief Build the internal component graph according to `config_`.
         *
         * Adds children (fc1, optional norm, activation, fc2) using helper add* methods.
         * Called from the constructor; does not perform shape-dependent build calls.
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
         * @brief Helper to create and add a Linear child component.
         *
         * The created Linear component uses the parent's name plus the provided suffix.
         *
         * @param suffix      Suffix appended to parent name for the child component.
         * @param in_features Number of input features for the linear layer.
         * @param out_features Number of output features for the linear layer.
         */
        void addLinear( const std::string& suffix, dim_t in_features, dim_t out_features )
        {
            auto cfg = LinearConfig( in_features, out_features )
                .withBias( config_.hasBias() );

            auto component = std::make_shared<LinearType>( this->getName() + "." + suffix, cfg, std::nullopt );

            this->addComponent( component );
        }

        /**
         * @brief Helper to create and add a LayerNorm child component.
         *
         * The LayerNorm is constructed with axis=-1 by default to normalize the last dimension.
         *
         * @param suffix Suffix appended to parent name for the child component.
         */
        void addLayerNorm( const std::string& suffix )
        {
            auto cfg = LayerNormConfig().withAxis( -1 );

            auto component = std::make_shared<LayerNormType>( this->getName() + "." + suffix, cfg, std::nullopt );

            this->addComponent( component );
        }

        /**
         * @brief Helper to create and add the configured activation component.
         *
         * Currently supports ActivationType::Gelu. Throws on unsupported activations.
         *
         * @param suffix Suffix appended to parent name for the child component.
         *
         * @throws std::invalid_argument if the activation type in config_ is unsupported.
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
         * @brief Validate input shape against the MLP configuration.
         *
         * Ensures the input tensor has rank >= 1 and that its last dimension
         * matches `config_.getInputFeatures()`.
         *
         * @param input_shape Shape to validate.
         *
         * @throws std::invalid_argument when rank < 1 or last-dimension mismatch.
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

        /**
         * @brief Clear cached non-owning forward pointers.
         *
         * Safe to call at any time; used to avoid accidental reuse of child-owned
         * tensors across forward/backward cycles. No side effects beyond pointer reset.
         */
        void clearForwardCache() noexcept
        {
            last_fc1_out_ = nullptr;
            last_norm_out_ = nullptr;
            last_act_out_ = nullptr;
            last_final_out_ = nullptr;
        }
    };
}