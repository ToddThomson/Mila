/**
 * @file MLP.ixx
 * @brief Dense feed-forward (MLP) block: Linear -> GELU -> Linear.
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

export module Dnn.Components.MLP;
export import :Config;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.CompositeComponent;
import Compute.MemoryResource;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.CpuMemoryResource;
import Dnn.Components.Linear;
import Dnn.Components.Gelu;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Dense feed-forward (MLP) composite component.
     *
     * Device-templated composite implementing the GPT-2 dense FFN:
     *   Input -> Linear(in_features, hidden_size) -> GELU -> Linear(hidden_size, in_features) -> Output
     *
     * The component is honest about the single activation it supports: GELU. The
     * gated FFN family (SwiGLU and friends) is a separate component (GatedMLP) with
     * a different 2H -> H shape contract; it is not expressible here and is not a
     * runtime option. A future generalized elementwise Activation component will
     * replace the fixed Gelu child, at which point the activation function becomes a
     * compile-time parameter (see Specifications/FfnAndMoE.md). Until then MLP is the
     * dense GELU FFN, full stop.
     *
     * The component composes child components (two Linear projections and a GELU)
     * and delegates forward/backward calls to them. Child components own their
     * intermediate tensors; MLP stores non-owning pointers to those tensors after
     * forward() to chain backward().
     *
     * Threading: call sites must ensure that forward/backward/zeroGradients are
     * invoked in a thread-safe manner relative to one another; this class does not
     * provide internal synchronization.
     *
     * @tparam TDeviceType Device type for execution (CPU, CUDA, ...).
     * @tparam TPrecision  Tensor data precision. Must be supported on the device.
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

        /**
         * @brief Construct an MLP component.
         *
         * The constructor validates the provided `config`, constructs the internal
         * child component graph (fc1 -> gelu -> fc2), and optionally creates and
         * assigns an execution context when `device_id` is provided.
         *
         * @param name      Component name used to name child subcomponents.
         * @param config    MLP configuration (input features, hidden size, bias).
         * @param device_id Optional device identifier; when present the MLP creates an owned execution context
         *                  bound to that device and sets it on the component. If the provided `device_id`
         *                  type does not match the template `TDeviceType`, an exception is thrown.
         *
         * @throws std::invalid_argument if `config` is invalid (via config.validate()).
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
         *   - gelu_->forward(...)
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

            last_act_out_ = &gelu_->forward( *last_fc1_out_ );

            last_final_out_ = &fc2_->forward( *last_act_out_ );

            return *last_final_out_;
        }

        /**
         * @brief Backward pass using captured forward intermediates.
         *
         * Uses the child-owned tensors captured by the most recent `forward()` invocation
         * to chain backward calls without recomputing forward:
         *   - fc2_->backward(captured_activation_output, output_grad)
         *   - gelu_->backward(...)
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

            if ( last_fc1_out_ == nullptr || last_act_out_ == nullptr )
            {
                throw std::runtime_error( "MLP::backward: forward() must be called before backward() to capture intermediates." );
            }

            auto& fc2_grad = fc2_->backward( *last_act_out_, output_grad );

            auto& act_grad = gelu_->backward( *last_fc1_out_, fc2_grad );

            auto& input_grad = fc1_->backward( input, act_grad );

            clearForwardCache();

            return input_grad;
        }

        /**
         * @brief Single-token inference convenience: fc1 -> gelu -> fc2 with no gradient capture.
         *
         * Relies on single-stream ordering for inter-op dependencies; the caller
         * synchronizes before reading results on the host.
         */
        TensorType& decode( const TensorType& input ) const
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "MLP must be built before decode()." );

            auto& fc1_out = fc1_->forward( input );

            auto& act_out = gelu_->forward( fc1_out );

            auto& fc2_out = fc2_->forward( act_out );

            return fc2_out;
        }

        /**
         * @brief Zero gradients for all child components.
         */
        void zeroGradients() override
        {
            fc1_->zeroGradients();

            if ( gelu_ )
            {
                gelu_->zeroGradients();
            }

            fc2_->zeroGradients();
        }

        // save_ is deliberately NOT overridden -- see the note in GptBlock. fc1_, gelu_ and
        // fc2_ are all resolved out of the child registry by name, so the base traversal
        // covers them and gives each its own scope.

        // ====================================================================
        // Identification and Description
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::Mlp;
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
            {
                stats += child->getMemoryStats();
            }

            return stats;
        }

        /**
         * @brief Human-readable status and configuration summary.
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
            oss << "Activation: Gelu" << std::endl;

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

            if ( gelu_ )
            {
                oss << "  - gelu: " << gelu_->getName() << std::endl;
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
         * Validates the input shape, computes the hidden shape, and builds each child
         * with the appropriate shape: fc1 receives the input shape, the GELU and fc2
         * receive the hidden shape. After building, cached forward pointers are cleared.
         *
         * @param context Build context; `inputShape().back()` must equal config_.getInputFeatures().
         *
         * @throws std::invalid_argument if the input shape rank < 1 or last dimension mismatches config.
         */
        void onBuilding( const BuildContext& context ) override
        {
            const auto& input_shape = context.inputShape();
            validateInputShape( input_shape );

            cached_input_shape_ = input_shape;

            cached_hidden_shape_ = input_shape;
            cached_hidden_shape_.back() = config_.getHiddenSize();

            fc1_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_1" );
            fc1_->build( context );

            gelu_ = this->template getComponentAs<GeluType>( this->getName() + ".gelu" );
            BuildContext gelu_context( cached_hidden_shape_, context.getRuntimeMode() );
            gelu_->build( gelu_context );

            fc2_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_2" );
            BuildContext fc2_context( cached_hidden_shape_, context.getRuntimeMode() );
            fc2_->build( fc2_context );

            clearForwardCache();
        }

        /**
         * @brief Propagate training-mode changes to child components.
         *
         * @param training_mode New training mode.
         */
        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            fc1_->setTrainingMode( training_mode );
            gelu_->setTrainingMode( training_mode );
            fc2_->setTrainingMode( training_mode );
        }

    private:

        MLPConfig config_;

        shape_t cached_input_shape_;
        shape_t cached_hidden_shape_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        std::shared_ptr<LinearType> fc1_{ nullptr };
        std::shared_ptr<GeluType> gelu_{ nullptr };
        std::shared_ptr<LinearType> fc2_{ nullptr };

        // Captured child-owned tensors from the most recent forward() call.
        // These are non-owning raw pointers to tensors owned by the child components.
        TensorType* last_fc1_out_{ nullptr };
        TensorType* last_act_out_{ nullptr };
        TensorType* last_final_out_{ nullptr };

        /**
         * @brief Build the internal component graph: fc1 -> gelu -> fc2.
         *
         * Called from the constructor; does not perform shape-dependent build calls.
         */
        void createGraph()
        {
            addLinear( "fc_1", config_.getInputFeatures(), config_.getHiddenSize() );
            addGelu( "gelu" );
            addLinear( "fc_2", config_.getHiddenSize(), config_.getInputFeatures() );
        }

        /**
         * @brief Helper to create and add a Linear child component.
         *
         * @param suffix       Suffix appended to parent name for the child component.
         * @param in_features  Number of input features for the linear layer.
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
         * @brief Helper to create and add the GELU activation child component.
         *
         * @param suffix Suffix appended to parent name for the activation child component.
         */
        void addGelu( const std::string& suffix )
        {
            auto gelu = std::make_shared<GeluType>( this->getName() + "." + suffix, GeluConfig(), std::nullopt );

            this->addComponent( gelu );
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
         */
        void clearForwardCache() noexcept
        {
            last_fc1_out_ = nullptr;
            last_act_out_ = nullptr;
            last_final_out_ = nullptr;
        }
    };
}
