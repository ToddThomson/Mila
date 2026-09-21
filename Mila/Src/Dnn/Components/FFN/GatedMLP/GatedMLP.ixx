/**
 * @file GatedMLP.ixx
 * @brief Gated feed-forward (GatedMLP) block: fused gate+up -> Swiglu gate -> down.
 */

module;
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <optional>

export module Dnn.Components.GatedMLP;
export import :Config;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ActivationType;
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
import Compute.Observation;
import Dnn.Components.Linear;
import Dnn.Components.Swiglu;
import Dnn.Quantization.Weight.Policies;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief Gated feed-forward (GatedMLP) composite component.
     *
     * Device-templated composite implementing the GLU (Gated Linear Unit) family of gated
     * FFNs -- SwiGLU (Llama, Qwen) and GeGLU (Gemma) -- and the single-expert reference for
     * a future MoE layer:
     *   Input -> fc_gate_up Linear(in -> 2H, fused) -> Swiglu gate (2H -> H)
     *         -> fc_down Linear(H -> in) -> Output
     *
     * The gate multiplies TGate(gate half) by the up half: SiLU gives SwiGLU, GELU gives
     * GeGLU. Both projections carry TWeightQuantization; the gate has no weights.
     *
     * MoE-readiness seams (FfnAndMoE.md s9): the injected-context path is the norm
     * (owned-context construction is a standalone convenience); forward operates on
     * the trailing feature dimension so it is valid for both [B, T, in] and gathered
     * [num_tokens, in] layouts.
     *
     * @tparam TDeviceType         Device type for execution.
     * @tparam TPrecision          Tensor data precision. Must be supported on the device.
     * @tparam TGate               Gate activation: Silu (SwiGLU) or Gelu (GeGLU, Gemma).
     * @tparam TWeightQuantization Weight quantization policy of both projections.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision, ActivationType TGate = ActivationType::Silu,
        WeightQuantPolicy TWeightQuantization = NoWeightQuant>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GatedMLP : public CompositeComponent<TDeviceType, TPrecision>
    {
        static_assert( TGate == ActivationType::Silu || TGate == ActivationType::Gelu,
            "GatedMLP gate must be Silu (SwiGLU) or Gelu (GeGLU); other gates await the activation unification." );

    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using ComponentPtr = typename CompositeComponentBase::ComponentPtr;
        using TensorType = Tensor<TPrecision, MR>;
        using LinearType = Linear<TDeviceType, TPrecision, TWeightQuantization>;
        using SwigluType = Swiglu<TDeviceType, TPrecision, TGate>;

        explicit GatedMLP( const std::string& name, const GatedMLPConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : CompositeComponentBase( name ), config_( config )
        {
            config_.validate();

            createGraph();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "GatedMLP: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );

                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~GatedMLP() override = default;

        /**
         * @brief Forward pass: fc_gate_up -> gate -> fc_down.
         *
         * Captures non-owning pointers to child-owned intermediates for backward().
         *
         * @param input Input tensor [..., in_features].
         * @return Reference to the final fc_down output (owned by that child).
         */
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GatedMLP component must be built before calling forward." );
            }

            last_gate_up_out_ = &fc_gate_up_->forward( input );

            last_gate_out_ = &gate_->forward( *last_gate_up_out_ );

            last_final_out_ = &fc_down_->forward( *last_gate_out_ );

            this->publish( ComputePass::Forward, "output", *last_final_out_ );

            return *last_final_out_;
        }

        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GatedMLP component must be built before calling backward." );
            }

            if ( last_gate_up_out_ == nullptr || last_gate_out_ == nullptr )
            {
                throw std::runtime_error( "GatedMLP::backward: forward() must be called before backward() to capture intermediates." );
            }

            auto& down_grad = fc_down_->backward( *last_gate_out_, output_grad );

            auto& gate_grad = gate_->backward( *last_gate_up_out_, down_grad );

            auto& input_grad = fc_gate_up_->backward( input, gate_grad );

            clearForwardCache();

            return input_grad;
        }

        /**
         * @brief Single-token inference convenience with no gradient capture.
         */
        TensorType& decode( const TensorType& input ) const
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "GatedMLP must be built before decode()." );

            auto& gate_up_out = fc_gate_up_->forward( input );

            auto& gate_out = gate_->forward( gate_up_out );

            auto& down_out = fc_down_->forward( gate_out );

            return down_out;
        }

        /**
         * @brief Install shared output slots into the three children (activation pooling).
         *
         * Must be called before build(). Each slot is owned and memory-accounted by the
         * installer, and must cover this component's build shape at its own width:
         * [..., 2H] for the gate+up projection, [..., H] for the gate, [..., in] for the
         * down projection. A pooling caller declares the same intent to getRequiredMemory()
         * through BuildContext::withInstalledOutput().
         */
        void installSharedOutputs(
            std::shared_ptr<TensorType> gate_up_output,
            std::shared_ptr<TensorType> gate_output,
            std::shared_ptr<TensorType> down_output )
        {
            if ( this->isBuilt() )
            {
                throw std::logic_error(
                    "GatedMLP '" + this->getName() + "': installSharedOutputs must be called before build()" );
            }

            childLinear( "fc_gate_up" )->installSharedOutput( std::move( gate_up_output ) );
            childGate()->installSharedOutput( std::move( gate_output ) );
            childLinear( "fc_down" )->installSharedOutput( std::move( down_output ) );
        }

        void zeroGradients() override
        {
            fc_gate_up_->zeroGradients();
            gate_->zeroGradients();
            fc_down_->zeroGradients();
        }

        // save_ is deliberately NOT overridden -- see the note in GptBlock. All three
        // members are resolved out of the child registry by name, so the base traversal
        // covers them and gives each its own scope.

        // ====================================================================
        // Identification and Description
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::GatedMlp;
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "output", ComputePassMask{ ComputePass::Forward } } };
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
         * @brief What onBuilding() would allocate for this context, without allocating.
         *
         * Each child is sized at its own width rather than the generic recursion's single
         * shape, exactly as onBuilding() builds it. Installed outputs are excluded by the
         * children themselves, from their own installed flag or the context's.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            validateInputShape( context.inputShape() );

            MemoryStats stats;

            stats += childLinear( "fc_gate_up" )->getRequiredMemory( context );
            stats += childGate()->getRequiredMemory( gateContext( context ) );
            stats += childLinear( "fc_down" )->getRequiredMemory( downContext( context ) );

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "====================" << std::endl;
            oss << "GatedMLP: " << this->getName() << std::endl;
            oss << "Input features: " << config_.getInputFeatures() << std::endl;
            oss << "Hidden size: " << config_.getHiddenSize() << std::endl;
            oss << "Bias: " << ( config_.hasBias() ? "enabled" : "disabled" ) << std::endl;
            oss << "Gate: " << ( TGate == ActivationType::Gelu ? "GeGLU (GELU)" : "SwiGLU (SiLU)" ) << std::endl;

            if ( this->hasExecutionContext() )
            {
                oss << "Device: " << this->getDeviceId().toString() << std::endl;
            }
            else
            {
                oss << "Device: (context not set)" << std::endl;
            }

            oss << "Sub-Components:" << std::endl;

            if ( fc_gate_up_ )
            {
                oss << "  - fc_gate_up: " << fc_gate_up_->getName() << std::endl;
            }
            if ( gate_ )
            {
                oss << "  - gate: " << gate_->getName() << std::endl;
            }
            if ( fc_down_ )
            {
                oss << "  - fc_down: " << fc_down_->getName() << std::endl;
            }

            return oss.str();
        }

    protected:

        /**
         * @brief Build child graph with the gated shape contract.
         *
         * fc_gate_up receives the input shape; the gate and fc_down receive the fused
         * 2H and the gated H shapes respectively. The child contexts are derived with
         * withShape so the prefill size, the installed-output declaration and the
         * parameter-initialization choice reach the children unchanged.
         */
        void onBuilding( const BuildContext& context ) override
        {
            validateInputShape( context.inputShape() );

            fc_gate_up_ = childLinear( "fc_gate_up" );
            fc_gate_up_->build( context );

            gate_ = childGate();
            gate_->build( gateContext( context ) );

            fc_down_ = childLinear( "fc_down" );
            fc_down_->build( downContext( context ) );

            clearForwardCache();
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            fc_gate_up_->setTrainingMode( training_mode );
            gate_->setTrainingMode( training_mode );
            fc_down_->setTrainingMode( training_mode );
        }

    private:

        GatedMLPConfig config_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        std::shared_ptr<LinearType> fc_gate_up_{ nullptr };
        std::shared_ptr<SwigluType> gate_{ nullptr };
        std::shared_ptr<LinearType> fc_down_{ nullptr };

        // Captured child-owned tensors from the most recent forward() call.
        TensorType* last_gate_up_out_{ nullptr };
        TensorType* last_gate_out_{ nullptr };
        TensorType* last_final_out_{ nullptr };

        void createGraph()
        {
            // Fused gate+up projection: one 2H-wide GEMM (matches Llama's fc_gate_up
            // layout and the converter weight format). The split happens inside Swiglu.
            addLinear( "fc_gate_up", config_.getInputFeatures(), 2 * config_.getHiddenSize() );
            addGate( "gate" );
            addLinear( "fc_down", config_.getHiddenSize(), config_.getInputFeatures() );
        }

        void addLinear( const std::string& suffix, dim_t in_features, dim_t out_features )
        {
            auto cfg = LinearConfig( in_features, out_features )
                .withBias( config_.hasBias() );

            auto component = std::make_shared<LinearType>( this->getName() + "." + suffix, cfg, std::nullopt );

            this->addComponent( component );
        }

        void addGate( const std::string& suffix )
        {
            auto gate = std::make_shared<SwigluType>( this->getName() + "." + suffix, SwigluConfig(), std::nullopt );

            this->addComponent( gate );
        }

        // Children by name, not by member pointer: the members are assigned in onBuilding(),
        // so they are still null when installSharedOutputs() and getRequiredMemory() run.
        std::shared_ptr<LinearType> childLinear( const std::string& suffix ) const
        {
            return this->template getComponentAs<LinearType>( this->getName() + "." + suffix );
        }

        std::shared_ptr<SwigluType> childGate() const
        {
            return this->template getComponentAs<SwigluType>( this->getName() + ".gate" );
        }

        BuildContext gateContext( const BuildContext& context ) const
        {
            shape_t gate_up_shape = context.inputShape();
            gate_up_shape.back() = 2 * config_.getHiddenSize();

            return context.withShape( gate_up_shape );
        }

        BuildContext downContext( const BuildContext& context ) const
        {
            shape_t hidden_shape = context.inputShape();
            hidden_shape.back() = config_.getHiddenSize();

            return context.withShape( hidden_shape );
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.empty() )
            {
                throw std::invalid_argument( "GatedMLP: input must have rank >= 1" );
            }

            int64_t input_features = input_shape.back();

            if ( input_features != config_.getInputFeatures() )
            {
                std::ostringstream oss;
                oss << "GatedMLP: input feature dimension mismatch. Expected "
                    << config_.getInputFeatures() << ", got " << input_features;
                throw std::invalid_argument( oss.str() );
            }
        }

        void clearForwardCache() noexcept
        {
            last_gate_up_out_ = nullptr;
            last_gate_out_ = nullptr;
            last_final_out_ = nullptr;
        }
    };
}
