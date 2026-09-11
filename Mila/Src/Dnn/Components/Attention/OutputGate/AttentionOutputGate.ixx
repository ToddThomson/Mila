/**
 * @file AttentionOutputGate.ixx
 * @brief Gate on the attention output: out = TGate(gate) * value, two separate inputs.
 *
 * Qwen 3.8 sets `attn_output_gate: true`, which doubles the query projection to emit
 * [query | gate] and scales the attention output elementwise by the gated half before the
 * output projection (Specifications/Qwen3.8.md section 2).
 *
 * This is NOT Swiglu with a different name. Swiglu splits ONE buffer into gate|value
 * halves, and both halves are the same projection's output; here the gate arrives from the
 * query projection while the value is the attention output, several operations later. That
 * makes it a binary component -- two independently-produced inputs of the same shape.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <format>
#include <optional>
#include "../../Activations/Activation/Kernels/ElementwiseActivation.h"

export module Dnn.Components.AttentionOutputGate;

export import Dnn.Components.AttentionOutputGateConfig;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.ActivationType;
import Dnn.Components.ActivationConfig;
import Dnn.Components.Activation;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;
import Dnn.TensorOps;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.OperationTraits;
import Compute.CpuMemoryResource;
import Compute.Observation;
import Serialization.ModelArchive;
import Serialization.Mode;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Elementwise gate over two same-shaped inputs: out = TGate(gate) * value.
     *
     * @tparam TDeviceType Compile-time device.
     * @tparam TPrecision  Activation/compute precision.
     * @tparam TGate       Gate function, fixed at compile time. SiLU is the default because it
     *                     is the common case across gated architectures -- NOT because of Qwen
     *                     3.8, whose `output_gate_type: "swish"` is a dead config key its
     *                     reference implementation never reads. That block passes Sigmoid.
     *
     * Stateless: no trainable parameters. The gate WEIGHTS are the second half of the query
     * projection and belong to that Linear, not here -- which is why this component owns
     * nothing and the parameter budget in Qwen3.8.md section 2 counts the gate under
     * "q (+gate)".
     *
     * Two launches, not one: the activation runs through the shared ElementwiseActivationOp
     * and the product through the TensorOps multiply, in place over this component's own
     * output. Fusing them into a single kernel would need a new operation type and a
     * dispatch row for a shape that runs on 16 of 64 layers; worth doing only if it ever
     * measures, and the composition is the correct starting point either way.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision,
        ActivationType TGate = ActivationType::Silu>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class AttentionOutputGate : public Component<TDeviceType, TPrecision>
    {
        static_assert( isElementwiseActivation( TGate ),
            "AttentionOutputGate's gate must be an elementwise function." );

    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using ComponentBase = Component<TDeviceType, TPrecision>;

        explicit AttentionOutputGate( const std::string& name,
            const AttentionOutputGateConfig& config = AttentionOutputGateConfig(),
            std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "AttentionOutputGate: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~AttentionOutputGate() override = default;

        /**
         * @brief out = TGate(gate) * value.
         *
         * @param gate  The gate half of the query projection, [B, T, num_heads * head_dim].
         * @param value The attention output, same shape.
         *
         * The activation writes into this component's output and the product then runs in
         * place over it, which is safe because both are elementwise. Doing it the other way
         * -- activating in place over `gate` -- would corrupt a caller-owned buffer that on
         * the pooled path is shared with every other layer.
         */
        TensorType& forward( const TensorType& gate, const TensorType& value )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error(
                    "AttentionOutputGate::forward: component must be built before forward pass" );
            }

            const auto& gate_shape = gate.shape();

            if ( gate_shape != value.shape() )
            {
                throw std::invalid_argument( std::format(
                    "AttentionOutputGate '{}': gate and value must have the same shape",
                    this->getName() ) );
            }

            if ( output_view_->shape() != gate_shape )
            {
                output_view_.emplace( output_->view( gate_shape ) );
            }

            operation_->forward( gate, *output_view_ );

            multiply( *output_view_, value, *output_view_, this->getExecutionContext() );

            this->publish( ComputePass::Forward, "output", *output_view_ );

            return *output_view_;
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        /**
         * @brief The compile-time gate function realized by this component.
         */
        static constexpr ActivationType getGateType() noexcept
        {
            return TGate;
        }

        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
        {
            SerializationMetadata meta;
            meta.set( "type", "AttentionOutputGate" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() )
                .set( "template_device", deviceTypeToString( TDeviceType ) )
                .set( "template_precision", static_cast<int64_t>( TPrecision ) )
                .set( "template_gate", activationTypeToString( TGate ) );

            archive.writeMetadata( "meta.json", meta );
        }

        dim_t parameterCount() const override { return 0; }
        std::vector<ITensor*> getParameters() const override { return {}; }
        std::vector<ITensor*> getGradients() const override { return {}; }

        const ComponentType getType() const override
        {
            return ComponentType::AttentionOutputGate;
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        /**
         * @brief Install a shared output slot (activation pooling).
         *
         * Must be called before build(). Mirrors Swiglu::installSharedOutput: forward()
         * always returns a shape-adjusted view, so a wider slot never leaks its geometry.
         * The slot is owned and memory-accounted by the installer.
         */
        void installSharedOutput( std::shared_ptr<TensorType> output )
        {
            if ( this->isBuilt() )
                throw std::logic_error( "AttentionOutputGate '" + this->getName()
                    + "': installSharedOutput must be called before build()" );

            output_ = std::move( output );
            output_installed_ = true;
        }

        std::vector<const ITensor*> getOutputs() const override
        {
            if ( output_ == nullptr )
            {
                return {};
            }

            return { output_.get() };
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "output", ComputePassMask{ ComputePass::Forward } } };
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            // An installed shared output slot is owned and counted by the installer.
            if ( output_ != nullptr && !output_installed_ )
            {
                stats.device_state_bytes += output_->getStorageSize();
            }

            return stats;
        }

        /**
         * @brief What onBuilding() would allocate for this context, without allocating.
         *
         * The build context carries ONE input shape; gate and value are the same shape, so
         * the output matches it directly. See Specifications/MemoryFootprint.md.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            MemoryStats stats;

            if ( !output_installed_ && !context.hasInstalledOutput() )
            {
                stats.device_state_bytes +=
                    storageBytes<TPrecision>( elementCount( context.inputShape() ) );
            }

            if ( operation_ )
            {
                stats.device_state_bytes += operation_->getRequiredStateMemorySize( context );
            }

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "AttentionOutputGate: " << this->getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Gate: " << activationTypeToString( TGate ) << std::endl;

            return oss.str();
        }

    protected:

        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const BuildContext& build_context ) override
        {
            const auto& input_shape = build_context.inputShape();

            operation_->build( build_context );

            DeviceId device_id = this->getExecutionContext()->getDeviceId();

            if ( output_installed_ )
            {
                if ( !output_ || output_->size() < elementCount( input_shape ) )
                    throw std::invalid_argument( "AttentionOutputGate '" + this->getName()
                        + "': installed shared output slot is smaller than the build shape requires" );
            }
            else
            {
                output_ = std::make_shared<TensorType>(
                    device_id, input_shape, this->getName() + ".output" );
            }

            output_view_.emplace( output_->view( input_shape ) );
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            operation_->setTrainingMode( training_mode );
        }

    private:

        using Functor = functor_of_t<TGate>;
        using OpType = typename OperationTraits<
            OperationType::ElementwiseActivationOp, TDeviceType, TPrecision, void>::template op_for<Functor>;

        AttentionOutputGateConfig config_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };

        // Self-allocated at build, or an installed shared slot (installSharedOutput)
        // that the component views a prefix of.
        std::shared_ptr<TensorType> output_{ nullptr };
        bool output_installed_{ false };
        std::optional<TensorType> output_view_;

        void createOperation()
        {
            // The backend is the shared elementwise activation op, which reads only the
            // leaky-ReLU alpha off its config; the gate function itself is the functor.
            operation_ = std::make_shared<OpType>(
                this->getExecutionContext(), ActivationConfig( TGate ) );

            if ( !operation_ )
            {
                throw std::runtime_error( std::format(
                    "AttentionOutputGate: failed to create compute backend operation for component '{}'",
                    this->getName() ) );
            }
        }
    };
}
