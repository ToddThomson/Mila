/**
 * @file Activation.ixx
 * @brief Unified elementwise activation component.
 *
 * Device- and precision-templated component whose activation function is a
 * compile-time ActivationType parameter (the same shape Linear carries for
 * TWeightQuantization). It maps that enum to a functor from the shared
 * ElementwiseActivation library and delegates compute to the
 * ElementwiseActivationOp backend. Added alongside Gelu, which it folds in later
 * (see Specifications/FfnAndMoE.md section 5). Stateless: no trainable parameters.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <format>
#include <utility>
#include <optional>
#include "Kernels/ElementwiseActivation.h"

export module Dnn.Components.Activation;
export import Dnn.Components.ActivationConfig;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.ActivationType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.OperationTraits;
import Compute.CpuMemoryResource;
import Serialization.ModelArchive;
import Serialization.Tensor;
import Serialization.Mode;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Compile-time map from an elementwise ActivationType to its functor.
     *
     * The primary template is intentionally undefined: instantiating it for a
     * non-elementwise value (e.g. ActivationType::Swiglu) is a hard compile error,
     * matching the "missing specialization = hard compile error" dispatch contract.
     */
    export template<ActivationType TFn>
    struct functor_of;

    template<> struct functor_of<ActivationType::None> { using type = Activations::Identity; };
    template<> struct functor_of<ActivationType::Gelu> { using type = Activations::GeluTanh; };
    template<> struct functor_of<ActivationType::Silu> { using type = Activations::Silu; };
    template<> struct functor_of<ActivationType::Relu> { using type = Activations::Relu; };
    template<> struct functor_of<ActivationType::Tanh> { using type = Activations::Tanh; };
    template<> struct functor_of<ActivationType::Sigmoid> { using type = Activations::Sigmoid; };
    template<> struct functor_of<ActivationType::LeakyRelu> { using type = Activations::LeakyRelu; };
    template<> struct functor_of<ActivationType::Mish> { using type = Activations::Mish; };

    // Exported because it is the enum->functor bridge for the WHOLE tree, not just for this
    // component: AttentionOutputGate reaches the same elementwise backend through it, and a
    // second copy of the map would be a second place for an activation to go missing.
    export template<ActivationType TFn>
    using functor_of_t = typename functor_of<TFn>::type;

    /**
     * @brief Unified elementwise activation component.
     *
     * @tparam TDeviceType Compile-time device (Cpu or Cuda).
     * @tparam TPrecision  Activation/compute precision.
     * @tparam TFn         Elementwise activation function, fixed at compile time.
     *
     * Construction follows the standalone/shared pattern (mirrors Gelu): pass a
     * DeviceId to own an ExecutionContext, or omit it and let a parent inject one.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision, ActivationType TFn = ActivationType::Gelu>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Activation : public Component<TDeviceType, TPrecision>
    {
        static_assert( isElementwiseActivation( TFn ),
            "Activation supports only elementwise functions; gated functions (Swiglu) belong to GatedMLP." );

    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using ComponentBase = Component<TDeviceType, TPrecision>;

        explicit Activation( const std::string& name, const ActivationConfig& config = ActivationConfig( TFn ),
            std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "Activation: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );

                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~Activation() override = default;

        // ====================================================================
        // Computation Dispatch
        // ====================================================================

        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Activation::forward: component must be built before forward pass" );
            }

            const auto& input_shape = input.shape();

            if ( output_view_->shape() != input_shape )
            {
                output_view_.emplace( output_->view( input_shape ) );
            }

            operation_->forward( input, *output_view_ );

            return *output_view_;
        }

        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Activation::backward: component must be built before backward pass" );
            }

            if ( !this->isTrainingMode() )
            {
                throw std::runtime_error( "Activation::backward: component must be in training mode" );
            }

            // Backends accumulate into input_grad_; zero it first to prevent buildup.
            zero( *input_grad_ );

            operation_->backward( input, output_grad, *input_grad_ );

            return *input_grad_;
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        /**
         * @brief The compile-time activation function realized by this component.
         */
        static constexpr ActivationType getActivationType() noexcept
        {
            return TFn;
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)mode;

            SerializationMetadata meta;
            meta.set( "type", "Activation" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() )
                .set( "template_device", deviceTypeToString( TDeviceType ) )
                .set( "template_precision", static_cast<int64_t>( TPrecision ) )
                .set( "template_function", activationTypeToString( TFn ) );

            archive.writeMetadata( "meta.json", meta );

            archive.writeMetadata( "config.json", config_.toMetadata() );
        }

        static std::unique_ptr<Activation> fromArchive_(
            ModelArchive& archive,
            const std::string& component_name,
            IExecutionContext* exec_context )
        {
            try
            {
                SerializationMetadata meta = archive.readMetadata( "meta.json" );
                validateMetadata_( meta, component_name );

                SerializationMetadata cfg = archive.readMetadata( "config.json" );
                ActivationConfig config( TFn );
                config.fromMetadata( cfg );
                config.validate();

                auto inst = std::make_unique<Activation>( component_name, config );

                if ( exec_context )
                {
                    inst->setExecutionContext( exec_context );
                }

                return inst;
            }
            catch ( const std::exception& e )
            {
                throw std::runtime_error(
                    std::format( "Activation::fromArchive: error for '{}': {}", component_name, e.what() ) );
            }
        }

        // ====================================================================
        // Parameters and Gradients (stateless)
        // ====================================================================

        dim_t parameterCount() const override { return 0; }
        std::vector<ITensor*> getParameters() const override { return {}; }
        std::vector<ITensor*> getGradients() const override { return {}; }

        // ====================================================================
        // Identification and Description
        // ====================================================================

        /**
         * @brief Install a shared output slot (activation pooling).
         *
         * Must be called before build(): onBuilding then skips output self-allocation
         * after validating the slot's storage covers the build shape. forward() already
         * re-views on a shape change, so a wider slot never leaks its geometry to callers.
         * Mirrors Swiglu::installSharedOutput; self-allocation remains the default, and
         * the slot is owned and memory-accounted by the installer.
         */
        void installSharedOutput( std::shared_ptr<TensorType> output )
        {
            if ( this->isBuilt() )
                throw std::logic_error(
                    "Activation '" + this->getName() + "': installSharedOutput must be called before build()" );

            output_ = std::move( output );
            output_installed_ = true;
        }

        const ComponentType getType() const override
        {
            return ComponentType::Activation;
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            // An installed shared output slot is owned and counted by the installer.
            if ( output_ != nullptr && !output_installed_ )
            {
                stats.device_state_bytes += output_->getStorageSize();
            }
            if ( input_grad_ != nullptr )
            {
                stats.device_gradient_bytes += input_grad_->getStorageSize();
            }

            return stats;
        }

        /**
         * @brief What onBuilding() would allocate for this context, without allocating.
         *
         * Elementwise, so the output carries the input's shape exactly.
         *
         * See Specifications/MemoryFootprint.md.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            MemoryStats stats;

            // An installed shared output slot is owned and counted by the installer.
            if ( !output_installed_ && !context.hasInstalledOutput() )
            {
                stats.device_state_bytes +=
                    storageBytes<TPrecision>( elementCount( context.inputShape() ) );
            }

            if ( context.isTrainingMode() )
            {
                stats.device_gradient_bytes +=
                    storageBytes<TPrecision>( elementCount( context.inputShape() ) );
            }

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Activation: " << this->getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Function: " << activationTypeToString( TFn ) << std::endl;

            return oss.str();
        }

    protected:

        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const BuildContext& build_context ) override
        {
            operation_->build( build_context );
            const auto& input_shape = build_context.inputShape();

            DeviceId dev_id = this->getExecutionContext()->getDeviceId();

            if ( output_installed_ )
            {
                if ( !output_ || output_->size() < elementCount( input_shape ) )
                    throw std::invalid_argument(
                        "Activation '" + this->getName() + "': installed shared output slot is smaller than the build shape requires" );
            }
            else
            {
                output_ = std::make_shared<TensorType>( dev_id, input_shape, this->getName() + ".output" );
            }

            output_view_.emplace( output_->view( input_shape ) );

            if ( build_context.isTrainingMode() )
            {
                input_grad_ = std::make_unique<TensorType>( dev_id, input_shape, this->getName() + ".input_grad" );
                zero( *input_grad_ );
            }
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            operation_->setTrainingMode( training_mode );
        }

    private:

        using Functor = functor_of_t<TFn>;
        using OpType = typename OperationTraits<
            OperationType::ElementwiseActivationOp, TDeviceType, TPrecision, void>::template op_for<Functor>;

        ActivationConfig config_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };

        // Self-allocated at build, or an installed shared slot (installSharedOutput)
        // that the component views a prefix of.
        std::shared_ptr<TensorType> output_{ nullptr };
        bool output_installed_{ false };
        std::optional<TensorType> output_view_;
        std::unique_ptr<TensorType> input_grad_{ nullptr };

        static void validateMetadata_( const SerializationMetadata& meta, const std::string& component_name )
        {
            int64_t version = meta.tryGetInt( "version" ).value_or( 0 );
            if ( version != 1 )
            {
                throw std::runtime_error(
                    std::format( "Activation: unsupported version {} for '{}'", version, component_name ) );
            }

            std::string type = meta.tryGetString( "type" ).value_or( "" );
            if ( type != "Activation" )
            {
                throw std::runtime_error(
                    std::format( "Activation: type mismatch for '{}': expected 'Activation', got '{}'",
                        component_name, type ) );
            }

            std::string file_device = meta.tryGetString( "template_device" ).value_or( "" );
            int64_t file_precision = meta.tryGetInt( "template_precision" ).value_or( -1 );
            std::string file_function = meta.tryGetString( "template_function" ).value_or( "" );

            std::string expected_device = deviceTypeToString( TDeviceType );
            int64_t expected_precision = static_cast<int64_t>( TPrecision );
            std::string expected_function = activationTypeToString( TFn );

            if ( file_device != expected_device )
            {
                throw std::runtime_error(
                    std::format( "Activation: device mismatch for '{}': archive='{}', expected='{}'",
                        component_name, file_device, expected_device ) );
            }

            if ( file_precision != expected_precision )
            {
                throw std::runtime_error(
                    std::format( "Activation: precision mismatch for '{}': archive={}, expected={}",
                        component_name, file_precision, expected_precision ) );
            }

            if ( file_function != expected_function )
            {
                throw std::runtime_error(
                    std::format( "Activation: function mismatch for '{}': archive='{}', expected='{}'",
                        component_name, file_function, expected_function ) );
            }
        }

        void createOperation()
        {
            operation_ = std::make_shared<OpType>( this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error(
                    std::format( "Activation: failed to create compute backend operation for component '{}'",
                        this->getName() ) );
            }
        }
    };
}
