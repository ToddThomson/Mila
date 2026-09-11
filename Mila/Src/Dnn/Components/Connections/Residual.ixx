/**
 * @file Residual.ixx
 * @brief Device-templated Residual connection component.
 *
 * The `Residual` component implements a residual shortcut y = x + F(x) with
 * configurable connection types (Addition, ScaledAddition, Gated) and optional
 * projection when input/output dimensions differ. Computation is delegated to
 * a device-specific binary operation backend obtained from the OperationRegistry.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <optional>
#include <format>
#include <algorithm>

export module Dnn.Components.Residual;
export import Dnn.Components.ResidualConfig;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.OperationTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.Observation;
import Serialization.ModelArchive;
import Serialization.Mode;

import Dnn.Components.Linear;
import Logging.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Device-templated Residual connection component.
     *
     * Delegates binary residual computation to a device-specific backend
     * operation. Parameters (if any) and any projection tensors are stored as
     * `Tensor` instances bound to the associated execution context.
     *
     * New API:
     * - `forward(...)` returns pointer to a component-owned output `ITensor`
     * - `backward(...)` returns pointer to a component-owned input-gradient for the first input.
     *   The component also owns the gradient for the second input which can be
     *   accessed via `getInputBGrad()`.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda).
     * @tparam TPrecision  Abstract tensor precision (TensorDataType).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Residual : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using ComponentBase = Component<TDeviceType, TPrecision>;

        /**
         * @brief Construct Residual component with optional ExecutionContext ownership.
         *
         * Supports two construction modes:
         * - Standalone mode (device_id provided): creates and owns an ExecutionContext.
         * - Shared mode (no device_id): parent must call setExecutionContext() prior to build().
         *
         * @param name Component name identifier (mandatory).
         * @param config Residual configuration.
         * @param device_id Optional device identifier to create owned ExecutionContext.
         *
         * @throws std::invalid_argument if build_config is invalid or device type mismatches.
         * @throws std::runtime_error if ExecutionContext creation fails (standalone mode).
         */
        explicit Residual( const std::string& name, const ResidualConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "Residual: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );

                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~Residual() override = default;

        /**
         * @brief Execute the forward pass and return component-owned output tensor.
         *
         * The returned pointer is owned by the component and is valid until the
         * component is destroyed or rebuilt. The backend BinaryOperation signature
         * is unchanged; the component provides the owned output tensor when
         * invoking the backend.
         *
         * @param input_a Left input tensor.
         * @param input_b Right input tensor.
         * @return Pointer to component-owned ITensor containing the forward result.
         *
         * @throws std::runtime_error if component has not been built or backend missing.
         */
        TensorType& forward( const TensorType& input_a, const TensorType& input_b )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Residual::forward: component must be built before forward pass" );
            }

            operation_->forward( input_a, input_b, *output_ );

            // DEBUG: 
            //this->synchronize();

            auto input_shape = input_a.shape();

            // With an installed shared slot the raw output_ may be wider than the
            // result; always return the shape-adjusted view in that case.
            if ( !output_installed_ && input_shape == leading_shape_ )
            {
                this->publish( ComputePass::Forward, "output", *output_ );

                return *output_;
            }

            output_view_ = std::make_unique<TensorType>( output_->view( input_shape ) );

            this->publish( ComputePass::Forward, "output", *output_view_ );

            return *output_view_;
        }

        /**
         * @brief Execute the backward pass and return component-owned gradient for input_a.
         *
         * Component owns both input gradients (for input_a and input_b). The method
         * returns the gradient tensor for `input_a`. The gradient for `input_b` can
         * be accessed via `getInputBGrad()` after calling `backward()`.
         *
         * @param input_a Left forward input.
         * @param input_b Right forward input.
         * @param output_grad Gradient with respect to the component output.
         * @return Pointer to component-owned ITensor containing gradient w.r.t. input_a.
         *
         * @throws std::runtime_error if backend not initialized or if component not built/training.
         */
        std::pair<TensorType&, TensorType&> backward(
            const TensorType& input_a,
            const TensorType& input_b,
            const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Residual::backward: component must be built before backward pass" );
            }

            if ( this->isInferenceMode() )
            {
                throw std::runtime_error( "Residual::backward: component must be in training mode to compute gradients" );
            }

            // Zero BOTH owned input gradient buffers before backward pass.
            //
            // The reason is CpuResidualOp, which accumulates (dX1[i] += dY[i]) and so
            // needs a clean buffer; CudaResidualOp assigns (dA[idx] = grad) and does not.
            // It is NOT to combine gradients arriving from several paths: each of these
            // buffers has exactly one producer, and where the residual stream forks the
            // summation is explicit in the owning block -- see GptBlock::backward, which
            // does `zero(d_res1_accum_); add(d_res1_from_res2, d_res1_from_ln2, ...)`.
            //
            // Assign and accumulate-into-zero are therefore equivalent here, which is why
            // the two backends disagree without any observable difference. Reconciling
            // them (and dropping the zeroing for a full-overwrite op) is an API Coherence
            // question parked in BACKLOG, not a defect.
            zero( *input_a_grad_ /*, this->getExecutionContext() */);
            zero( *input_b_grad_ /*, this->getExecutionContext() */);

            operation_->backward(
                input_a,
                input_b,
                output_grad,
                *input_a_grad_,
                *input_b_grad_ );

            return { *input_a_grad_, *input_b_grad_ };
        }

        /**
         * @brief Block until all device operations submitted by this component complete.
         *
         * @throws std::runtime_error if ExecutionContext has not been set.
         */
        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        /**
         * @brief Serialize component parameters into the provided archive.
         *
         * Placeholder; concrete implementations should write named parameter
         * tensors into the archive.
         */
        void save_( ModelArchive&, SerializationMode ) const override
        {
            // Deliberately empty: parameterCount() is 0, so there is nothing to serialize.
        }

        // ====================================================================
        // Identification and Description
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::Residual;
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        /**
         * @brief Return a human-readable description of the component.
         *
         * Includes configured name, training/built state, backend presence,
         * device information and parameter count to aid debugging and logging.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Residual: " << this->getName() << std::endl;
            oss << "Training mode: " << (this->isTrainingMode() ? "true" : "false") << std::endl;
            oss << "Built: " << (this->isBuilt() ? "true" : "false") << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;

            return oss.str();
        }

        /**
         * @brief Number of trainable parameters.
         *
         * Residual has no trainable parameters.
         *
         * @return 0
         */
        dim_t parameterCount() const override
        {
            return 0;
        }

        /**
         * @brief Return non-owning pointers to parameter tensors.
         *
         * Residual has no trainable parameter tensors by default; return empty list.
         */
        std::vector<ITensor*> getParameters() const override
        {
            return {};
        }

        /**
         * @brief Return non-owning pointers to parameter gradient tensors.
         *
         * Residual has no trainable parameters by default; return empty list.
         */
        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        /**
         * @brief Install a shared output slot (activation pooling).
         *
         * Must be called before build(): onBuilding then skips output self-allocation
         * after validating the slot's storage covers the build shape, and forward()
         * always returns a shape-adjusted view so a wider slot never leaks its
         * geometry to callers. Mirrors Linear::installSharedWeight; self-allocation
         * remains the default. The slot is owned and memory-accounted by the installer.
         */
        void installSharedOutput( std::shared_ptr<TensorType> output )
        {
            if ( this->isBuilt() )
                throw std::logic_error(
                    "Residual '" + this->getName() + "': installSharedOutput must be called before build()" );

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

            // An installed shared output slot is owned and counted by the installer -- so
            // getOutputs() reports a tensor this does not, which is correct for both: one
            // asks what is produced, the other what is owned.
            if ( output_ != nullptr && !output_installed_ )
            {
                stats.device_state_bytes += output_->getStorageSize();
            }

            if ( input_a_grad_ != nullptr )
            {
                stats.device_gradient_bytes += input_a_grad_->getStorageSize();
            }

            if ( input_b_grad_ != nullptr )
            {
                stats.device_gradient_bytes += input_b_grad_->getStorageSize();
            }

            return stats;
        }

        /**
         * @brief What onBuilding() would allocate for this context, without allocating.
         *
         * See Specifications/MemoryFootprint.md.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            const auto& input_shape = context.inputShape();
            const dim_t elements = elementCount( input_shape );

            MemoryStats stats;

            // An installed shared output slot is owned and counted by the installer.
            if ( !output_installed_ && !context.hasInstalledOutput() )
            {
                stats.device_state_bytes += storageBytes<TPrecision>( elements );
            }

            if ( operation_ )
            {
                stats.device_state_bytes += operation_->getRequiredStateMemorySize( context );
            }

            if ( context.isTrainingMode() )
            {
                // Two inputs, two input gradients.
                stats.device_gradient_bytes += 2 * storageBytes<TPrecision>( elements );
            }

            return stats;
        }

    protected:

        /**
         * @brief Hook invoked after ExecutionContext is set on the base Component.
         *
         * Create the device-specific BinaryOperation backend via the OperationRegistry.
         */
        void onExecutionContextSet() override
        {
            createOperation();
        }

        /**
         * @brief Build the Residual component from the provided BuildContext.
         *
         * Allocates the component-owned output buffer and, for Training-mode
         * builds, the gradient buffers for both inputs.
         *
         * ## Output buffer
         *
         * The output buffer matches the full input shape. Residual is a
         * pure elementwise addition with no sequence dimension concern --
         * RuntimeMode does not influence output buffer allocation.
         *
         * ## Gradient buffers
         *
         * input_a_grad_ and input_b_grad_ are allocated only for
         * RuntimeMode::Training builds. Both are the same shape as the
         * input -- Residual is elementwise addition and both inputs are
         * always symmetric in shape.
         *
         * @param build_config Full input shape and RuntimeMode for this build.
         */
        void onBuilding( const BuildContext& build_config ) override
        {
            const auto& input_shape = build_config.inputShape();

            operation_->build( build_config );

            auto device = this->getExecutionContext()->getDeviceId();

            if ( output_installed_ )
            {
                dim_t needed = 1;
                for ( auto d : input_shape )
                    needed *= d;

                if ( !output_ || output_->size() < needed )
                    throw std::invalid_argument(
                        "Residual '" + this->getName() + "': installed shared output slot is smaller than the build shape requires" );
            }
            else
            {
                output_ = std::make_shared<TensorType>(
                    device, input_shape, this->getName() + ".output" );
            }

            if ( build_config.isTrainingMode() )
            {
                input_a_grad_ = std::make_unique<TensorType>(
                    device, input_shape, this->getName() + ".input_a.grad" );
                zero( *input_a_grad_ );

                input_b_grad_ = std::make_unique<TensorType>(
                    device, input_shape, this->getName() + ".input_b.grad" );
                zero( *input_b_grad_ );
            }
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Inform backend operation of the new training mode. When leaving
         * training, explicitly unbind any parameter-gradient pointers on the
         * backend to avoid accidental use or pinned memory.
         *
         * Called with Component's training mutex held; do not call setTrainingMode() here.
         */
        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            operation_->setTrainingMode( training_mode );
        }

    private:

        using OpType = typename OperationTraits<OperationType::ResidualOp, TDeviceType, TPrecision>::type;

        ResidualConfig config_;
        shape_t leading_shape_;

        std::shared_ptr<OpType> operation_{ nullptr };
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        // Self-allocated at build, or an installed shared slot (installSharedOutput)
        // that the component views a prefix of.
        std::shared_ptr<TensorType> output_{ nullptr };
        bool output_installed_{ false };
        std::unique_ptr<TensorType> output_view_{ nullptr };
        std::unique_ptr<TensorType> input_a_grad_{ nullptr };
        std::unique_ptr<TensorType> input_b_grad_{ nullptr };

        /**
         * @brief Create backend BinaryOperation from OperationRegistry.
         *
         * Called by onExecutionContextSet(). Looks up "ResidualOp" in the
         * OperationRegistry and creates a device-specific implementation.
         *
         * @throws std::runtime_error if operation creation fails.
         */
        void createOperation()
        {
            operation_ = std::make_shared<OpType>( this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Residual: Failed to create compute backend operation." );
            }
        }
    };
}
