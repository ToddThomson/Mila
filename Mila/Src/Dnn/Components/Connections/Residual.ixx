/**
 * @file Residual.ixx
 * @brief Device-templated Residual connection component.
 *
 * The `Residual` component implements a residual shortcut y = x + F(x) with
 * configurable connection types (Addition, ScaledAddition, Gated) and optional
 * projection when input/output dimensions differ. Computation is delegated to
 * a device-specific binary operation backend obtained from the OperationRegistry.
 *
 * This implementation is device- and precision-parameterized and follows the
 * same component interface used by other layers (see `Component.ixx` and `Linear.ixx`).
 *
 * @tparam TDeviceType Compile-time device identifier (DeviceType::Cpu or DeviceType::Cuda).
 * @tparam TPrecision  Abstract tensor precision (TensorDataType).
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

export module Dnn.Components.Residual;
export import :Config;

import Dnn.Component;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Precision;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.UnaryOperation;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;
import Serialization.Mode;

import Dnn.Components.Linear;

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
         * @throws std::invalid_argument if config is invalid or device type mismatches.
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
         * @brief Return the total number of scalar parameters in this component.
         *
         * This includes gating/scaling parameters and any projection parameters.
         */
        size_t parameterCount() const override
        {
            return 0;
        }

        /**
         * @brief Execute the forward pass.
         *
         * Delegates to the backend binary operation. Inputs and outputs are
         * provided as abstract `ITensor` references to remain device-agnostic.
         *
         * @throws std::runtime_error if component has not been built or backend missing.
         */
        void forward( const ITensor& input_a, const ITensor& input_b, ITensor& output )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Residual::forward: component must be built before forward pass" );
            }

            if ( !operation_ )
            {
                throw std::runtime_error( "Residual::forward: operation backend not initialized" );
            }

            operation_->forward( input_a, input_b, output );
        }

        /**
         * @brief Execute the backward pass
         *
         * @throws std::runtime_error if backend not initialized.
         */
        void backward( 
            const ITensor& input_a, const ITensor& input_b,
            const ITensor& output_grad, 
            ITensor& input_a_grad, ITensor& input_b_grad )
        {
            if ( !operation_ )
            {
                throw std::runtime_error( "Residual::backward: operation backend not initialized" );
            }

            operation_->backward(
                input_a,
                input_b,
                output_grad,
                input_a_grad,
                input_b_grad
            );
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
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            // No-op placeholder; serialize parameter tensors if needed
            (void)archive;
            (void)mode;
        }

        std::vector<ITensor*> getParameters() const override
        {
            return {};
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
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
            oss << "Training mode: " << (this->isTraining() ? "true" : "false") << std::endl;
            oss << "Built: " << (this->isBuilt() ? "true" : "false") << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;

            return oss.str();
        }

    protected:

        // ====================================================================
        // Lifecycle hooks aligned with Component base
        // ====================================================================

        /**
         * @brief Hook invoked after ExecutionContext is set on the base Component.
         *
         * Create the device-specific BinaryOperation backend via the OperationRegistry.
         */
        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const shape_t& input_shape ) override
        {
            if ( !operation_ )
            {
                throw std::runtime_error( "Residual::onBuilding: operation backend not initialized. Ensure execution context was set." );
            }

            operation_->build( input_shape );

            input_shape_ = input_shape;
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Inform backend operation of the new training mode. When leaving
         * training, explicitly unbind any parameter-gradient pointers on the
         * backend to avoid accidental use or pinned memory.
         *
         * Called with Component's training mutex held; do not call setTraining() here.
         */
        void onTrainingChanging( bool is_training ) override
        {
            if ( operation_ )
            {
                operation_->setTraining( is_training );
            }
        }

    private:

        ResidualConfig config_;
        shape_t input_shape_;

        std::shared_ptr<BinaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

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
            operation_ = OperationRegistry::instance()
                .createBinaryOperation<TDeviceType, TPrecision>(
                    "ResidualOp", this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Residual: Failed to create compute backend operation." );
            }
        }
    };
}