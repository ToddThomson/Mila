/**
 * @file Softmax.ixx
 * @brief Device-templated Softmax activation module.
 *
 * Delegates compute to a UnaryOperation backend. Module is stateless (no trainable
 * parameters) and exposes configuration to callers.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <optional>

export module Dnn.Components.Softmax;
export import :Config;

import Dnn.Component;
import Dnn.ComponentType;
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
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Softmax activation module (device-templated).
     *
     * Delegates computation to a device-specific UnaryOperation implementation
     * registered in the OperationRegistry.
     *
     * Softmax is a stateless activation function with no trainable parameters.
     * The operation computes: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
     * across a specified axis.
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
    class Softmax : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * @brief Construct Softmax with optional ExecutionContext ownership.
         *
         * Supports two construction modes:
         *
         * **Standalone mode (device_id provided)**:
         * - Creates and owns an ExecutionContext for the specified device.
         * - Registers the owned context with the base Component class via setExecutionContext().
         * - Backend operation is created immediately in onExecutionContextSet() hook.
         * - Use case: Unit tests, standalone component usage.
         *
         * **Shared mode (device_id not provided)**:
         * - Does not create ExecutionContext; expects parent to provide one.
         * - Parent (Network/CompositeComponent) calls setExecutionContext() after construction.
         * - Backend operation created when parent sets context.
         * - Use case: Components added to Network via addComponent<Softmax>(...).
         *
         * @param config Softmax configuration (axis and name).
         * @param device_id Optional device identifier. If provided, creates owned ExecutionContext
         *                  for standalone mode. If nullopt, expects shared context from parent.
         *
         * @throws std::invalid_argument if config is invalid (via config.validate()).
         * @throws std::invalid_argument if device_id.type does not match TDeviceType.
         * @throws std::runtime_error if ExecutionContext creation fails (standalone mode).
         * @throws std::runtime_error if backend operation creation fails in onExecutionContextSet().
         *
         * @note In standalone mode, setExecutionContext() is called to register the owned
         *       context with the base class, enabling getExecutionContext() and triggering
         *       the onExecutionContextSet() hook for operation creation.
         *
         * @example
         * // Standalone mode (owns context)
         * SoftmaxConfig config;
         * config.withAxis(-1);
         * Softmax<DeviceType::Cpu, TensorDataType::FP32> softmax(config, Device::Cpu());
         *
         * @example
         * // Shared mode (borrows parent's context)
         * Network<DeviceType::Cpu, TensorDataType::FP32> net(Device::Cpu(), "my_net");
         * net.addComponent<Softmax>("softmax", SoftmaxConfig().withAxis(-1));
         */
        explicit Softmax( const std::string& name, const SoftmaxConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if (device_id.has_value())
            {
                if (device_id->type != TDeviceType)
                {
                    throw std::invalid_argument( "Softmax: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );

                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~Softmax() override = default;

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - delegates to backend operation.
         *
         * Computes softmax activation across the configured axis.
         */
        void forward( const ITensor& input, ITensor& output )
        {
            if (!this->isBuilt())
            {
                throw std::runtime_error( "Softmax module must be built before calling forward." );
            }

            operation_->forward( input, output );
        }

        /**
         * @brief Backward pass - delegates to backend operation.
         *
         * Computes gradient: dX = Y * (dY - dot(Y, dY))
         * where Y is the softmax output.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if (!this->isBuilt())
            {
                throw std::runtime_error( "Softmax module must be built before calling backward." );
            }

            if (!this->isTraining())
            {
                throw std::runtime_error( "Softmax module must be in training mode to call backward. Call setTraining(true) first." );
            }

            operation_->backward( input, output_grad, input_grad );
        }

        // ====================================================================
        // Synchronization
        // ====================================================================

        /**
         * @brief Wait for all asynchronous work submitted by this module to complete.
         *
         * Synchronizes the underlying ExecutionContext. On CPU implementations this 
         * may be a no-op. Use to ensure results are visible on the host or to measure 
         * synchronous timings.
         */
        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Persist module state to archive.
         *
         * Softmax is stateless (no trainable tensors) but persists:
         * - Module type and version metadata
         * - Configuration (axis)
         *
         * @param archive Archive to write to.
         * @param mode Serialization mode (currently unused for stateless components).
         */
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)archive;
            (void)mode;
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        /**
         * @brief Number of trainable parameters.
         *
         * Softmax is stateless and exposes no trainable parameters.
         *
         * @return 0
         */
        size_t parameterCount() const override
        {
            return 0;
        }

        /**
         * @brief Get trainable parameter tensors.
         *
         * Softmax has no trainable parameters.
         *
         * @return Empty vector.
         */
        std::vector<ITensor*> getParameters() const override
        {
            return {};
        }

        /**
         * @brief Get parameter gradient tensors.
         *
         * Softmax has no trainable parameters, therefore no gradients.
         *
         * @return Empty vector.
         */
        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::Softmax;
        }

        /**
         * @brief Get the device identifier for this module.
         *
         * Returns the DeviceId from the ExecutionContext. In standalone mode,
         * this is the device specified at construction. In shared mode, this
         * is the parent's device.
         *
         * @return DeviceId indicating device type and index.
         */
        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        /**
         * @brief Generate human-readable description of the module.
         *
         * Produces a multi-line string showing:
         * - Module name
         * - Device type
         * - Axis configuration
         *
         * @return Formatted string representation.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Softmax: " << this->getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Axis: " << config_.getAxis() << std::endl;
            oss << "Parameter count: 0 (stateless)" << std::endl;

            return oss.str();
        }

        // ====================================================================
        // Configuration accessors
        // ====================================================================

        /**
         * @brief Get the softmax axis.
         *
         * @return The axis along which softmax is computed.
         */
        int64_t getAxis() const noexcept
        {
            return config_.getAxis();
        }

        /**
         * @brief Get the configuration.
         *
         * @return Reference to the SoftmaxConfig.
         */
        /*const SoftmaxConfig& getConfig() const noexcept
        {
            return config_;
        }*/

    protected:

        // ====================================================================
        // Lifecycle hooks
        // ====================================================================

        /**
         * @brief Hook invoked after ExecutionContext is set.
         *
         * Called by Component::setExecutionContext() after the context is
         * registered. Creates the backend UnaryOperation using the OperationRegistry.
         *
         * This hook is triggered in two scenarios:
         * - Standalone mode: Immediately in constructor after owned context creation
         * - Shared mode: When parent calls setExecutionContext() after construction
         *
         * @throws std::runtime_error if operation creation fails.
         */
        void onExecutionContextSet() override
        {
            createOperation();
        }

        /**
         * @brief Hook invoked during build() to initialize component with input shape.
         *
         * Softmax is stateless and has no parameters to allocate. This method
         * validates the input shape and delegates to the backend operation's
         * build method to cache dimension computations.
         *
         * @param input_shape Expected shape for input tensors.
         *
         * @throws std::invalid_argument if input_shape is invalid or axis out of bounds.
         * @throws std::runtime_error if backend build fails.
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            operation_->setParameters( nullptr, nullptr );

            operation_->build( input_shape );
        }

        /**
         * @brief Hook invoked when training mode changes.
         *
         * Propagates training mode to the backend operation. Called by
         * Component::setTraining() with the training mutex held.
         *
         * @param is_training New training mode state.
         *
         * @note Do not call setTraining() from this hook (reentrancy prohibited).
         */
        void onTrainingChanging( bool is_training ) override
        {
            operation_->setTraining( is_training );
        }

    private:
        SoftmaxConfig config_;
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        /**
         * @brief Validate input shape for softmax operation.
         *
         * Ensures the input has valid rank and the configured axis is within bounds.
         */
        void validateInputShape( const ITensor& input ) const
        {
            const auto& input_shape = input.shape();
            validateInputShape( input_shape );
        }

        /**
         * @brief Validate input shape for softmax operation.
         *
         * Ensures the input has valid rank and the configured axis is within bounds.
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.empty())
            {
                throw std::invalid_argument( "Softmax: input must have rank >= 1" );
            }

            int64_t axis = config_.getAxis();
            const int64_t ndim = static_cast<int64_t>(input_shape.size());

            if (axis < 0)
            {
                axis = ndim + axis;
            }

            if (axis < 0 || axis >= ndim)
            {
                throw std::invalid_argument( "Softmax: axis out of bounds for input shape" );
            }
        }

        /**
         * @brief Create the backend compute operation.
         *
         * Uses the shared ExecutionContext from the base class to request a
         * device-specific UnaryOperation from the OperationRegistry.
         */
        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "SoftmaxOp",
                    this->getExecutionContext(),
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Failed to create Softmax compute backend operation." );
            }
        }
    };
}