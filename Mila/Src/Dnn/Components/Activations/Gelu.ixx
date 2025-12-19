/**
 * @file Gelu.ixx
 * @brief GELU activation component implementation.
 *
 * Device-templated GELU component that delegates computation to a registered
 * device-specific UnaryOperation backend.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <format>
#include <utility>
#include <optional>

export module Dnn.Components.Gelu;
export import :Config;

import Dnn.Component;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;
import Serialization.Tensor;
import Serialization.Mode;
import Serialization.Metadata;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Gaussian Error Linear Unit (GELU) activation component.
     *
     * Device-templated GELU component that performs forward (and optionally
     * backward) computation by delegating to a registered device-specific
     * UnaryOperation implementation found via the OperationRegistry.
     *
     * @tparam TDeviceType Compile-time device identifier (DeviceType::Cpu or DeviceType::Cuda).
     * @tparam TPrecision  Tensor data precision used by this component.
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
     * Preconditions:
     * - The component's `build(const shape_t&)` must be called before `forward()` to
     *   fully initialize the backend operation.
     * - Shared mode requires parent to call setExecutionContext() before build().
     *
     * Behavior:
     * - Stateless: no trainable parameters; `parameterCount()` returns 0 and
     *   `save()`/`load()` are minimal but include template metadata so a loader
     *   can validate instantiation parameters.
     * - `forward()` delegates to the backend UnaryOperation. The caller is
     *   responsible for providing device-compatible `ITensor` objects.
     * - `backward()` computes input gradients for GELU activation (no parameter gradients).
     *
     * Threading / Synchronization:
     * - Component does not guarantee thread-safety; call `synchronize()` to wait
     *   for outstanding device work to complete when needed.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Gelu : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using TensorType = Tensor<TPrecision, MR>;
        using ComponentBase = Component<TDeviceType, TPrecision>;

        /**
         * @brief Construct GELU activation component with optional ExecutionContext ownership.
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
         * - Use case: Components added to Network via addComponent<Gelu>(...).
         *
         * @param name Component name identifier (mandatory).
         * @param config GELU configuration (approximation method).
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
         * GeluConfig config;
         * Gelu<DeviceType::Cpu, TensorDataType::FP32> gelu("gelu", config, Device::Cpu());
         *
         * @example
         * // Shared mode (borrows parent's context)
         * Network<DeviceType::Cpu, TensorDataType::FP32> net(Device::Cpu(), "my_net");
         * net.addComponent<Gelu>("gelu", GeluConfig());
         */
        explicit Gelu( const std::string& name, const GeluConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "Gelu: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );

                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~Gelu() override = default;

        // ====================================================================
        // Computation
        // ====================================================================

        /**
         * @brief Run the forward computation for this GELU component.
         *
         * Delegates the computation to the device-specific UnaryOperation backend.
         * The caller must ensure `input` and `output` are allocated on the same
         * device as this component and that `build()` has been called.
         *
         * @param input Const reference to the input tensor (non-owning).
         * @param output Reference to the tensor that will receive results (caller-allocated).
         *
         * @throws std::runtime_error if component has not been built via build().
         * @throws std::runtime_error if operation backend is not initialized.
         */
        void forward( const ITensor& input, ITensor& output )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Gelu::forward: component must be built before forward pass" );
            }

            if ( !operation_ )
            {
                throw std::runtime_error( "Gelu::forward: operation backend not initialized" );
            }

            operation_->forward( input, output );
        }

        /**
         * @brief Compute gradients with respect to the component input.
         *
         * Delegates to the backend UnaryOperation::backward implementation to compute
         * the gradient of GELU with respect to the input. For GELU, there are no
         * trainable parameters, so this only computes input gradients.
         *
         * The gradient computation follows the chain rule:
         * dL/dinput = dL/doutput * dGELU(input)/dinput
         *
         * @param input Const reference to the original forward input.
         * @param output_grad Const reference to the gradient w.r.t. component output (?L/?output).
         * @param input_grad Reference to the tensor to be populated with the gradient
         *                   w.r.t. the component input (?L/?input, caller-allocated).
         *
         * @throws std::runtime_error if component has not been built via build().
         * @throws std::runtime_error if component is not in training mode.
         * @throws std::runtime_error if operation backend is not initialized.
         *
         * @note GELU has no parameters, so no parameter gradients are computed.
         * @note The implementation accumulates into input_grad (does not overwrite).
         * @note Requires setTraining(true) for gradient computation in some backends.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Gelu::backward: component must be built before backward pass" );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Gelu::backward: component must be in training mode to compute gradients" );
            }

            if ( !operation_ )
            {
                throw std::runtime_error( "Gelu::backward: operation backend not initialized" );
            }

            operation_->backward( input, output_grad, input_grad );
        }

        /**
         * @brief Wait for all asynchronous work submitted by this component to complete.
         *
         * Synchronizes the underlying ExecutionContext. On CPU implementations this
         * may be a no-op. Use to ensure results are visible on the host or to measure
         * synchronous timings.
         *
         * @throws std::runtime_error if ExecutionContext has not been set.
         */
        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        /**
         * @brief Return the configured GELU approximation method.
         *
         * @return Configured GeluConfig::ApproximationMethod value (Exact or Tanh).
         */
        GeluConfig::ApproximationMethod getApproximationMethod() const
        {
            return config_.getApproximationMethod();
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Persist component state to archive.
         *
         * GELU is stateless (no trainable tensors) but persists:
         * - Component type ("Gelu") and version (1)
         * - Component name from config
         * - Template parameters (device type and precision) for loader validation
         * - Serialized GeluConfig (approximation method)
         *
         * Files written:
         * - meta.json: Component metadata and template parameters
         * - config.json: GeluConfig serialization
         *
         * @param archive Archive to write to.
         * @param mode Serialization mode (currently unused, all state is always saved).
         */
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)mode;

            SerializationMetadata meta;
            meta.set( "type", "Gelu" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() )
                .set( "template_device", deviceTypeToString( TDeviceType ) )
                .set( "template_precision", static_cast<int64_t>(TPrecision) );

            archive.writeMetadata( "meta.json", meta );

            SerializationMetadata cfg;
            cfg.set( "approximation_method",
                static_cast<int64_t>(config_.getApproximationMethod()) );

            archive.writeMetadata( "config.json", cfg );
        }

        /**
         * @internal
         * @brief Factory method to reconstruct Gelu component from archive.
         *
         * Called by ComponentFactory during deserialization. Reconstructs a Gelu
         * instance in shared mode (using provided ExecutionContext from parent).
         *
         * Reads and validates:
         * - meta.json: Component type, version, template parameters
         * - config.json: GeluConfig
         *
         * @param archive Archive to read from.
         * @param component_name Name of the component in the archive (for diagnostics).
         * @param exec_context Execution context for the new component (shared from parent).
         * @return Unique pointer to reconstructed Gelu component.
         *
         * @throws std::runtime_error if version mismatch or type mismatch.
         * @throws std::runtime_error if device type or precision mismatch.
         * @throws std::runtime_error if metadata parsing fails.
         */
        static std::unique_ptr<Gelu> fromArchive_(
            ModelArchive& archive,
            const std::string& component_name,
            IExecutionContext* exec_context )
        {
            try
            {
                SerializationMetadata meta = archive.readMetadata( "meta.json" );
                validateMetadata_( meta, component_name );

                SerializationMetadata cfg = archive.readMetadata( "config.json" );

                GeluConfig config;
                auto approx_method = static_cast<GeluConfig::ApproximationMethod>(
                    cfg.getInt( "approximation_method" ));
                config.withApproximationMethod( approx_method );
                config.validate();

                return std::make_unique<Gelu>( component_name, config );
            }
            catch ( const std::exception& e )
            {
                throw std::runtime_error(
                    std::format( "Gelu::fromArchive: error for '{}': {}",
                        component_name, e.what() )
                );
            }
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        /**
         * @brief Number of trainable parameters.
         *
         * GELU is stateless and exposes no trainable parameters.
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
         * GELU has no trainable parameters.
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
         * GELU has no trainable parameters, therefore no gradients.
         *
         * @return Empty vector.
         */
        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        // ====================================================================
        // State and Configuration
        // ====================================================================

        /**
         * @brief Get the device identifier for this component.
         *
         * Returns the DeviceId from the ExecutionContext. In standalone mode,
         * this is the device specified at construction. In shared mode, this
         * is the parent's device.
         *
         * @return DeviceId indicating device type and index.
         *
         * @throws std::runtime_error if ExecutionContext has not been set.
         */
        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        /**
         * @brief Generate human-readable description of the component.
         *
         * Produces a multi-line string showing:
         * - Component name
         * - Device type
         * - Approximation method
         *
         * @return Formatted string representation.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Gelu: " << this->getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Approximation Method: " << config_.toString( config_.getApproximationMethod() ) << std::endl;

            return oss.str();
        }

    protected:

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
         * State guards:
         * - Expects ExecutionContext to be set (guaranteed by Component::setExecutionContext)
         * - Can be called before build() (operation creation is independent of shape)
         *
         * @throws std::runtime_error if operation creation fails.
         * @throws std::runtime_error if "GeluOp" is not registered for this device/precision.
         */
        void onExecutionContextSet() override
        {
            createOperation();
        }

        /**
         * @brief Hook invoked during build() to initialize backend operation.
         *
         * Delegates shape-dependent initialization to the backend UnaryOperation.
         * Must be called before `forward()` or `backward()`.
         *
         * State guards:
         * - Expects ExecutionContext to be set (required for operation creation)
         * - Expects operation to be initialized (created in onExecutionContextSet)
         * - Expects component to be unbuilt (guaranteed by Component::build)
         *
         * @param input_shape Expected shape for input tensors.
         *
         * @throws std::invalid_argument if input_shape is incompatible with the component configuration.
         * @throws std::runtime_error if backend allocation or build fails.
         * @throws std::runtime_error if operation is not initialized.
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            if ( !operation_ )
            {
                throw std::runtime_error(
                    std::format( "Gelu::onBuilding: operation backend not initialized for component '{}'. "
                        "Ensure onExecutionContextSet() was called successfully.",
                        this->getName() )
                );
            }

            operation_->build( input_shape );
            input_shape_ = input_shape;
        }

        /**
         * @brief Hook invoked when training mode changes.
         *
         * Propagates training mode to the backend operation. Called by
         * Component::setTraining() with the training mutex held.
         *
         * State guards:
         * - Expects operation to be initialized (should be created in onExecutionContextSet)
         * - Can be called before or after build()
         *
         * @param is_training New training mode state.
         *
         * @note Do not call setTraining() from this hook (reentrancy prohibited).
         * @note If operation is not initialized, silently returns (may occur during construction).
         */
        void onTrainingChanging( bool is_training ) override
        {
            if ( operation_ )
            {
                operation_->setTraining( is_training );
            }
        }

    private:

        GeluConfig config_;
        shape_t input_shape_;
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        /**
         * @brief Validate metadata from archive during deserialization.
         *
         * Verifies:
         * - Version is 1
         * - Type is "Gelu"
         * - Device type matches TDeviceType
         * - Precision matches TPrecision
         *
         * @param meta Parsed metadata.
         * @param component_name Component name for error messages.
         *
         * @throws std::runtime_error if validation fails.
         */
        static void validateMetadata_( const SerializationMetadata& meta, const std::string& component_name )
        {
            int64_t version = meta.tryGetInt( "version" ).value_or( 0 );
            if ( version != 1 )
            {
                throw std::runtime_error(
                    std::format( "Gelu: unsupported version {} for '{}'",
                        version, component_name )
                );
            }

            std::string type = meta.tryGetString( "type" ).value_or( "" );
            if ( type != "Gelu" )
            {
                throw std::runtime_error(
                    std::format( "Gelu: type mismatch for '{}': expected 'Gelu', got '{}'",
                        component_name, type )
                );
            }

            std::string file_device = meta.tryGetString( "template_device" ).value_or( "" );
            int64_t file_precision = meta.tryGetInt( "template_precision" ).value_or( -1 );

            std::string expected_device = deviceTypeToString( TDeviceType );
            int64_t expected_precision = static_cast<int64_t>(TPrecision);

            if ( file_device != expected_device )
            {
                throw std::runtime_error(
                    std::format( "Gelu: device mismatch for '{}': archive='{}', expected='{}'",
                        component_name, file_device, expected_device )
                );
            }

            if ( file_precision != expected_precision )
            {
                throw std::runtime_error(
                    std::format( "Gelu: precision mismatch for '{}': archive={}, expected={}",
                        component_name, file_precision, expected_precision )
                );
            }
        }

        /**
         * @brief Create backend UnaryOperation from OperationRegistry.
         *
         * Called by onExecutionContextSet() hook. Looks up "GeluOp" in the
         * OperationRegistry and creates a device-specific implementation.
         *
         * @throws std::runtime_error if operation creation fails.
         * @throws std::runtime_error if "GeluOp" is not registered for this device/precision.
         */
        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "GeluOp", this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error(
                    std::format( "Gelu: Failed to create compute backend operation for component '{}'",
                        this->getName() )
                );
            }
        }
    };
}