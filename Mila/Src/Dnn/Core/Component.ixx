/**
 * @file Component.ixx
 * @brief Base component interface for Mila DNN components.
 *
 * Provides the abstract `Component` template defining the shared lifecycle,
 * parameter, execution mode, and introspection APIs used by all Mila modules.
 */

module;
#include <string>
#include <memory>
#include <ostream>
#include <vector>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <format>

export module Dnn.Component;
export import :BuildConfig;
export import :MemoryStats;

import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Serialization.Tensor;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Abstract base class for neural network components.
     *
     * Component enforces a single ownership model: all components receive a
     * non-owning pointer to an IExecutionContext that is owned by the parent
     * (Network or test fixture). This ensures consistent resource sharing and
     * eliminates dual constructor patterns.
     *
     * ## Ownership model
     *
     * - Components NEVER own ExecutionContext.
     * - Parent (Network/CompositeComponent) owns and provides shared context.
     * - Tests explicitly create context and pass raw pointer to components.
     *
     * ## Component build lifecycle
     *
     * Components progress through a well-defined lifecycle. Each stage has
     * a single responsibility and a designated hook for subclass extension.
     *
     * ### Stage 1 — Construction
     *
     * The component is constructed with its name and component config.
     * No device resources are allocated. No ExecutionContext is required.
     *
     * @code
     *   auto linear = std::make_unique<Linear>( "fc", config, context );
     * @endcode
     *
     * ### Stage 2 — Build  [ onBuilding() ]
     *
     * build() is called with a BuildConfig carrying the leading shape
     * { B, T, ... } and the ExecutionMode that governs buffer allocation.
     *
     * | Mode      | allocationSeqLen() | Gradient buffers    |
     * |-----------|--------------------|---------------------|
     * | Inference | 1                  | never allocated     |
     * | Training  | leading_shape[1]   | allocated on demand |
     *
     * Allocated in onBuilding():
     *   - owned_output_         forward output buffer sized by allocationSeqLen()
     *   - owned_decode_output_  decode output buffer (decode-capable components)
     *   - kv_cache_             KV cache (decode-capable components)
     *   - operation_ buffers    via operation_->build()
     *
     * NOT allocated in onBuilding():
     *   - gradient buffers      (deferred to first setEvaluation( false ))
     *   - backward state        (deferred to first setEvaluation( false ))
     *
     * The same BuildConfig is cascaded unchanged through CompositeComponent
     * and Network to all child components.
     *
     * @code
     *   BuildConfig config( shape_t{ batch_size, seq_length } );
     *   config.withExecutionMode( ExecutionMode::Training );
     *   model->build( config );
     * @endcode
     *
     * ### Stage 3 — Evaluation mode  [ onEvaluationChanging() ]
     *
     * Only valid for Training-built components. setEvaluation( false ) triggers
     * gradient buffer allocation on the first call. Subsequent calls zero
     * existing buffers without reallocating. setEvaluation( true ) zeros
     * gradient buffers and disables the backward path.
     *
     * Allocated on first setEvaluation( false ):
     *   - owned_input_grad_     input gradient buffer
     *   - weight_grad_          weight gradient buffer
     *   - bias_grad_            bias gradient buffer
     *
     * @code
     *   model->setEvaluation( true );    // suspend backward — eval checkpoint
     *   generateSample( model );
     *   model->setEvaluation( false );   // resume training
     * @endcode
     *
     * ### Stage 4 — Forward / Decode / Backward
     *
     * Runtime dimensions are read from the input tensor shape on each call.
     * No shape information is cached from build time beyond what is in
     * build_config_.
     *
     * ### Lifecycle invariants
     *
     *   build()                  requires ExecutionContext to be set
     *   setEvaluation()          requires build() to have completed
     *   setEvaluation()          requires ExecutionMode::Training
     *   forward()                requires build() to have completed
     *   backward()               requires isTraining() == true
     *   decode()                 requires build() to have completed
     *
     * ## Base class provides
     *
     * - build() / lifecycle management with protected onBuilding() hook
     * - execution mode query: isInference(), isTraining(), isEvaluating()
     * - evaluation mode transitions with serialized onEvaluationChanging() hook
     * - parameter and gradient access
     * - synchronization and serialization
     * - short human-readable diagnostics via toString()
     *
     * @tparam TDeviceType Compile-time device identifier for this component.
     * @tparam TPrecision  Tensor data precision for this component.
     *
     * @par Trusted Collaborators
     * This class grants private access to:
     * - CompositeComponent: Parent components that manage child execution
     *   contexts and aggregate parameters from child components.
     * - Network: Top-level graph coordinator that collects parameters and
     *   gradients for optimization and serialization.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Component
    {
        template<DeviceType, TensorDataType>
        friend class CompositeComponent;

        template<DeviceType, TensorDataType>
        friend class Network;

    public:

        /**
         * @brief Construct component with required name identifier.
         *
         * The name is used for identification, logging, and serialization.
         * Names must be valid identifiers: start with a letter, contain only
         * letters, digits, '.', '_', '-', and be 1-128 characters long.
         *
         * @param name Component name identifier (mandatory).
         *
         * @throws std::invalid_argument if name is not a valid identifier.
         */
        explicit Component( const std::string& name )
            : name_( validateName( name ) ), built_( false )
        {}

        virtual ~Component() = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Build the component with a raw leading shape (convenience overload).
         *
         * Constructs a BuildConfig with the default ExecutionMode::Inference
         * and delegates to the canonical build( const BuildConfig& ).
         *
         * Prefer the BuildConfig overload when the execution mode must be
         * explicitly set (e.g. Training).
         *
         * @param leading_shape The leading dimensions { B, T, ... } used as
         *                      allocation bounds for all component buffers.
         *
         * @throws std::runtime_error    if the component is already built.
         * @throws std::runtime_error    if no ExecutionContext has been set.
         * @throws std::invalid_argument if leading_shape is invalid.
         * @throws Any exception from onBuilding().
         */
        virtual void build( const shape_t& leading_shape ) final
        {
            BuildConfig config( leading_shape );
            build( config );
        }

        /**
         * @brief Build the component with the provided BuildConfig (canonical overload).
         *
         * Validates the config, stores it as build_config_, then invokes the
         * onBuilding() hook for component-specific buffer allocation and
         * initialization.
         *
         * After onBuilding() returns without throwing, the component is marked
         * built and isBuilt() returns true. If onBuilding() throws, built_
         * remains false and build() may be retried — but only if the
         * onBuilding() implementation leaves component state coherent on failure.
         *
         * The stored BuildConfig is accessible to derived classes via the
         * protected build_config_ member throughout the component lifetime.
         *
         * @param config Build-time configuration carrying the leading shape
         *               { B, T, ... }, ExecutionMode, and optional
         *               micro-batching settings.
         *
         * @throws std::runtime_error    if the component is already built.
         * @throws std::runtime_error    if no ExecutionContext has been set.
         * @throws std::invalid_argument if config.validate() fails.
         * @throws Any exception from onBuilding().
         */
        virtual void build( const BuildConfig& config ) final
        {
            if ( isBuilt() )
            {
                throw std::runtime_error( "Component already built. Cannot rebuild." );
            }

            if ( !hasExecutionContext() )
            {
                throw std::runtime_error(
                    std::format(
                        "Component::build: ExecutionContext must be set before building component '{}'",
                        getName() ) );
            }

            build_config_ = config;

            onBuilding( build_config_ );

            built_ = true;
        }

        /**
         * @brief Returns true if build() has completed successfully.
         */
        virtual bool isBuilt() const final
        {
            return built_;
        }

        /**
         * @brief True if the component was built for the training path
         *        and is not currently in evaluation mode.
         *
         * Output buffers are sized at T=full. Gradient buffers are allocated
         * lazily on the first setEvaluation( false ) call.
         * Always returns false before build() or when isEvaluating() is true.
         */
        bool isTraining() const
        {
            return built_ && !is_training_.load();
        }

        /**
         * @brief Toggle evaluation mode on a Training-built component.
         *
         * Provides a serialized transition between training and evaluation
         * behavioral modes. Idempotent — calling with the current state is
         * a no-op.
         *
         * ## Allocation policy
         *
         * Gradient buffers are allocated once on the first transition to
         * training mode ( setEvaluation( false ) ) and retained for the
         * lifetime of the component. They are never destroyed on entry to
         * evaluation mode — only zeroed. This avoids repeated GPU allocation
         * costs in workflows that toggle between training and evaluation
         * (e.g. periodic eval checkpoints during a training run).
         *
         * ## Transition behavior
         *
         * **First setEvaluation( false ) — enter training:**
         *   - Allocate all gradient and backward state buffers.
         *   - Zero all allocated buffers.
         *   - Bind gradients to the operation via operation_->setGradients().
         *
         * **Subsequent setEvaluation( false ) — resume training:**
         *   - Zero existing gradient buffers (no reallocation).
         *   - Re-bind gradients to the operation.
         *
         * **setEvaluation( true ) — enter evaluation:**
         *   - Zero all gradient buffers (prevent stale gradient leakage
         *     across mode switches).
         *   - Retain all allocations.
         *
         * ## Separation of concerns
         *
         * | Hook                     | Responsibility                              |
         * |--------------------------|---------------------------------------------|
         * | onBuilding()             | Forward and decode path buffer allocation   |
         * | onEvaluationChanging()   | Backward path buffer allocation and zeroing |
         *
         * ## Thread safety
         *
         * The transition is serialized by an internal mutex. The underlying
         * is_evaluating_ atomic is updated before invoking the hook.
         * If the hook throws, the prior state is restored and the exception
         * is rethrown.
         *
         * @param is_evaluating True to enter evaluation mode, false to resume training.
         *
         * @throws std::runtime_error if called before build().
         * @throws std::runtime_error if the component was built with
         *         ExecutionMode::Inference.
         * @throws Any exception propagated from onEvaluationChanging(); prior
         *         evaluation state is restored on throw.
         */
        void setTraining( bool is_training )
        {
            ensureBuilt( "setTraining" );

            std::lock_guard<std::mutex> lock( mode_mutex_ );

            if ( is_training_.load() == is_training )
            {
                return;
            }

            bool prev = is_training_.load();
            is_training_.store( is_training );

            try
            {
                onTrainingChanging( is_training );
            }
            catch ( ... )
            {
                is_training_.store( prev );
                throw;
            }
        }

        // ====================================================================
        // Synchronization
        // ====================================================================

        /**
         * @brief Wait for outstanding device work submitted by this component.
         *
         * On CPU this may be a no-op. Use to ensure results are visible to
         * the host or to measure synchronous timings.
         */
        virtual void synchronize() = 0;

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        /**
         * @brief Return number of trainable parameters.
         *
         * For leaf components this is the element count of owned parameter
         * tensors. CompositeComponent and Network implementations should
         * return the recursive aggregate across all children.
         */
        virtual size_t parameterCount() const = 0;

        /**
         * @brief Clear all model-owned gradients for this component.
         *
         * Default implementation is a no-op. Composite components should
         * override to recurse to children. Leaf components should override
         * to zero their parameter and activation gradients using
         * device-aware helpers.
         */
        virtual void zeroGradients()
        {}

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @internal
         * @brief Persist this component into the provided archive.
         *
         * Contract:
         * - Write stable component metadata needed by an inference loader:
         *     - `type` (string): canonical component type name (e.g. "Linear").
         *     - `version` (int/string): component serialization format version.
         *     - optional `name` or path used by composite containers.
         * - Write `config` data containing all shape-affecting hyper-parameters
         *   (e.g. input/output sizes, bias flags). The inference loader must be
         *   able to construct an instance from these config values.
         * - Write all trainable and persistent tensors as named entries in a
         *   stable canonical order (e.g. "weight", "bias", "running_mean").
         *   Each tensor entry MUST include dtype and shape metadata plus raw
         *   bytes; avoid device-specific handles so archives are device-agnostic.
         * - Do NOT write optimizer state or transient training-only buffers;
         *   those belong in trainer checkpoints, not the inference artifact.
         *
         * Implementations should use the ModelArchive helpers to emit structured
         * component entries so composite components can save nested children
         * deterministically.
         *
         * Postcondition:
         * - Archive contains sufficient metadata and tensor blobs to allow an
         *   inference runtime to recreate the same component type and load its
         *   parameters on a possibly different device.
         */
        virtual void save_( ModelArchive& archive, SerializationMode mode ) const = 0;

        // ====================================================================
        // State and Configuration
        // ====================================================================

        /**
         * @brief Get the component's name identifier.
         *
         * The name is used for logging, diagnostics, and serialization.
         *
         * @return Component full hierarchical path.
         */
        const std::string getName() const
        {
            return name_;
        }

        /**
         * @brief Get the component type identifier.
         *
         * Used for serialization and runtime type identification.
         *
         * @return Component type enum value.
         */
        virtual const ComponentType getType() const = 0;

        // ====================================================================
        // Device information and access
        // ====================================================================

        /**
         * @brief Get the compute device id associated with this component.
         *
         * Must return the device on which parameters and operations execute.
         */
        virtual DeviceId getDeviceId() const = 0;

        /**
         * @brief Compile-time device type for this component instance.
         */
        static constexpr DeviceType getDeviceType()
        {
            return TDeviceType;
        }

        /**
         * @brief Compile-time tensor precision for this component instance.
         */
        static constexpr TensorDataType getPrecision() noexcept
        {
            return TPrecision;
        }

        /**
         * @brief Return the current memory allocation breakdown for this component.
         *
         * Reflects allocations at the moment of the call. The returned stats
         * naturally track the component lifecycle:
         *
         *   After construction              — parameters only
         *   After build( Inference )        — parameters + T=1 state buffers
         *   After build( Training )         — parameters + T=full state buffers
         *   After setEvaluation( false )    — parameters + state + gradients
         *
         * For CompositeComponent and Network, the returned stats are the
         * recursive aggregate of all child components.
         *
         * May be called at any time — no lifecycle preconditions.
         *
         * @return MemoryStats reflecting current allocations.
         */
        virtual MemoryStats getMemoryStats() const = 0;

        // ====================================================================
        // Operators
        // ====================================================================

        /**
         * @brief Stream output uses toString() to provide a human-readable
         *        description of the component.
         */
        friend std::ostream& operator<<( std::ostream& os, const Component& component )
        {
            os << component.toString();
            return os;
        }

        /**
         * @brief Produce a short, human-readable description of the component.
         *
         * Implementations should keep output concise and avoid throwing.
         */
        virtual std::string toString() const = 0;

        /**
         * @brief Load a parameter from serialized tensor data.
         *
         * Loads raw tensor bytes directly into an existing parameter tensor,
         * handling precision conversion and device upload as needed.
         *
         * The component validates that the blob's shape matches the parameter's
         * expected shape, then delegates to the backend to perform:
         * - Precision conversion (blob dtype → parameter dtype)
         * - Device upload (CPU bytes → target device)
         *
         * @param name Parameter name used to locate the target tensor.
         * @param blob Serialized tensor metadata and raw bytes.
         *
         * @throws std::runtime_error if component has no parameters to load.
         * @throws std::runtime_error if blob shape doesn't match parameter shape.
         */
        virtual void loadParameter( const std::string& name, const Serialization::TensorBlob& blob )
        {
            throw std::runtime_error(
                std::format( "Component '{}' does not support parameter loading", getName() ) );
        }

        /**
         * @brief List all available parameter names for this component.
         *
         * Returns an empty vector by default. Leaf components with parameters
         * should override to return their canonical parameter name list in the
         * same stable order used by save_() and loadParameter().
         */
        virtual std::vector<std::string> getParameterNames() const
        {
            return {};
        }

        /**
         * @brief Return non-owning pointers to parameter tensors.
         *
         * The returned tensor pointers remain valid for the lifetime of the
         * component. Order should be canonical (weights before biases).
         */
        virtual std::vector<ITensor*> getParameters() const = 0;

        /**
         * @brief Return non-owning pointers to parameter gradient tensors.
         *
         * Only valid when isTraining() is true.
         *
         * @throws std::runtime_error if called when not in training mode or
         *         before the component has been built.
         */
        virtual std::vector<ITensor*> getGradients() const = 0;

    protected:

        /**
         * @brief The BuildConfig stored at build time.
         *
         * Available to derived classes throughout the component lifetime —
         * in onBuilding(), onEvaluationChanging(), forward(), backward(),
         * and any other method that needs build-time configuration.
         *
         * Key uses:
         * - build_config_.allocationSeqLen() — use when sizing output buffers
         *   in onBuilding(). Returns 1 for Inference, leading_shape[1] for Training.
         * - build_config_.isInference() / isTraining() — query the policy.
         * - build_config_.batchSize() — the batch dimension.
         *
         * Initialized to a placeholder before build() completes. Only valid
         * after isBuilt() returns true.
         */
        BuildConfig build_config_{ shape_t{ 1 } };

        // ====================================================================
        // Execution Context
        // ====================================================================

        /**
         * @brief Set the execution context for this component.
         *
         * Establishes the device and execution environment. Can only be called
         * once — the execution context is immutable after setting.
         *
         * Called by:
         * - The component itself (standalone mode with owned context)
         * - Parent composite when adding child (shared context mode)
         * - ComponentFactory during deserialization
         *
         * After setting the context, the onExecutionContextSet() hook is
         * invoked to allow the component to perform context-dependent
         * initialization.
         *
         * @param context Non-owning pointer to execution context (must be non-null).
         *
         * @throws std::invalid_argument if context is null.
         * @throws std::runtime_error    if context has already been set.
         * @throws std::invalid_argument if context device type doesn't match TDeviceType.
         * @throws std::runtime_error    if onExecutionContextSet() throws; context
         *                               is restored to nullptr on failure.
         */
        void setExecutionContext( IExecutionContext* context )
        {
            if ( !context )
            {
                throw std::invalid_argument(
                    std::format(
                        "Component::setExecutionContext: context cannot be null for component '{}'",
                        getName() ) );
            }

            if ( exec_context_ )
            {
                throw std::runtime_error(
                    std::format(
                        "Component::setExecutionContext: context already set for component '{}'. "
                        "ExecutionContext is immutable after initial assignment.",
                        getName() ) );
            }

            DeviceId device_id = context->getDeviceId();

            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format(
                        "Component::setExecutionContext: device type mismatch for component '{}'. "
                        "Expected {}, but context has device type {}",
                        getName(),
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type ) ) );
            }

            exec_context_ = context;

            try
            {
                onExecutionContextSet();
            }
            catch ( const std::exception& e )
            {
                exec_context_ = nullptr;

                throw std::runtime_error(
                    std::format(
                        "Component::setExecutionContext: onExecutionContextSet() failed for component '{}': {}",
                        getName(), e.what() ) );
            }
        }

        /**
         * @brief Get the shared execution context.
         *
         * Provides access to the execution context for derived classes to:
         * - Query device information
         * - Create tensors on the correct device
         * - Pass to backend operations
         * - Synchronize device work
         *
         * @return Non-owning pointer to execution context (guaranteed non-null).
         *
         * @throws std::runtime_error if context has not been set.
         */
        IExecutionContext* getExecutionContext() const
        {
            if ( !exec_context_ )
            {
                throw std::runtime_error(
                    std::format(
                        "Component::getExecutionContext: context not set for component '{}'. "
                        "Call setExecutionContext() before accessing the context.",
                        getName() ) );
            }

            return exec_context_;
        }

        /**
         * @brief Check if execution context has been set.
         *
         * @return true if context is set, false otherwise.
         */
        bool hasExecutionContext() const noexcept
        {
            return exec_context_ != nullptr;
        }

        /**
         * @brief Lifecycle hook: Called immediately after ExecutionContext is set.
         *
         * Override this to perform initialization that requires a valid
         * ExecutionContext. At the time this is called, getExecutionContext()
         * is guaranteed to return a valid context.
         *
         * Common uses:
         * - Composite components: Create and configure child components.
         * - Device resource allocation: Query device capabilities.
         *
         * Default implementation does nothing.
         *
         * @throws Any exception thrown will cause setExecutionContext() to fail
         *         and restore the component to a "context not set" state.
         */
        virtual void onExecutionContextSet()
        {}

        // ====================================================================
        // Build Hook
        // ====================================================================

        /**
         * @brief Hook invoked by build() to allocate component buffers.
         *
         * Receives the stored BuildConfig. Implementations must use
         * config.allocationSeqLen() when sizing output buffers — this is
         * the single call that makes Inference and Training allocate the
         * correct buffer sizes automatically without per-component logic.
         *
         * @code
         *   // Example — Linear component:
         *   shape_t out_shape =
         *   {
         *       config.batchSize(),
         *       config.allocationSeqLen(),   // 1 for Inference, T for Training
         *       config_.getOutputFeatures()
         *   };
         *   owned_output_ = std::make_unique<TensorType>( device, out_shape,
         *       this->getName() + ".output" );
         * @endcode
         *
         * The default implementation forwards to the legacy
         * onBuilding( const shape_t& ) overload for backwards compatibility.
         * New components should override this overload directly.
         *
         * @note Do not call build() or onBuilding() from within this hook.
         * @note Implementations should either succeed fully or leave no partial
         *       state, as a failed build() may be retried.
         *
         * @param config Build-time configuration. Use config.allocationSeqLen()
         *               to obtain the correct output buffer sequence dimension.
         */
        virtual void onBuilding( const BuildConfig& config )
        {
            onBuilding( config.leadingShape() );
        }

        /**
         * @brief Legacy hook for backwards compatibility.
         *
         * Called by the default onBuilding( const BuildConfig& ) implementation.
         * Existing components that override this signature continue to work
         * unchanged. New components should override onBuilding( const BuildConfig& ).
         *
         * @deprecated Override onBuilding( const BuildConfig& ) instead.
         *
         * @param leading_shape The leading dimensions { B, T, ... }.
         */
        virtual void onBuilding( const shape_t& leading_shape )
        {}

        // ====================================================================
        // Evaluation Mode Hook
        // ====================================================================

        /**
         * @brief Hook invoked by setEvaluation() when evaluation mode changes.
         *
         * Only called on Training-built components. Never called on
         * Inference-built components.
         *
         * ## Allocation policy
         *
         * Gradient buffers are allocated once on the first transition to
         * training mode ( is_evaluating = false ) and retained for the
         * lifetime of the component. They are never destroyed on entry to
         * evaluation mode — only zeroed. This avoids repeated GPU allocation
         * costs in workflows that toggle frequently between training and
         * evaluation.
         *
         * ## Transition behavior
         *
         * **First setEvaluation( false ) — enter training:**
         *   - Allocate all gradient and backward state buffers.
         *   - Zero all allocated buffers.
         *   - Bind gradients to the operation via operation_->setGradients().
         *
         * **Subsequent setEvaluation( false ) — resume training:**
         *   - Zero existing gradient buffers (no reallocation).
         *   - Re-bind gradients to the operation.
         *
         * **setEvaluation( true ) — enter evaluation:**
         *   - Zero all gradient buffers (prevent stale gradient leakage).
         *   - Retain all allocations.
         *
         * ## Separation of concerns
         *
         * | Hook                     | Responsibility                              |
         * |--------------------------|---------------------------------------------|
         * | onBuilding()             | Forward and decode path buffer allocation   |
         * | onEvaluationChanging()   | Backward path buffer allocation and zeroing |
         *
         * ## Example implementation
         *
         * @code
         * void onEvaluationChanging( bool is_evaluating ) override
         * {
         *     if ( !is_evaluating )
         *     {
         *         if ( weight_grad_ == nullptr )
         *         {
         *             // First transition to training — allocate once.
         *             auto device = this->getExecutionContext()->getDeviceId();
         *             weight_grad_ = std::make_unique<TensorType>(
         *                 device, weight_->shape(), this->getName() + ".weight.grad" );
         *             bias_grad_ = std::make_unique<TensorType>(
         *                 device, bias_->shape(), this->getName() + ".bias.grad" );
         *         }
         *         zero( *weight_grad_ );
         *         zero( *bias_grad_ );
         *         operation_->setGradients( weight_grad_.get(), bias_grad_.get() );
         *     }
         *     else
         *     {
         *         // Entering evaluation — zero to prevent stale gradient leakage.
         *         if ( weight_grad_ != nullptr ) zero( *weight_grad_ );
         *         if ( bias_grad_  != nullptr ) zero( *bias_grad_  );
         *     }
         * }
         * @endcode
         *
         * @note Do not allocate gradient buffers in onBuilding(). The evaluation
         *       state is always true (training inactive) during build.
         *
         * @param is_evaluating True when entering evaluation mode, false when
         *                      entering training mode.
         */
        virtual void onTrainingChanging( bool is_training )
        {}

        // ====================================================================
        // Parameter Loading Helper
        // ====================================================================

        /**
         * @brief Load a tensor blob into a parameter tensor with validation.
         *
         * Validates dtype and shape match then copies blob data into the tensor.
         * Intended for use in loadParameter() overrides.
         *
         * @param param_name  Parameter name (used in error messages).
         * @param blob        Source tensor blob from the model archive.
         * @param target      Destination tensor (must be initialized).
         * @param expected_shape Expected tensor shape for validation.
         *
         * @throws std::invalid_argument if dtype or shape mismatch.
         */
        template<TensorDataType TPrecision, typename TMemoryResource>
        void loadParameterFromBlob(
            const std::string& param_name,
            const Serialization::TensorBlob& blob,
            Tensor<TPrecision, TMemoryResource>& target,
            const shape_t& expected_shape )
        {
            if ( blob.metadata.dtype != TPrecision )
            {
                throw std::invalid_argument(
                    std::format( "Parameter '{}' dtype mismatch. Expected {}, got {}",
                        param_name,
                        tensorDataTypeToString( TPrecision ),
                        tensorDataTypeToString( blob.metadata.dtype ) ) );
            }

            if ( blob.metadata.shape != expected_shape )
            {
                throw std::invalid_argument(
                    std::format( "Component '{}' parameter '{}' shape mismatch. Expected {}, got {}",
                        getName(),
                        param_name,
                        shapeToString( expected_shape ),
                        shapeToString( blob.metadata.shape ) ) );
            }

            copyFromBlob( blob, target );
        }

    private:

        std::string name_;
        IExecutionContext* exec_context_{ nullptr };
        bool built_{ false };
        //BuildConfig build_config_{ shape_t{ 1 } };

        std::atomic<bool> is_training_{ false };
        std::mutex mode_mutex_;

        /**
         * @brief Throws if the component has not yet been built.
         *
         * Used as a precondition guard in public methods that require
         * build() to have completed.
         *
         * @param method Caller name for the error message.
         *
         * @throws std::runtime_error if !built_.
         */
        void ensureBuilt( const char* method ) const
        {
            if ( !built_ )
            {
                throw std::runtime_error(
                    std::format( "Component::{}: component '{}' must be built first",
                        method, getName() ) );
            }
        }

        /**
         * @brief Validates the component name.
         *
         * Enforces identifier rules: must start with a letter and contain only
         * letters, digits, '.', '_', '-' with length between 1 and 128 characters.
         *
         * @param name Name to validate.
         *
         * @throws std::invalid_argument if name is not a valid identifier.
         */
        static const std::string& validateName( const std::string& name )
        {
            if ( !isIdentifier( name ) )
            {
                throw std::invalid_argument(
                    std::format(
                        "Component: invalid name '{}'. Name must start with a letter and contain only "
                        "letters, digits, '.', '_', '-' (1..128 chars)",
                        name ) );
            }

            return name;
        }

        /**
         * @brief Checks if a string is a valid component identifier.
         *
         * Rules: start with A-Za-z, then allow A-Za-z0-9._- ; length 1..128.
         *
         * @param s String to check.
         * @return true if valid identifier, false otherwise.
         */
        static bool isIdentifier( const std::string& s ) noexcept
        {
            constexpr std::size_t kMinLen = 1;
            constexpr std::size_t kMaxLen = 128;

            if ( s.size() < kMinLen || s.size() > kMaxLen )
            {
                return false;
            }

            auto isAsciiAlpha = []( unsigned char c ) noexcept
                {
                    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
                };

            auto isAsciiAlphaNum = []( unsigned char c ) noexcept
                {
                    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
                };

            const unsigned char first = static_cast<unsigned char>(s[ 0 ]);

            if ( !isAsciiAlpha( first ) )
            {
                return false;
            }

            for ( unsigned char uc : s )
            {
                if ( isAsciiAlphaNum( uc ) )
                {
                    continue;
                }

                if ( uc == '.' || uc == '_' || uc == '-' )
                {
                    continue;
                }

                return false;
            }

            return true;
        }
    };
}