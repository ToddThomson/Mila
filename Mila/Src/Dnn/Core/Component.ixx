/**
 * @file Component.ixx
 * @brief Base component interface for Mila DNN components.
 *
 * Provides the abstract `Component` template defining the shared lifecycle,
 * parameter, training-mode and introspection APIs used by all Mila modules.
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

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Abstract base class for neural network components.
     *
     * Component enforces a single ownership model: all components receive a non-owning
     * pointer to an IExecutionContext that is owned by the parent (Network or test fixture).
     * This ensures consistent resource sharing and eliminates dual constructor patterns.
     *
     * Ownership model:
     * - Components NEVER own ExecutionContext
     * - Parent (Network/CompositeComponent) owns and provides shared context
     * - Tests explicitly create context and pass raw pointer to components
     *
     * The base class provides:
     * - build / lifecycle management with protected onBuilding() hook,
     * - parameter and gradient access,
     * - synchronization and serialization,
     * - training/evaluation mode transitions with a serialized onTrainingChanging() hook,
     * - short human-readable diagnostics via `toString()`.
     *
     * @tparam TDeviceType Compile-time device identifier for this component.
     * @tparam TPrecision Tensor data precision for this component.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Component
    {
        // Allow composite parent to manage child contexts
        template<DeviceType, TensorDataType>
        friend class CompositeComponent;

    public:

        /**
         * @brief Construct component with required name identifier.
         *
         * The name is used for identification, logging, and serialization.
         * Names must be valid identifiers: start with a letter, contain only
         * letters, digits, '.', '_', '-', and be 1-128 characters long.
         *
         * @param name Component name identifier (mandatory)
         *
         * @throws std::invalid_argument if name is not a valid identifier
         */
        explicit Component( const std::string& name )
            : name_( validateName( name ) ), built_( false ), is_training_( false )
        {
        }
        
        virtual ~Component() = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Build the component with the given input shape.
         *
         * This method performs a one-time initialization of the component based
         * on the input shape. It is idempotent in the sense that calling build()
         * multiple times is an error rather than a no-op.
         *
         * Build sequence:
         * 1. Verify component is not already built (throws if already built)
         * 2. Invoke onBuilding(input_shape) hook for pre-build setup
         * 3. Mark component as built
         *
         * Exception safety:
         * - If onBuilding() throws, the component remains unbuilt
         * - Component is in a valid state and build() may be retried
         *
         * @param input_shape The shape of the input tensor.
         *
         * @throws std::logic_error if component is already built.
         * @throws std::invalid_argument if input_shape is invalid for this component.
         * @throws Any exception from onBuilding().
         */
        virtual void build( const shape_t& input_shape ) final
        {
            if ( isBuilt() )
            {
                throw std::logic_error( "Component already built. Cannot rebuild." );
            }

            if ( !hasExecutionContext() )
            {
                throw std::runtime_error(
                    std::format( "Component::build: ExecutionContext must be set before building component '{}'",
                        getName() )
                );
            }

            onBuilding( input_shape );

            built_ = true;
        }

        /**
         * @brief Check if the component has been built.
         *
         * @return true if build() has completed successfully.
         */
        virtual bool isBuilt() const final
        {
            return built_;
        }

        /**
         * @brief Centralized logic for toggling training mode.
         *
         * This method provides a serialized transition between evaluation and
         * training modes. It is idempotent: calling with the current mode is a no-op.
         *
         * Behavior:
         * - Thread-safety: the transition is serialized by an internal mutex.
         * - Atomic update: the underlying `is_training_` atomic is updated to the
         *   new value before invoking the hook.
         * - Hook: invokes `onTrainingChanging( is_training )` while the mutex is held.
         * - Exception safety: if the hook throws, `setTraining()` restores the
         *   previous training state and rethrows the exception.
         *
         * Usage:
         * - Call `setTraining(true)` to enable training behavior (allocate gradients,
         *   enable backward, etc.).
         * - Call `setTraining(false)` to enter evaluation mode (disable gradient use).
         *
         * @param is_training True to enable training-mode behavior.
         *
         * @throws Any exception propagated from the `onTrainingChanging()` hook;
         *         if thrown, the prior training state is restored.
         */
        void setTraining( bool is_training )
        {
            std::lock_guard<std::mutex> lk( training_mutex_ );

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

        /**
         * @brief Query whether the module is configured for training behavior.
         *
         * This performs a relaxed atomic read and is safe for concurrent access.
         *
         * @return true if module is in training mode.
         */
        bool isTraining() const
        {
            return is_training_.load();
        }

        // ====================================================================
        // Synchronization
        // ====================================================================

        /**
         * @brief Wait for outstanding device work submitted by this module.
         *
         * On CPU this may be a no-op. Use to ensure results are visible to the
         * host or to measure synchronous timings.
         */
        virtual void synchronize() = 0;

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        /**
         * @brief Return number of scalar trainable parameters owned by this module.
         */
        virtual size_t parameterCount() const = 0;

        /**
         * @brief Return non-owning pointers to parameter tensors.
         *
         * The returned tensor pointers remain valid for the lifetime of the
         * module. Order should be canonical (weights before biases).
         */
        virtual std::vector<ITensor*> getParameters() const = 0;

        /**
         * @brief Return non-owning pointers to parameter gradient tensors.
         *
         * Only valid when the module is in training mode.
         *
         * @throws std::runtime_error if called when not in training mode or
         *         before the module has been built.
         */
        virtual std::vector<ITensor*> getGradients() const = 0;

        // ====================================================================
        // Serialization
        // ====================================================================

         /**
         * @internal
         * @brief Persist this module into the provided archive.
         *
         * Contract:
         * - Write stable module metadata needed by an inference loader:
         *     - `type` (string): canonical module type name (e.g. "Linear").
         *     - `version` (int/string): module serialization format version.
         *     - optional `name` or path used by composite containers.
         * - Write `config` data containing all shape-affecting hyper-parameters
         *   (for example input/output sizes, bias flags). The inference loader
         *   must be able to construct an instance from these config values.
         * - Write all trainable and persistent tensors as named entries in a
         *   stable canonical order (e.g. "weights", "bias", "running_mean").
         *   Each tensor entry MUST include dtype and shape metadata plus raw
         *   bytes; avoid device-specific handles so archives are device-agnostic.
         * - Do not write optimizer state or transient training-only buffers here;
         *   those belong in trainer checkpoints, not the inference artifact.
         *
         * Implementations should use the ModelArchive helpers to emit structured
         * module entries so composite modules can save nested child modules
         * deterministically.
         *
         * Postcondition:
         * - Archive contains sufficient metadata and tensor blobs to allow an
         *   inference runtime to recreate the same module type and load its
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
         * @return Component name string
         */
        const std::string getName() const
        {
            return name_;
        }

        // ====================================================================
        // Device information and access
        // ====================================================================

        /**
         * @brief Get the compute device id associated with this module.
         *
         * Must return the device on which parameters and operations execute.
         */
        virtual DeviceId getDeviceId() const = 0;

        /**
         * @brief Compile-time device type for this module instance.
         */
        static constexpr DeviceType getDeviceType()
        {
            return TDeviceType;
        }

        /**
         * @brief Compile-time tensor precision for this module instance.
         */
        static constexpr TensorDataType getPrecision() noexcept
        {
            return TPrecision;
        }

        // ====================================================================
        // Operators
        // ====================================================================

        /**
         * @brief Stream output uses `toString()` to provide a human-readable
         * description of the component.
         */
        friend std::ostream& operator<<( std::ostream& os, const Component& component )
        {
            os << component.toString();

            return os;
        }

        /**
         * @brief Produce a short, human-readable description of the module.
         *
         * Implementations should keep output concise and avoid throwing.
         */
        virtual std::string toString() const = 0;

    protected:

        /**
         * @brief Set the execution context for this component.
         *
         * This method establishes the device and execution environment for the component.
         * It can only be called once - the execution context is immutable after setting.
         *
         * Called by:
         * - The component itself (standalone mode with owned context)
         * - Parent composite when adding child (shared context mode)
         * - ComponentFactory during deserialization
         *
         * After setting the context, the onExecutionContextSet() hook is invoked to allow
         * the component to perform context-dependent initialization.
         *
         * @param context Non-owning pointer to execution context (must be non-null)
         *
         * @throws std::invalid_argument if context is null
         * @throws std::runtime_error if context has already been set (immutability violation)
         * @throws std::invalid_argument if context device type doesn't match TDeviceType
         */
        void setExecutionContext( IExecutionContext* context )
        {
            if ( !context )
            {
                throw std::invalid_argument(
                    std::format( "Component::setExecutionContext: context cannot be null for component '{}'",
                        getName() )
                );
            }

            if ( exec_context_ )
            {
                throw std::runtime_error(
                    std::format( "Component::setExecutionContext: context already set for component '{}'. "
                        "ExecutionContext is immutable and cannot be changed after initial assignment.",
                        getName() )
                );
            }

            DeviceId device_id = context->getDeviceId();

            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format( "Component::setExecutionContext: device type mismatch for component '{}'. "
                        "Expected {}, but context has device type {}",
                        getName(),
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type ) )
                );
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
                    std::format( "Component::setExecutionContext: onExecutionContextSet() failed for component '{}': {}",
                        getName(), e.what() )
                );
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
         */
        IExecutionContext* getExecutionContext() const
        {
            if ( !exec_context_ )
            {
                throw std::runtime_error(
                    std::format( "Component::getExecutionContext: context not set for component '{}'. "
                        "Call setExecutionContext() or provide DeviceId to constructor.",
                        getName() )
                );
            }

            return exec_context_;
        }

        /**
         * @brief Check if execution context has been set.
         *
         * @return true if context is set, false otherwise
         */
        bool hasExecutionContext() const noexcept
        {
            return exec_context_ != nullptr;
        }

        /**
         * @brief Lifecycle hook: Called immediately after ExecutionContext is set.
         *
         * Override this to perform initialization that requires a valid ExecutionContext.
         * At the time this is called, getExecutionContext() is guaranteed to return a
         * valid context.
         *
         * Common uses:
         * - Composite components: Create and configure child components
         * - Leaf components: Usually don't need this (use onBuilding instead)
         * - Device resource allocation: Query device capabilities, allocate memory pools
         *
         * Default implementation does nothing.
         *
         * @throws Any exception thrown will cause setExecutionContext() to fail and
         *         restore the component to a "context not set" state.
         */
        virtual void onExecutionContextSet() {}

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * The hook is called while `training_mutex_` is held and after the
         * `is_training_` atomic has been updated to the new value. Implementations
         * should perform any bind/unbind or allocation/free actions required for
         * the new mode (for example, allocate/free gradient buffers or bind/unbind
         * backend gradient pointers).
         *
         * Preconditions and expectations:
         * - MUST NOT call `setTraining()` (no reentrancy).
         * - Should avoid throwing; if an exception is thrown it will be
         *   propagated to the caller of `setTraining()` and the previous state
         *   will be restored by `setTraining()`.
         *
         * Threading:
         * - Hook runs with `training_mutex_` held; callers may use `isTraining()`
         *   to observe the updated mode inside the hook.
         *
         * @param is_training true if training mode is enabled (gradients allowed).
         */
        virtual void onTrainingChanging( bool is_training )
        {
        }

        /**
         * @brief Hook invoked before the component is built.
         *
         * The hook is called after `isBuilt()` has been verified as false but
         * before the component is marked as built. Implementations should perform
         * any validation, child component preparation, or pre-build setup required.
         *
         * Preconditions and expectations:
         * - MUST NOT call `build()` (no reentrancy).
         * - Should avoid throwing; if an exception is thrown it will be
         *   propagated to the caller of `build()` and the component will
         *   remain in an unbuilt state.
         * - Called while the component is guaranteed to be in an unbuilt state.
         *
         * Threading:
         * - Hook runs before any state modification; implementations requiring
         *   thread safety should use appropriate synchronization.
         *
         * Common use cases:
         * - CompositeComponent: propagate build to children
         * - Network: validate graph connectivity
         * - Custom components: allocate auxiliary resources
         *
         * @param input_shape The shape that will be used for building.
         */
        virtual void onBuilding( const shape_t& input_shape )
        {
        }

    private:

        std::string name_;

        IExecutionContext* exec_context_{ nullptr };

        std::atomic<bool> is_training_{ false };
        std::mutex training_mutex_;

        bool built_{ false };

        /**
         * @brief Validates the component name.
         *
         * Enforces identifier rules: must start with a letter and contain only
         * letters, digits, '.', '_', '-' with length between 1 and 128 characters.
         *
         * @param name Name to validate
         *
         * @throws std::invalid_argument if name is not a valid identifier
         */
        static const std::string& validateName( const std::string& name )
        {
            if ( !isIdentifier( name ) )
            {
                throw std::invalid_argument(
                    std::format(
                        "Component: invalid name '{}'. Name must start with a letter and contain only "
                        "letters, digits, '.', '_', '-' (1..128 chars)",
                        name
                    )
                );
            }

            return name;
        }

        /**
         * @brief Checks if string is a valid identifier.
         *
         * Rules: start with A-Za-z, then allow A-Za-z0-9._- ; length 1..128.
         *
         * @param s String to check
         * @return true if valid identifier, false otherwise
         */
        static bool isIdentifier( const std::string& s ) noexcept
        {
            constexpr std::size_t kMinLen = 1;
            constexpr std::size_t kMaxLen = 128;

            if ( s.size() < kMinLen || s.size() > kMaxLen )
            {
                return false;
            }

            auto isAsciiAlpha = []( unsigned char c ) noexcept {
                return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
            };

            auto isAsciiAlphaNum = []( unsigned char c ) noexcept {
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