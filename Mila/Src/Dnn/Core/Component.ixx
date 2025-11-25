/**
 * @file Component.ixx
 * @brief Base component interface for Mila DNN components.
 *
 * Provides the abstract `Module` template defining the shared lifecycle,
 * parameter, training-mode and introspection APIs used by all Mila modules.
 */

module;
#include <string>
#include <memory>
#include <ostream>
#include <vector>
#include <mutex>
#include <atomic>

export module Dnn.Component;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Abstract base class for neural network modules.
     *
     * @tparam TDeviceType Compile-time device identifier for this module.
     *
     * Module provides a minimal, device-parametrized interface for:
     * - build / lifecycle management,
     * - parameter and gradient access,
     * - synchronization and serialization,
     * - training/evaluation mode transitions with a serialized hook,
     * - short human-readable diagnostics via `toString()`.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Component
    {
    public:
        
        virtual ~Component() = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Query whether the module has been built (shape-dependent init done).
         *
         * Implementations must return true only after a successful call to
         * `build(const shape_t&)` and any required child modules are built.
         */
        virtual bool isBuilt() const = 0;

        /**
         * @brief Perform shape-dependent initialization and allocate parameters.
         *
         * Preconditions:
         * - The provided `input_shape` must be compatible with this module's
         *   configuration. Implementations should validate and throw
         *   std::invalid_argument on mismatch.
         *
         * Postconditions:
         * - Module is ready for `forward()` / `backward()` and `isBuilt()` will
         *   return true if build succeeds.
         *
         * Rebuild policy:
         * - Modules may choose to throw if `build()` is called when already
         *   built; concrete modules document their policy.
         *
         * @param input_shape Expected input tensor shape for forward calls.
         */
        virtual void build( const shape_t& input_shape ) = 0;
		// REVIEW: Can we enforce that build() can only be called once at the Module level?

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
        virtual void save_( ModelArchive& archive, SerializationMode model ) const = 0;

        // ====================================================================
        // State and Configuration
        // ====================================================================

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
            catch (...)
            {
                // Revert to previous state on exception to preserve invariants.
                is_training_.store( prev );
                throw;
            }
        }

        /**
         * @brief Query whether the module is configured for training behavior.
         *
         * This performs a relaxed atomic read and is safe for concurrent access.
         * Subclasses may override to implement derived behavior.
         *
         * @return true if module is in training mode.
         */
        bool isTraining() const
        {
            return is_training_.load();
        }

        /**
         * @brief Module name used for logging and diagnostics.
         *
         * Implementations should return a stable identifier used in `toString()`
         * and when saving parameters.
         */
        virtual std::string getName() const = 0;

        // ====================================================================
        // Device information and access
        // ====================================================================

        /**
         * @brief Compile-time device type for this module instance.
         */
        static constexpr DeviceType getDeviceType()
        {
            return TDeviceType;
        }

        static constexpr TensorDataType getPrecision() noexcept
        {
            return TPrecision;
        }


        /**
         * @brief Get the compute device associated with this module.
         *
         * Must return the device on which parameters and operations execute.
         */
        virtual std::shared_ptr<ComputeDevice> getDevice() const = 0;

        // ====================================================================
        // Operators
        // ====================================================================

        /**
         * @brief Stream output uses `toString()` to provide a human-readable
         * description of the module.
         */
        friend std::ostream& operator<<( std::ostream& os, const Component& module )
        {
            os << module.toString();

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

    private:

        std::atomic<bool> is_training_{ false };
        std::mutex training_mutex_;
    };
}