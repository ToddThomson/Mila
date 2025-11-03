/**
 * @file Module.ixx
 * @brief Base module interface for Mila DNN components.
 *
 * This file declares the templated abstract `Module` class which defines the
 * minimal interface required by all neural network layers and composite
 * modules in the Mila framework.
 *
 * The `Module` type is device-parameterized by the compile-time template
 * parameter `TDeviceType`. Concrete implementations provide device-specific
 * behavior (CPU, CUDA, etc.) while relying on the abstract tensor interface
 * `ITensor` so callers can remain device-agnostic at the API level.
 *
 * Key responsibilities of a Module implementation:
 * - perform forward and backward passes over `ITensor` objects,
 * - expose and persist trainable parameters via `save` / `load`,
 * - report parameter counts and human-readable descriptions,
 * - support execution synchronization when device streams are used.
 *
 * Note:
 * - The base class imposes no ownership model for tensors passed to the
 *   interface; implementations should document any lifetime requirements.
 * - The base class does not mandate specific thread-safety guarantees; module
 *   implementations should document their concurrency properties.
 */

module;
#include <string>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <type_traits>
#include <sstream>
#include <format>
#include <ostream>
#include <cstddef>
#include <vector>

export module Dnn.Module;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Abstract base class for neural network modules.
     *
     * @tparam TDeviceType Compile-time device identifier for this module.
     *
     * Implementations must override the pure virtual interface to provide
     * concrete behaviour. The interface is intentionally minimal to allow both
     * simple layers (e.g. Linear) and composite modules (containers of modules)
     * to share a common API.
     */
    export template<DeviceType TDeviceType>
    class Module
    {
    public:
        virtual ~Module() = default;
        
        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Check if this module has been built and is ready for use.
         *
         * A module is considered built when all its parameters have been allocated
         * and initialized based on the input shape. This method should return true
         * only after build() has been successfully called with a valid input shape.
         *
         * For composite modules, this should recursively check that all child modules
         * are also built.
         *
         * @return true if the module is fully built and ready for forward/backward passes
         * @return false if build() has not been called or shape inference is incomplete
         *
         * @note Operations like parameterCount(), parameters(), save(), and forward()
         *       should only be called when isBuilt() returns true
         * @note Calling build() multiple times is idempotent - subsequent calls after
         *       the first successful build should have no effect
         *
         * @see build()
         */
        virtual bool isBuilt() const = 0;

        /**
         * @brief Initialize module parameters based on input tensor shape.
         *
         * This method performs shape inference and parameter allocation for modules
         * that cannot fully initialize during construction. Parameters are allocated
         * using the ExecutionContext provided at construction time.
         *
         * The build process typically involves:
         * - Inferring any unknown dimensions from the input shape
         * - Allocating and initializing learnable parameters using exec_context_
         * - Validating shape compatibility with module configuration
         * - Recursively building child modules for composite modules
         *
         * @param input_shape Expected shape of input tensors for this module
         *
         * @throws std::invalid_argument if input_shape is incompatible with module config
         * @throws std::runtime_error if parameter allocation fails
         *
         * @note All allocations use the ExecutionContext provided at construction
         * @note Must be called before forward(), parameterCount(), or save()
         * @note Idempotent - subsequent calls after first successful build have no effect
         *
         * @see isBuilt()
         * @see ExecutionContext
         */
        virtual void build( const shape_t& input_shape ) = 0;

        // ====================================================================
		// Compute Operation dispatch
        // ====================================================================

        /**
         * @brief Run the forward computation for this module.
         *
         * @param input Const reference to the input tensor. Implementations
         *        should not assume ownership. The input may be a view or a
         *        scalar tensor depending on the operation.
         * @param output Reference to the tensor that will receive the result.
         *        The caller is responsible for allocating an appropriately
         *        sized tensor unless the implementation documents otherwise.
         *
         * Implementations should document whether they support in-place
         * operations (i.e. where `output` aliases `input`) and any shape
         * broadcasting rules they follow.
         */
        virtual void forward( const ITensor& input, ITensor& output ) = 0;
        
        /**
         * @brief Compute gradients for this module's input given the gradient
         *        of the module output.
         *
         * @param input Const reference to the original forward input. Used by
         *        some layers to compute input gradients.
         * @param output_grad Const reference to the gradient with respect to
         *        the module output.
         * @param input_grad Reference to the tensor that will be populated
         *        with the gradient with respect to the module input. The
         *        caller is responsible for allocation unless noted.
         *
         * Implementations should document whether they accumulate into
         * `input_grad` or overwrite it.
         */
        virtual void backward( 
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) = 0;

		// ====================================================================
		// Synchronization
		// ====================================================================

        /**
         * @brief Synchronize this module's execution stream.
         *
         * Blocks until all asynchronous operations submitted by this module
         * have completed. On CPU-only modules this may be a no-op. Use this
         * when timing, debugging, or when results must be visible on the host.
         */
        virtual void synchronize() = 0;

		// ====================================================================
        // Parameters
		// ====================================================================

		// TJT: Access to parameters is required for optimizers, but the ownership
		// model is unclear. Leaving this commented out for now.

        //virtual std::vector<Tensor*> parameters() = 0;

        /**
         * @brief Return the number of trainable parameters in this module.
         *
         * The count should include all scalar and tensor parameters that are
         * stored by the module and would be persisted by `save`.
         */
        virtual size_t parameterCount() const = 0;

		// ====================================================================
		// Serialization
		// ====================================================================

        /**
         * @brief Persist module parameters into the provided archive.
         *
         * @param archive Archive object used to write named parameter blobs.
         *
         * Implementations should serialize all state required to reconstruct
         * the module parameters (weights, biases, hyper-parameters that affect
         * shape, etc.). May throw on IO or serialization errors.
         */
        virtual void save( ModelArchive& archive ) const = 0;

        /**
         * @brief Load module parameters from the provided archive.
         *
         * @param archive Archive object used to read named parameter blobs.
         *
         * Implementations must handle missing or incompatible data gracefully
         * (throwing exceptions where appropriate) and restore internal state
         * so that subsequent forward/backward calls are valid.
         */
        virtual void load( ModelArchive& archive ) = 0;

        // ====================================================================
        // State and Configuration
        // ====================================================================

        /**
         * @brief Set whether the module is in training mode.
         *
         * Some modules alter behavior between training and evaluation modes
         * (for example, dropout or batch-norm). Calling this sets the
         * module-local flag; composite modules should propagate the flag to
         * child modules.
         *
         * @param is_training True to enable training-mode behavior.
         */
        virtual void setTraining( bool is_training ) = 0;

        /**
         * @brief Query whether the module is in training mode.
         *
         * @returns True if the module is configured for training behavior.
         */
        virtual bool isTraining() const = 0;

        /**
         * @brief Get the module's name.
         *
         * Names are used for logging, diagnostics, and when saving parameters.
         * Implementations should return a stable identifier.
         *
         * @returns The module name string.
         */
        virtual std::string getName() const = 0;
        
        // ====================================================================
        // Device information and access
        // ====================================================================

        /**
         * @brief Compile-time device type for this module instance.
         *
         * @returns The DeviceType specified by the template parameter.
         */
        static constexpr DeviceType getDeviceType() {
            return TDeviceType;
        }

        /**
         * @brief Get the compute device associated with this module.
         *
         * Returns the device on which this module's parameters are allocated and
         * operations are executed. The device is determined by the ExecutionContext
         * provided at module construction time.
         *
         * This method is typically used to:
         * - Allocate intermediate tensors on the same device as the module
         * - Ensure input tensors are on the correct device before forward pass
         * - Create compatible output tensors with proper device placement
         *
         * For composite modules, all child modules should share the same device.
         *
         * @return Shared pointer to the ComputeDevice associated with this module
         *
         * @throws std::runtime_error if called before module construction or if
         *         the ExecutionContext has been destroyed
         *
         * @note The returned device pointer remains valid for the lifetime of the module
         * @note This is a runtime (virtual) method, not compile-time like getDeviceType()
         *
         * @see getDeviceType() for compile-time device type information
         * @see ExecutionContext::getDevice()
         *
         * Example usage:
         * @code
         * auto model = std::make_shared<MLP<DeviceType::Cuda, TensorDataType::FP32>>( exec_ctx, config );
         * auto device = model->getDevice();
         *
         * // Create intermediate tensor on same device as model
         * Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> buffer( device, shape );
         * @endcode
         */
        virtual std::shared_ptr<ComputeDevice> getDevice() const = 0;

        // ====================================================================
        // Operators
        // ====================================================================

        /**
         * @brief Stream output uses `toString()` to provide a human-readable
         * description of the module.
         */
        friend std::ostream& operator<<( std::ostream& os, const Module& module ) {
            os << module.toString();
            return os;
        }

		// ====================================================================
		// Debugging and Description
		// ====================================================================

        /**
         * @brief Produce a short, human-readable description of the module.
         *
         * The returned string should be a single-line description suitable for
         * logging. It typically contains the module type, name (if any), and
         * key configuration values (shapes, sizes).
         *
         * @returns A human-readable single-line description.
         */
        virtual std::string toString() const = 0;
    };
}