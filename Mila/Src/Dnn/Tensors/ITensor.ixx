/**
 * @file ITensor.ixx
 * @brief Interface providing minimal representation for tensor data across different implementations
 *
 * This interface defines the essential methods needed to access tensor information
 * in a type-erased manner, allowing different tensor implementations to be used
 * interchangeably through a common base interface.
 *
 * Memory Access Design:
 * - Public API: Type-safe, host-only access via Tensor::data() (only for host-accessible memory)
 * - Protected API: Raw pointer access via ITensor::rawData() (for TensorOps implementations)
 * - Device code: Kernel parameters receive raw pointers directly (not through this interface)
 */

module;
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

export module Dnn.ITensor;
import Dnn.TensorOps.Base;
import Dnn.TensorDataType;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.MemoryResource;

namespace Mila::Dnn
{
    /**
     * @brief Abstract interface providing essential tensor information and data access
     *
     * ITensor serves as a type-erased interface for accessing tensor data regardless of the
     * specific element type or memory resource implementation. This interface enables polymorphic
     * access to tensor properties and data, making it possible to write generic code that works
     * with different tensor implementations.
     *
     * The interface provides:
     * - Raw data access through void pointers (protected, for TensorOps)
     * - Shape and dimensional information
     * - Data type identification and validation
     * - Type checking utilities for compile-time safety
     *
     * Memory Access Levels:
     * - Public Tensor::data(): Type-safe, host-accessible only (requires constraint)
     * - Protected rawData(): Type-erased, for TensorOps implementations (CPU/CUDA)
     * - Device kernels: Raw pointers passed as kernel parameters (not through this interface)
     *
     * @note This interface is implemented by concrete tensor classes like Tensor<T, MR>
     * @see Tensor
     */
    export class ITensor {
    public:
        virtual ~ITensor() = default;

        /**
		 * @brief Check if the tensor is a scalar (rank 0)
         */
		virtual bool isScalar() const noexcept = 0;

        /**
         * @brief Get tensor dimensional structure
         * @return Immutable reference to vector containing the size of each dimension
         * @note The returned vector's length equals the tensor's rank (number of dimensions)
         * @note Shape values represent the number of elements along each dimension
         */
        virtual const std::vector<int64_t>& shape() const = 0;

        /**
         * @brief Get total number of elements in the tensor
         * @return Total element count (product of all dimension sizes)
         * @note Scalars (rank 0) have size 1 by convention
         * @note Empty tensors (any dimension size 0) have size 0
         */
        virtual size_t size() const = 0;

        /**
         * @brief Get the size in bytes of a single tensor element.
         *
         * Returns the byte size for the tensor's data type (e.g., 4 for FP32, 2 for FP16).
         *
         * @return Size in bytes of one element
         */
        virtual size_t elementSize() const = 0;

        /**
         * @brief Get tensor element data type identifier
         * @return TensorDataType enumeration value corresponding to the element type
         * @note This can be used for runtime type checking and dispatch
         * @see TensorDataType, getDataTypeName(), isType()
         */
        virtual TensorDataType getDataType() const = 0;

        /**
         * @brief Get human-readable name of the tensor's data type
         * @return String representation of the tensor's element type
         * @note Useful for debugging, logging, and user-facing displays
         * @see getDataType(), isType()
         */
        virtual std::string getDataTypeName() const = 0;

		virtual std::shared_ptr<Compute::ComputeDevice> getDevice() const = 0;

        /**
         * @brief Get the device type from the memory resource
         *
         * Convenience method that queries the memory resource for its device type.
         * Equivalent to getMemoryResource()->device_type but more convenient.
         *
         * @return DeviceType enum value (CPU, CUDA, Metal, etc.)
         */
        virtual Compute::DeviceType getDeviceType() const = 0;

        /**
         * @brief Returns the tensor's unique identifier
         *
         * Provides a stable identifier assigned during tensor construction useful for
         * diagnostics and error reporting at the operation layer.
         *
         * @return Unique identifier string (implementation-defined format)
         */
        virtual std::string getUId() const = 0;

        /**
         * @brief Returns the tensor's optional user-assigned name
         *
         * Provides a human-friendly name when available for diagnostics and logging.
         *
         * @return Name string (may be empty if not assigned)
         */
        virtual std::string getName() const = 0;

        /**
         * @brief Type-safe check for tensor element type
         * @tparam T The element type to check against
         * @return true if the tensor's elements are of type T, false otherwise
         * @note This provides compile-time type safety when working with type-erased tensors
         * @note Prefer this method over manual type checking for better safety
         *
         * @code
         * std::shared_ptr<ITensor> tensor = getTensor();
         * if (tensor->isType<float>()) {
         *     // Safe to use with float operations
         * }
         * @endcode
         */
        template<typename T>
        bool isType() const {
            return getDataType() == getTensorDataTypeEnum<T>();
        }

        /**
         * @brief Get raw pointer to tensor data (protected internal API)
         *
         * Provides type-erased access to the underlying tensor memory for
         * TensorOps implementations. This is the internal API used by
         * device-specific operations (CPU, CUDA, etc.) to access tensor memory.
         *
         * Access Control:
         * - Public API: Use Tensor::data() for type-safe host access
         * - Protected API: Use rawData() in TensorOps for device operations
         * - Not exposed to end users for safety
         *
         * Safety Contract:
         * - For host-accessible memory: Pointer is valid for host dereferencing
         * - For device-only memory: Pointer is ONLY valid for device operations
         *   (kernel launches, cudaMemcpy, etc.) - DO NOT dereference on host
         * - Returns nullptr for empty or uninitialized tensors
         *
         * @return Raw void pointer to tensor data, or nullptr if empty
         *
         * @note This pointer's lifetime is tied to the tensor's buffer
         * @note For device memory, this pointer must not be dereferenced on host
         * @note TensorOps implementations are responsible for proper device handling
         *
         * Example Usage (in TensorOps):
         * @code
         * // In CudaTensorOps::copy()
         * const void* src_data = static_cast<const ITensor&>(src).rawData();
         * void* dst_data = static_cast<ITensor&>(dst).rawData();
         * 
         * // Safe: Used for cudaMemcpy (doesn't dereference on host)
         * cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, stream);
         * @endcode
         */
        virtual void* rawData() = 0;

        /**
         * @brief Get raw pointer to tensor data (const version)
         *
         * Const version of rawData() for read-only access to tensor memory.
         *
         * @return Raw const void pointer to tensor data, or nullptr if empty
         * @see rawData() for detailed documentation
         */
        virtual const void* rawData() const = 0;

        /**
         * @brief Get the memory resource managing this tensor's storage
         *
         * Provides access to the memory resource for efficient dispatch to
         * device-specific operations and zero-copy tensor operations when
         * memory resources are compatible.
         *
         * @return Pointer to the memory resource (never null for valid tensors)
         * @note Memory resource lifetime is managed by the tensor
         * @note Enables efficient type-safe downcasting and dispatch
         */
        virtual Compute::MemoryResource* getMemoryResource() const = 0;
    };
}