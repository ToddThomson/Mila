/**
 * @file ITensor.ixx
 * @brief Interface providing minimal representation for tensor data across different implementations
 *
 * This interface defines the essential methods needed to access tensor information
 * in a type-erased manner, allowing different tensor implementations to be used
 * interchangeably through a common base interface.
 */

module;
#include <vector>
#include <memory>
#include <string>

export module Dnn.ITensor;

import Dnn.TensorDataType;
import Compute.DeviceType;
import Compute.DeviceContext;
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
     * - Raw data access through void pointers
     * - Shape and dimensional information
     * - Data type identification and validation
     * - Type checking utilities for compile-time safety
     *
     * @note This interface is implemented by concrete tensor classes like Tensor<T, MR>
     * @see Tensor
     */
    export class ITensor {
    public:
        virtual ~ITensor() = default;

        /**
         * @brief Get tensor dimensional structure
         * @return Immutable reference to vector containing the size of each dimension
         * @note The returned vector's length equals the tensor's rank (number of dimensions)
         * @note Shape values represent the number of elements along each dimension
         */
        virtual const std::vector<size_t>& shape() const = 0;

        /**
         * @brief Get total number of elements in the tensor
         * @return Total element count (product of all dimension sizes)
         * @note Scalars (rank 0) have size 1 by convention
         * @note Empty tensors (any dimension size 0) have size 0
		 */
		virtual size_t size() const = 0;

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
         * @brief Type-safe check for tensor element type
         * @tparam T The element type to check against
         * @return true if the tensor's elements are of type T, false otherwise
         * @note This provides compile-time type safety when working with type-erased tensors
         * @note Prefer this method over manual type checking for better safety
         *
         * @code
         * std::shared_ptr<ITensor> tensor = getTensor();
         * if (tensor->isType<float>()) {
         *     float* data = static_cast<float*>(tensor->data());
         *     // Safe to use data as float*
         * }
         * @endcode
         */
        template<typename T>
        bool isType() const {
            return getDataType() == getTensorDataTypeEnum<T>();
        }

    protected:

        /**
         * @brief Get the device context for this tensor
         *
         * Provides access to the device context for stream management, multi-GPU
         * coordination, and device-specific operations.
         *
         * @return Shared pointer to device context (never null for valid tensors)
         * @note Device context manages device binding, streams, and synchronization
         * @note Essential for kernel launches and device coordination
         */
        virtual std::shared_ptr<Compute::DeviceContext> getDeviceContext() const = 0;

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

        // Allow Module to access protected members
        template<Compute::DeviceType>
        friend class Module;
    };
}