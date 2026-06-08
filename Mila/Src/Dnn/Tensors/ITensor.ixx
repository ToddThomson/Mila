/**
 * @file ITensor.ixx
 * @brief Interface providing minimal representation for tensor data across different implementations.
 *
 * Defines the essential methods needed to access tensor information in a type-erased
 * manner. Concrete implementations (Tensor<T, MR>) provide typed access on top of this base.
 */

module;
#include <memory>
#include <string>
#include <cstdint>

export module Dnn.ITensor;

import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.MemoryResource;

namespace Mila::Dnn
{
    /**
     * @brief Abstract interface providing essential tensor information and data access.
     *
     * ITensor serves as a type-erased interface for accessing tensor data regardless of the
     * specific element type or memory resource implementation. This interface enables polymorphic
     * access to tensor properties and data, making it possible to write generic code that works
     * with different tensor implementations.
     *
     * Memory access levels:
     * - Public Tensor::data()  : Type-safe, host-accessible tensors only.
     * - Protected rawData()    : Type-erased void pointer for TensorOps implementations.
     * - Device kernels         : Raw pointers passed as kernel parameters (not through this interface).
     *
     * @see Tensor
     */
    export class ITensor
    {
    public:
        virtual ~ITensor() = default;

        /**
         * @brief Check if this tensor is a view of another tensor's data.
         *
         * @return true if this is a view, false if it owns its buffer.
         */
        virtual bool isView() const = 0;

        /**
         * @brief Check if the tensor is a scalar (rank 0).
         */
        virtual bool isScalar() const noexcept = 0;

        /**
         * @brief Get the tensor dimensional structure.
         *
         * @return Immutable reference to the TensorShape describing each dimension size.
         *         Length (ndim) equals the tensor's rank. Scalars return an empty shape.
         */
        virtual const shape_t& shape() const = 0;

        /**
         * @brief Get total number of elements in the tensor.
         *
         * @return Product of all dimension sizes. Scalars return 1; empty tensors return 0.
         */
        virtual size_t size() const = 0;

        /**
         * @brief Get the size in bytes of a single tensor element.
         *
         * @return Byte size for the tensor's data type (e.g. 4 for FP32, 2 for FP16).
         */
        virtual size_t elementSize() const = 0;

        /**
         * @brief Get the total storage size in bytes backing the tensor's buffer.
         *
         * Authoritative byte count for raw memory operations (memset, memcpy, etc.).
         *
         * @return Number of bytes occupied by the tensor's logical elements, or 0 if no buffer.
         */
        virtual size_t getStorageSize() const = 0;

        /**
         * @brief Get tensor element data type identifier.
         *
         * @return TensorDataType enumeration value for the element type.
         */
        virtual TensorDataType getDataType() const = 0;

        /**
         * @brief Get human-readable name of the tensor's data type.
         *
         * @return String representation of the element type (e.g. "FP32", "BF16").
         */
        virtual std::string getDataTypeName() const = 0;

        virtual Compute::DeviceId getDeviceId() const = 0;

        /**
         * @brief Get the device type from the memory resource.
         *
         * @return DeviceType enum value (Cpu, Cuda, etc.).
         */
        virtual Compute::DeviceType getDeviceType() const = 0;

        /**
         * @brief Returns the tensor's unique identifier.
         *
         * Stable identifier assigned at construction, useful for diagnostics.
         *
         * @return Unique identifier string (implementation-defined format).
         */
        virtual std::string getUId() const = 0;

        /**
         * @brief Returns the tensor's optional user-assigned name.
         *
         * @return Name string (may be empty if not assigned).
         */
        virtual std::string getName() const = 0;

        /**
         * @brief Type-safe check for tensor element type.
         *
         * @tparam T The element type to check against.
         * @return true if the tensor's elements are of type T.
         */
        template<typename T>
        bool isType() const
        {
            return getDataType() == getTensorDataTypeEnum<T>();
        }

        /**
         * @brief Get raw pointer to tensor data.
         *
         * Type-erased access for TensorOps implementations. For device-only memory,
         * this pointer must not be dereferenced on the host — it is only valid for
         * device operations (kernel launches, cudaMemcpy, etc.).
         *
         * @return Raw void pointer to tensor data, or nullptr if empty.
         */
        virtual void* rawData() = 0;

        /**
         * @brief Get raw pointer to tensor data (const version).
         *
         * @return Raw const void pointer to tensor data, or nullptr if empty.
         */
        virtual const void* rawData() const = 0;

        /**
         * @brief Get the memory resource managing this tensor's storage.
         *
         * @return Pointer to the memory resource (never null for valid tensors).
         */
        virtual Compute::MemoryResource* getMemoryResource() const = 0;
    };
}