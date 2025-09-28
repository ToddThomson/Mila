/**
 * @file ITensorData.ixx
 * @brief Interface providing minimal representation for tensor data across different implementations
 *
 * This interface defines the essential methods needed to access tensor information
 * in a type-erased manner, allowing different tensor implementations to be used
 * interchangeably through a common base interface.
 */

module;
#include <vector>
#include <string>

export module Dnn.TensorData;

import Dnn.TensorDataType;

namespace Mila::Dnn
{
    /**
     * @brief Abstract interface providing essential tensor information and data access
     *
     * ITensorData serves as a type-erased interface for accessing tensor data regardless of the
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
    export class ITensorData {
    public:
        virtual ~ITensorData() = default;

        /**
         * @brief Get mutable raw pointer to tensor data
         * @return Non-const void pointer to the underlying tensor data buffer
         * @warning Caller is responsible for casting to the appropriate element type
         * @note Use getDataType() or isType<T>() to verify the correct type before casting
         */
        virtual void* rawData() = 0;

        /**
         * @brief Get immutable raw pointer to tensor data
         * @return Const void pointer to the underlying tensor data buffer
         * @warning Caller is responsible for casting to the appropriate element type
         * @note Use getDataType() or isType<T>() to verify the correct type before casting
         */
        virtual const void* rawData() const = 0;

        /**
         * @brief Get tensor dimensional structure
         * @return Immutable reference to vector containing the size of each dimension
         * @note The returned vector's length equals the tensor's rank (number of dimensions)
         * @note Shape values represent the number of elements along each dimension
         */
        virtual const std::vector<size_t>& shape() const = 0;

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
         * @brief Type-safe check for tensor element type
         * @tparam T The element type to check against
         * @return true if the tensor's elements are of type T, false otherwise
         * @note This provides compile-time type safety when working with type-erased tensors
         * @note Prefer this method over manual type checking for better safety
         *
         * @code
         * std::shared_ptr<ITensorData> tensor = getTensor();
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
    };
}