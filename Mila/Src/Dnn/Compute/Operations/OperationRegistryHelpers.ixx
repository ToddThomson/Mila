/**
 * @file OperationRegistryHelpers.ixx
 * @brief Compile-time templated helpers for querying the OperationRegistry.
 *
 * This module exports two lightweight, compile-time helpers that forward to
 * Mila::Dnn::Compute::OperationRegistry template methods:
 *
 * - `getRegisteredOperations<DeviceType, TensorDataType>()`
 *     Returns the list of registered operation names for the specified compile-time
 *     device and tensor data type.
 *
 * - `isOperationRegistered<DeviceType, TensorDataType>(const std::string&)`
 *     Returns true if the named operation is registered for the specified compile-time
 *     device and tensor data type, false otherwise.
 *
 * Notes:
 * - These helpers perform no runtime device/data-type switching; they are intended
 *   for use when `DeviceType` and `TensorDataType` are known at compile time.
 * - The helpers merely forward to `OperationRegistry::getRegisteredOperations` and
 *   `OperationRegistry::isOperationRegistered` and preserve the registry's semantics.
 * - Keep uses in hot paths minimal to avoid unnecessary copies of returned vectors.
 */

module;
#include <string>
#include <vector>

export module Compute.OperationRegistryHelpers;

import Dnn.TensorDataType;
import Compute.DeviceType;
import Compute.OperationRegistry;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Templated helper returning registered operation names for a compile-time
     * device and tensor data type.
     */
    export template<DeviceType TDevice, TensorDataType TDataType>
        std::vector<std::string> getRegisteredOperations() {
        return OperationRegistry::instance().template getRegisteredOperations<TDevice, TDataType>();
    }

    /**
     * @brief Templated helper that checks whether a named operation is registered
     * for a compile-time device and tensor data type.
     */
    export template<DeviceType TDevice, TensorDataType TDataType>
        bool isOperationRegistered( const std::string& operation_name ) {
        return OperationRegistry::instance().template isOperationRegistered<TDevice, TDataType>( operation_name );
    }
}