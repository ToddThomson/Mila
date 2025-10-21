/**
 * @file OperationBase.ixx
 * @brief Core abstraction for neural network operations in the Mila framework
 */

module;
#include <string>
#include <memory>
#include <stdexcept>

export module Compute.OperationBase;

import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Compute.OperationType;

namespace Mila::Dnn::Compute
{
    export template <DeviceType TDeviceType, TensorDataType TPrecision>
    class OperationBase {
        public:
            static constexpr DeviceType device_type = TDeviceType;
            static constexpr TensorDataType data_type = TPrecision;
            using DataTypeTraits = TensorDataTypeTraits<TPrecision>;

            virtual ~OperationBase() = default;

            /**
             * @brief Gets the operation type identifier.
             */
            virtual OperationType getOperationType() const = 0;

            /**
             * @brief Gets the device type for this operation.
             */
            virtual DeviceType getDeviceType() const { return TDeviceType; }

            /**
             * @brief Gets the tensor data type for this operation.
             */
            virtual TensorDataType getDataType() const { return TPrecision; }

            virtual std::string getName() const = 0;
    };
}