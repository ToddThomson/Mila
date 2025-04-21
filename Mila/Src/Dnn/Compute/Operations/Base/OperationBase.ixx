module;
#include <string>  
#include <memory>  

export module Compute.OperationBase;

import Dnn.TensorTraits;
import Compute.DeviceType; 
import Compute.DeviceContext;
import Compute.OperationType;  

namespace Mila::Dnn::Compute
{
    /**
    * @brief Base class for all compute operations.
    *
    * @tparam TInput The data type of the input tensor elements.
    * @tparam TDataType The data type of the precision tensor elements.
    */
    export template <typename TInput, typename TPrecision, DeviceType TDeviceType>
        requires ValidTensorTypes<TInput, TPrecision>
    class OperationBase {
    public:
        /**
        * @brief Constructs an OperationBase object with a specific device context.
        *
        * @param operation_type The type of the operation.
        * @param context The device context to use for this operation.
        */
        OperationBase( OperationType operation_type, std::shared_ptr<DeviceContext> context )
            : operation_type_( operation_type ), device_context_( context ) {}

        /**
        * @brief Virtual destructor for the OperationBase class.
        */
        virtual ~OperationBase() = default;

        /**
        * @brief Gets the name of the operation.
        *
        * @return The name of the operation.
        */
        virtual std::string getName() const = 0;

        /**
        * @brief Gets the device context.
        *
        * @return The device context.
        */
        std::shared_ptr<DeviceContext> getDeviceContext() const {
            return device_context_;
        }

        /**
        * @brief Gets the device type.
        *
        * @return The device type.
        */
        DeviceType getDeviceType() const {
            return device_context_->getDevice()->getDeviceType();
        }

        /**
        * @brief Gets the operation type.
        *
        * @return The operation type.
        */
        OperationType getOperationType() const {
            return operation_type_;
        }

    private:
        OperationType operation_type_; ///< The operation type.
        std::shared_ptr<DeviceContext> device_context_; ///< The device context.
    };
}