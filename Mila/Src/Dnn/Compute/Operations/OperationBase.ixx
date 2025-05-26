/**
 * @file OperationBase.ixx
 * @brief Base class for all compute operations in the Mila neural network framework.
 * @details Provides a common interface and functionality for neural network operations
 *          across different device types and precision levels.
 */

module;
#include <string>  
#include <memory>  

export module Compute.OperationBase;

import Dnn.TensorTraits;
import Compute.DeviceType; 
import Compute.DeviceContext;
import Compute.Precision;
import Compute.OperationType;  

namespace Mila::Dnn::Compute
{
    /**
     * @brief Base class for all compute operations in the Mila neural network framework.
     *
     * @details This abstract base class defines the common interface for all operations that can be
     * performed in the neural network computation graph, regardless of the device type
     * (CPU, CUDA, etc). Specific operations inherit from this class and implement their
     * specialized behavior while adhering to a consistent interface.
     *
     * @tparam TInput1 The data type of the first input tensor elements.
     *                 Must satisfy ValidTensorType constraint.
     * @tparam TInput2 The data type of the second input tensor elements, defaults to TInput1.
     *                 Must satisfy ValidTensorType constraint.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput1.
     *                 Must satisfy ValidFloatTensorType constraint.
     * @tparam TDeviceType The target device type for the operation, defaults to DeviceType::Cuda.
     */
    export template <DeviceType TDeviceType = DeviceType::Cuda, typename TInput1 = float, typename TInput2 = TInput1, typename TOutput = TInput1>
        requires ValidTensorTypes<TInput1, TInput2> && ValidFloatTensorType<TOutput>
    class OperationBase {
    public:

        /**
         * @brief Constructs an OperationBase object with a specific device context and compute precision.
         *
         * @details Initializes the operation with the specified operation type and device context,
         * using the template parameter-specified compute precision.
         *
         * @param operation_type The type of the operation (from OperationType enum).
         * @param context The device context to use for this operation. Must not be null.
         * @param precision_policy The compute precision policy to use for this operation.
         * @throw std::invalid_argument May throw if context is null (implementation dependent).
         */
        OperationBase( OperationType operation_type, std::shared_ptr<DeviceContext> context )
            : operation_type_( operation_type ), device_context_( context ) {}

        /**
         * @brief Virtual destructor for the OperationBase class.
         *
         * @details Ensures proper cleanup of derived class resources when destroyed through
         * a base class pointer. Default implementation is sufficient for this base class.
         */
        virtual ~OperationBase() = default;

        /**
         * @brief Gets the name of the operation.
         *
         * @details This pure virtual function must be implemented by derived classes to return
         * a unique identifier string for the specific operation type. The name should be
         * descriptive and consistent across framework components.
         *
         * @return std::string A unique name identifying this operation type.
         */
        virtual std::string getName() const = 0;

        /**
         * @brief Gets the device context associated with this operation.
         *
         * @details The device context contains information about the execution environment,
         * including the device, streams, and memory resources. This context is used for
         * all device interactions performed by this operation.
         *
         * @return std::shared_ptr<DeviceContext> The device context for this operation.
         */
        std::shared_ptr<DeviceContext> getDeviceContext() const {
            return device_context_;
        }

        /**
         * @brief Gets the device type for this operation.
         *
         * @details This is a convenience method that retrieves the device type from the
         * associated device context. It delegates to the device context's device to
         * determine the actual hardware target.
         *
         * @return DeviceType The type of device (CPU, CUDA, etc.) for this operation.
         */
        DeviceType getDeviceType() const {
            return device_context_->getDevice()->getDeviceType();
        }

        /**
         * @brief Gets the operation type enumeration value.
         *
         * @details Returns the operation type that was specified during construction.
         * This identifies the category of neural network operation being performed.
         *
         * @return OperationType The enumeration value identifying this operation's category.
         */
        OperationType getOperationType() const {
            return operation_type_;
        }

    private:
        OperationType operation_type_;      ///< The operation type identifier.
        std::shared_ptr<DeviceContext> device_context_; ///< The device context for execution.
    };
}