/**
 * @file OperationBase.ixx
 * @brief Base class for all compute operations in the Mila neural network framework.
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
     * This abstract base class defines the common interface for all operations that can be
     * performed in the neural network computation graph, regardless of the device type
     * (CPU, CUDA, etc). Specific operations inherit from this class and implement their
     * specialized behavior while adhering to a consistent interface.
     *
     * @tparam TOutput The data type of the output tensor elements.
     *                 Must satisfy ValidFloatTensorType constraint.
     * @tparam TInput1 The data type of the first input tensor elements, defaults to TOutput.
     *                 Must satisfy ValidTensorType constraint.
     * @tparam TInput2 The data type of the second input tensor elements, defaults to TInput1.
     *                 Must satisfy ValidTensorType constraint.
     * @tparam TDeviceType The target device type for the operation, defaults to CUDA.
     */
    export template <typename TOutput, typename TInput1 = TOutput, typename TInput2 = TInput1, DeviceType TDeviceType = DeviceType::Cuda>
        requires ValidFloatTensorType<TOutput>&& ValidTensorTypes<TInput1, TInput2>
    class OperationBase {
    public:
        /**
         * @brief Constructs an OperationBase object with a specific device context.
         *
         * Initializes the operation with the specified operation type and device context.
         * The device context determines where and how the operation will be executed.
         *
         * @param operation_type The type of the operation (from OperationType enum).
         * @param context The device context to use for this operation.
         */
        OperationBase( OperationType operation_type,  ComputePrecision compute_precision, std::shared_ptr<DeviceContext> context )
            : operation_type_( operation_type ), compute_precision_( compute_precision ), device_context_( context ) {}

        /**
         * @brief Constructs an OperationBase object with a specific device context and default compute precision.
         *
         * Initializes the operation with the specified operation type and device context,
         * using the default compute precision (same as output tensor type).
         *
         * @param operation_type The type of the operation (from OperationType enum).
         * @param context The device context to use for this operation.
         */
        OperationBase( OperationType operation_type, std::shared_ptr<DeviceContext> context )
            : operation_type_( operation_type ), compute_precision_( ComputePrecision::Default ), device_context_( context ) {}

        /**
         * @brief Virtual destructor for the OperationBase class.
         *
         * Ensures proper cleanup of derived class resources when destroyed through
         * a base class pointer.
         */
        virtual ~OperationBase() = default;

        /**
         * @brief Gets the name of the operation.
         *
         * This pure virtual function must be implemented by derived classes to return
         * a unique identifier string for the specific operation type.
         *
         * @return std::string A unique name identifying this operation type.
         */
        virtual std::string getName() const = 0;

        /**
         * @brief Gets the device context associated with this operation.
         *
         * The device context contains information about the execution environment,
         * including the device, streams, and memory resources.
         *
         * @return std::shared_ptr<DeviceContext> The device context for this operation.
         */
        std::shared_ptr<DeviceContext> getDeviceContext() const {
            return device_context_;
        }

        /**
         * @brief Gets the device type for this operation.
         *
         * This is a convenience method that retrieves the device type from the
         * associated device context.
         *
         * @return DeviceType The type of device (CPU, CUDA, etc.) for this operation.
         */
        DeviceType getDeviceType() const {
            return device_context_->getDevice()->getDeviceType();
        }

        /**
         * @brief Gets the operation type enumeration value.
         *
         * Returns the operation type that was specified during construction.
         *
         * @return OperationType The enumeration value identifying this operation's category.
         */
        OperationType getOperationType() const {
            return operation_type_;
        }

        /**
         * @brief Gets the compute precision for this operation.
         *
         * Returns the precision that should be used for internal computations.
         *
         * @return ComputePrecision The enumeration value specifying the computation precision.
         */
        ComputePrecision getComputePrecision() const {
            return compute_precision_;
        }

    private:
        OperationType operation_type_; ///< The operation type identifier.
		ComputePrecision compute_precision_; ///< The precision for internal computation.
        std::shared_ptr<DeviceContext> device_context_; ///< The device context for execution.
    };
}