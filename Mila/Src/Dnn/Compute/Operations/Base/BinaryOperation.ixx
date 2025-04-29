/**
 * @file BinaryOperation.ixx
 * @brief Abstract base class for binary operations in the compute framework.
 *
 * This file defines the `BinaryOperation` class, which serves as an interface
 * for operations that take two input tensors and produce a single output tensor.
 * Derived classes must implement the `forward()` method to define the forward
 * computation for the operation.
 *
 * The `BinaryOperation` class is templated to support various data types for
 * input and output tensors, as well as different device types (e.g., CPU, CUDA).
 *
 * @tparam TInput The data type of the input tensor elements.
 * @tparam TDataType The data type of the output and computation tensor elements.
 * @tparam TDevice The device type (e.g., CPU, CUDA) on which the operation is executed.
 *
 * @see UnaryOperation
 * @see OperationBase
 */

module;
#include <memory>  
#include <vector>  
#include <type_traits>  
#include <stdexcept>

export module Compute.BinaryOperation;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.ComputeDevice;
import Compute.CudaDevice;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.DeviceContext;
import Compute.DeviceContextHelpers;
import Compute.DeviceType;
import Compute.CudaMemoryResource;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
    /**
    * @brief Abstract class for binary operations.
    *
    * @tparam TInput The data type of the input tensor elements.
    * @tparam TDataType The data type of the output and computation tensor elements.
    * @tparam TDevice The device type (e.g., CPU, CUDA) on which the operation is executed.
    */
    export template <typename TPrecision, typename TInput = TPrecision, DeviceType TDeviceType = DeviceType::Cuda>
        requires ValidFloatTensorType<TPrecision> && ValidTensorType<TInput>
    class BinaryOperation : public OperationBase<TPrecision, TInput, TDeviceType> {
    public:
        /**
        * @brief Memory resource type based on device type.
        *
        * This type alias automatically selects the appropriate memory resource type
        * based on the device type. For CUDA devices, it uses CudaMemoryResource;
        * for CPU devices, it uses HostMemoryResource.
        */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, HostMemoryResource>;

        /**
        * @brief Constructs a BinaryOperation with the specified operation type.
        * Creates a device context that matches the TDevice type.
        *
        * @param operation_type The type of the operation.
        */
        BinaryOperation( OperationType operation_type )
            : OperationBase<TPrecision, TInput, TDeviceType>( operation_type, CreateCompatibleContext<TDeviceType>() ) {}

        /**
         * @brief Constructs a BinaryOperation with the specified operation type and device context.
         * Validates that the context is compatible with TDevice.
         *
         * @param operation_type The type of the operation.
         * @param context The device context to use for this operation.
         * @throws std::runtime_error If the provided context is incompatible with TDevice.
         */
        BinaryOperation( OperationType operation_type, std::shared_ptr<DeviceContext> context )
            : OperationBase<TPrecision, TInput, TDeviceType>( operation_type, ValidateContext<TDeviceType>( context ) ) {}

        /**
        * @brief Virtual destructor for proper cleanup of derived classes.
        */
        virtual ~BinaryOperation() = default;

        /**
        * @brief Executes the forward pass of a binary operation.
        *
        * @param input1 The first input tensor.
        * @param input2 The second input tensor.
        * @param parameters The parameters for the operation.
        * @param attributes The attributes for the operation (if any).
        * @param output The output tensor.
        * @param output_state Cache for the output tensors.
        */
        virtual void forward(
            const Tensor<TInput, MR>& input1,
            const Tensor<TInput, MR>& input2,
            const std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& parameters,
            const OperationAttributes& attributes,
            Tensor<TPrecision, MR>& output,
            std::vector<std::shared_ptr<Tensor<TPrecision, MR>>>& output_state ) const = 0;
    };
}
