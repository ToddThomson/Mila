/**
 * @file UnaryOperation.ixx
 * @brief Device-agnostic unary operation interface using abstract tensor data types
 *
 * This module provides a framework for implementing unary operations (single input,
 * single output) across heterogeneous compute environments using abstract TensorDataType
 * enumeration. The system enables seamless operation across different devices (CPU, CUDA)
 * without exposing device-specific concrete types to host compilation.
 *
 * Key architectural features:
 * - Abstract data type system prevents device-specific compilation dependencies
 * - Device-agnostic operation interface with compile-time device validation
 * - Type-safe parameter handling with automatic compatibility checking
 * - Support for scalar tensor operations (activations, reductions, etc.)
 * - Extensible design for various neural network operations
 */

module;
#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <string>

export module Compute.UnaryOperation;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.DeviceTypeTraits.Cpu;
import Compute.DeviceContext;
import Compute.DeviceContextHelpers;
import Compute.CudaDeviceMemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDevice;
import Compute.OperationBase;
import Compute.OperationType;
import Compute.OperationAttributes;

namespace Mila::Dnn::Compute
{
    export template <DeviceType TDeviceType, TensorDataType TPrecision>
    class UnaryOperation : public OperationBase<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;

        virtual ~UnaryOperation() = default;

        /**
         * @brief Forward pass computation.
         *
         * Concrete implementations extract execution resources from either:
         * - A bound execution context (if provided at construction), or
         * - The input tensor's device (for unbound operations)
         */
        virtual void forward(
            const ITensor& input,
            const Parameters& parameters,
            ITensor& output,
            OutputState& output_state ) const = 0;

        /**
         * @brief Backward pass computation.
         */
        virtual void backward(
            const ITensor& grad_output,
            const ITensor& input,
            const Parameters& parameters,
            const OutputState& output_state,
            ITensor& grad_input,
            Parameters& grad_parameters ) const = 0;
    };
}