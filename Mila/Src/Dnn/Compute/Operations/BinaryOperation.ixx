/**
 * @file BinaryOperation.ixx
 * @brief Abstract device-agnostic binary operation interface.
 *
 * Provides the typed abstract interface used by device-specific binary operation
 * implementations. See member documentation for API contracts and semantics.
 */

module;
#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <string>

export module Compute.BinaryOperation;

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
import Compute.CudaDeviceMemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDevice;
import Compute.OperationBase;
import Compute.OperationType;
import Compute.OperationAttributes;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Abstract interface for binary operations (two inputs -> one output).
     *
     * Implementations perform a device- and precision-specific computation that
     * consumes two inputs and produces a single output. Implementations should
     * document allocation and accumulation semantics (e.g., whether gradients
     * are accumulated or overwritten) and are free to use the provided
     * execution context or the device associated with input tensors.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision  Abstract tensor precision (TensorDataType)
     */
    export template <DeviceType TDeviceType, TensorDataType TPrecision>
    class BinaryOperation : public OperationBase<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;

        virtual ~BinaryOperation() = default;

        /**
         * @brief Forward pass.
         *
         * Compute output = f(inputA, inputB, parameters).
         *
         * Parameters:
         *  - inputA, inputB: Forward inputs (type-erased `ITensor`). Implementations
         *    should validate shapes and types as required.
         *  - parameters: Optional vector of parameter tensors (learnable weights,
         *    biases, gates, projection matrices). May be empty for stateless ops.
         *  - output: Output tensor to write results into. Implementations may resize
         *    or reallocate `output`. Callers must not rely on prior contents.
         *  - output_state: Optional container for any intermediate tensors that must
         *    be preserved for the backward pass (e.g., reduction indices, cached
         *    normalization factors). Implementations should push any required
         *    shared_ptr<Tensor> objects into this vector.
         *
         * Allocation / ownership notes:
         *  - Implementations decide whether to reuse, rebind or allocate device
         *    memory for `output` and `output_state` members.
         *  - Concrete operation documentation must state whether outputs are
         *    overwritten or accumulated.
         */
        virtual void forward(
            const ITensor& inputA,
            const ITensor& inputB,
            const Parameters& parameters,
            ITensor& output,
            OutputState& output_state ) const = 0;

        /**
         * @brief Backward pass (gradient computation).
         *
         * Compute gradients with respect to inputs and parameters using the
         * forward inputs/output and the provided output gradient.
         *
         * Parameters:
         *  - inputA, inputB: The forward inputs as provided to `forward`.
         *  - output: The forward output as produced by `forward`.
         *  - output_gradient: Gradient of the loss with respect to `output`.
         *  - parameters: Same parameter vector passed to `forward`.
         *  - parameter_gradients: Output vector for parameter gradients. Implementations
         *    should ensure `parameter_gradients.size()` matches `parameters.size()` and
         *    accumulate into entries (or assign, as documented by the concrete op).
         *  - inputA_gradient, inputB_gradient: Output tensors to receive gradients
         *    for the corresponding inputs. Implementations should document whether
         *    these are accumulated (added) or overwritten.
         *  - output_state: Forward-pass cached tensors required for gradient computation.
         *
         * Behavior and expectations:
         *  - Implementations must handle broadcasting/reduction semantics consistently
         *    with forward. When broadcasting was applied in forward, backward must
         *    reduce gradients appropriately.
         *  - Implementations should validate shapes and provide clear error messages
         *    on mismatch.
         *  - Thread-safety and accumulation semantics are implementation-specific;
         *    callers should follow the concrete operation documentation.
         */
            virtual void backward(
                const ITensor& inputA,
                const ITensor& inputB,
                const ITensor& output,
                const ITensor& output_gradient,
                const Parameters& parameters,
                Parameters& parameter_gradients,
                ITensor& inputA_gradient,
                ITensor& inputB_gradient,
                const OutputState& output_state ) const = 0;
    };
}