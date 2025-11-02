/**
 * @file OperationBase.ixx
 * @brief Core abstraction for neural network operations in the Mila framework
 */

module;
#include <string>
#include <memory>
#include <stdexcept>

export module Compute.OperationBase;

import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Compute.OperationType;

namespace Mila::Dnn::Compute
{
    export template <DeviceType TDeviceType, TensorDataType TPrecision>
        class OperationBase
    {
    public:
        static constexpr DeviceType device_type = TDeviceType;
        static constexpr TensorDataType data_type = TPrecision;
        using DataTypeTraits = TensorDataTypeTraits<TPrecision>;

        virtual ~OperationBase() = default;

        /**
         * @brief Check if operation has been built for a specific input shape.
         *
         * Operations may defer expensive setup (kernel compilation, workspace
         * allocation, algorithm selection) until input shapes are known.
         *
         * @return true if build() has completed successfully
         * @return false if operation requires build() before execution
         *
         * @see build()
         */
        virtual bool isBuilt() const
        {
            return is_built_;
        }

        /**
         * @brief Prepare operation for execution with known input shape(s).
         *
         * This method enables shape-dependent optimizations:
         * - Kernel compilation with template specialization for dimensions
         * - Workspace/scratch memory allocation
         * - Algorithm selection based on problem size
         * - Validation of shape compatibility
         * - Caching of dimension-dependent constants
         *
         * Default implementation is a no-op. Operations requiring build-time
         * setup should override this method.
         *
         * @param input_shapes Shape(s) of input tensor(s). Unary operations
         *                     receive single shape, binary operations receive
         *                     pair of shapes.
         *
         * @note Idempotent - safe to call multiple times
         * @note Must be called before forward() or backward() if isBuilt() is false
         * @note After build(), input shapes in forward/backward must match
         *
         * @throws std::invalid_argument if shapes are incompatible with operation
         * @throws std::runtime_error if compilation or allocation fails
         */
        virtual void build( const shape_t& input_shape )
        {
            // Default: no build required
            is_built_ = true;
        }

        /**
         * @brief Bind module-owned parameter tensors to the operation.
         *
         * - The Module remains the canonical owner of parameter tensors.
         * - The operation MUST NOT take ownership or free the provided pointers.
         * - Implementations may cache `rawData()` pointers derived from the provided
         *   `ITensor` during `build()` for hot-path use.
         *
         * Default: no-op for stateless operations.
         */
        virtual void setParameters( ITensor* weight, ITensor* bias )
        {
            (void)weight;
            (void)bias;
        }

        /**
         * @brief Gets the operation type identifier.
         */
        virtual OperationType getOperationType() const = 0;

        /**
         * @brief Gets the device type for this operation.
         */
        virtual DeviceType getDeviceType() const
        {
            return TDeviceType;
        }

        /**
         * @brief Gets the tensor data type for this operation.
         */
        virtual TensorDataType getDataType() const
        {
            return TPrecision;
        }

        /**
         * @brief Get the name of the operation.
         *
         * Used for logging, debugging, and naming associated tensors.
         *
         * @return std::string Name of the operation
		 */
        virtual std::string getName() const = 0;

    protected:
        
        bool is_built_{ false };
    };
}