/**
 * @file OperationBase.ixx
 * @brief Core abstraction for neural network operations in the Mila framework.
 */

module;
#include <string>

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
    class Operation
    {
    public:
        static constexpr DeviceType device_type = TDeviceType;
        static constexpr TensorDataType data_type = TPrecision;
        using DataTypeTraits = TensorDataTypeTraits<TPrecision>;

        virtual ~Operation() = default;

        /**
         * @brief Whether build() completed successfully for a concrete input shape.
         */
        virtual bool isBuilt() const
        {
            return is_built_;
        }

        /**
         * @brief Prepare the operation for a concrete input shape.
         *
         * Default implementation is a no-op. Operations requiring shape-dependent
         * setup should override this method.
         */
        virtual void build( [[maybe_unused]] const shape_t& input_shape )
        {
			// Default: no build required by stateless operations
            is_built_ = true;
        }

        /**
         * @brief Bind module-owned parameter tensors to the operation.
         *
         * The module retains ownership of the provided ITensor objects. Implementations
         * may cache rawData() pointers for hot-path access but MUST NOT free the
         * provided pointers.
         *
         * Default: no-op for stateless operations.
         */
        virtual void setParameters( ITensor* weight, ITensor* bias )
        {
            (void)weight;
            (void)bias;
        }

        /**
         * @brief Bind module-owned gradient tensors to the operation.
         *
         * New canonical API for binding gradient buffers. Mirrors semantics of
         * `setParameters()` but for gradients used during backward().
         *
         * The operation MUST NOT take ownership of the provided pointers. Implementations
         * may cache rawData() pointers for hot-path writes.
         *
         * Default: no-op for stateless operations.
         */
        virtual void setGradients( ITensor* weight_grad, ITensor* bias_grad )
        {
            (void)weight_grad;
            (void)bias_grad;
        }

        /**
         * @brief Clear any cached gradient pointers held by the operation.
         *
         * Explicit unbind called by modules before freeing/resetting module-owned
         * gradient buffers. Implementations MUST null-out any cached raw pointers
         * and MUST NOT throw. Marked noexcept so it is safe to call from destructors
         * or during state transitions.
         */
        virtual void clearGradients() noexcept
        {
			// Default: no-op for stateless operations
        }

        /**
         * @brief Configure operation training-mode behavior.
         *
         * Implementations may use this to enable/disable training-specific work.
         */
        virtual void setTraining( bool is_training )
        {
            is_training_ = is_training;
        }

        /**
         * @brief Query whether operation is configured for training.
         */
        virtual bool isTraining() const
        {
            return is_training_;
        }

        /**
         * @brief Operation type identifier.
         */
        virtual OperationType getOperationType() const = 0;

        /**
         * @brief Device type for this operation.
         */
        virtual DeviceType getDeviceType() const
        {
            return TDeviceType;
        }

        /**
         * @brief Tensor data type for this operation.
         */
        virtual TensorDataType getDataType() const
        {
            return TPrecision;
        }

        /**
         * @brief Human-readable operation name.
         */
        virtual std::string getName() const = 0;

    protected:

        bool is_built_{ false };
        bool is_training_{ false };
    };
}