/**
 * @file CpuSoftmaxOp.ixx
 * @brief CPU implementation of Softmax operation (TensorDataType-based).
 *
 * Ported to the ExecutionContext / TensorDataType UnaryOperation interface
 * following the two-phase initialization pattern used by CpuLayerNormOp.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <math.h>
#include <cmath>
#ifdef USE_OMP
#include <omp.h>
#endif
#include <cstdint>
#include <limits>

export module Compute.CpuSoftmaxOp;

import Dnn.Components.Softmax;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
import Dnn.TensorPartitioning;
import Dnn.ComponentConfig;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.CpuExecutionContext;
import Compute.OperationType;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CpuTensorDataTypeTraits;
import Compute.CpuDevice;
import Compute.Precision;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief CPU implementation of Softmax using abstract TensorDataType API.
     *
     * Template parameter TPrecision selects the abstract tensor precision (e.g. FP32).
     * float is the corresponding CPU host representation for that precision.
     *
     * Design philosophy:
     * - Two-phase initialization: build() does all setup, forward()/backward() are pure dispatch
     * - Softmax has no trainable parameters (stateless operation)
     * - All dimension computation happens once in build()
     * - Forward/backward are hot-path methods with minimal overhead
     * - Uses numerically-stable softmax: exp(x - max(x))
     */
    export class CpuSoftmaxOp : public UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using MR = CpuMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorType = Tensor<TensorDataType::FP32, MR>;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;

        CpuSoftmaxOp( std::shared_ptr<CpuExecutionContext> context, const SoftmaxConfig& config )
            : context_( context ), config_( config )
        {
            if (!context_)
            {
                throw std::runtime_error( "CpuSoftmaxOp requires a CPU execution context" );
            }

            config_.validate();
        }

        // ====================================================================
        // Parameters
        // ====================================================================

        /**
         * @brief Set parameter tensor references (no-op for Softmax - stateless operation).
         *
         * Softmax has no trainable parameters, so this method validates that
         * the inputs are null and does nothing else.
         */
        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            if (weight != nullptr || bias != nullptr)
            {
                throw std::invalid_argument( "CpuSoftmaxOp::setParameters - Softmax has no trainable parameters" );
            }
        }

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Build the operation for a concrete input shape.
         *
         * This is the COLD PATH where all setup, validation, and computation happens ONCE.
         * After build() completes, forward() and backward() become pure dispatch methods.
         *
         * Responsibilities:
         *  1. Validate input shape compatibility
         *  2. Compute and cache normalized axis
         *  3. Compute and cache dimension sizes (outer, dim, inner)
         *  4. Cache OMP parallelization threshold
         *
         * After build(), the operation is ready for zero-overhead forward/backward dispatch.
         */
        void build( const shape_t& input_shape ) override
        {
            const auto& shape = input_shape;
            const int64_t ndim = static_cast<int64_t>(shape.size());

            if (ndim == 0)
            {
                throw std::invalid_argument( "CpuSoftmaxOp::build - input must have rank >= 1" );
            }

            int64_t axis = config_.getAxis();

            if (axis < 0)
                axis = ndim + axis;

            if (axis < 0 || axis >= ndim)
            {
                throw std::invalid_argument( "CpuSoftmaxOp::build - axis out of bounds" );
            }

            cached_axis_ = axis;

            int64_t outer_size = 1;
            for (int64_t i = 0; i < axis; ++i)
                outer_size *= shape[i];

            int64_t dim_size = shape[axis];

            int64_t inner_size = 1;
            for (int64_t i = axis + 1; i < ndim; ++i)
                inner_size *= shape[i];

            cached_outer_size_ = outer_size;
            cached_dim_size_ = dim_size;
            cached_inner_size_ = inner_size;

            enable_omp_ = (outer_size * inner_size > 100);

            UnaryOperationBase::build( input_shape );
        }

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to CPU kernel.
         *
         * All setup, validation, and dimension computation was done in build().
         * This method extracts raw pointers and dispatches directly to the
         * numerically-stable softmax kernel using pre-computed cached dimensions.
         *
         * Algorithm: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
         * Zero redundant work - maximum performance.
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const float* in_data = static_cast<const float*>(input.rawData());
            float* out_data = static_cast<float*>(output.rawData());

            const int64_t outer_size = cached_outer_size_;
            const int64_t dim_size = cached_dim_size_;
            const int64_t inner_size = cached_inner_size_;

#pragma omp parallel for collapse(2) if(enable_omp_)
            for (int64_t outer = 0; outer < outer_size; ++outer)
            {
                for (int64_t inner = 0; inner < inner_size; ++inner)
                {
                    const float* slice_in = in_data + (outer * dim_size * inner_size) + inner;
                    float* slice_out = out_data + (outer * dim_size * inner_size) + inner;

                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        float v = slice_in[i * inner_size];
                        if (v > max_val) max_val = v;
                    }

                    long double sum = 0.0L;
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        long double val = expl( static_cast<long double>( slice_in[i * inner_size] - max_val ) );
                        slice_out[i * inner_size] = static_cast<float>( val );
                        sum += val;
                    }

                    long double inv_sum = 1.0L / sum;
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        slice_out[i * inner_size] = static_cast<float>( static_cast<long double>( slice_out[i * inner_size] ) * inv_sum );
                    }
                }
            }
        }

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to CPU kernel.
         *
         * Similar to forward(), this method does minimal work and dispatches
         * directly to the backward kernel using cached dimensions from build().
         *
         * Algorithm: dX = Y * (dY - dot(Y, dY))
         * where Y is the softmax output (probabilities) and dY is the gradient.
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            const float* probs = static_cast<const float*>(input.rawData());
            const float* dY = static_cast<const float*>(output_grad.rawData());
            float* dX = static_cast<float*>(input_grad.rawData());

            const int64_t outer_size = cached_outer_size_;
            const int64_t dim_size = cached_dim_size_;
            const int64_t inner_size = cached_inner_size_;

#pragma omp parallel for collapse(2) if(enable_omp_)
            for (int64_t outer = 0; outer < outer_size; ++outer)
            {
                for (int64_t inner = 0; inner < inner_size; ++inner)
                {
                    const float* slice_p = probs + (outer * dim_size * inner_size) + inner;
                    const float* slice_dy = dY + (outer * dim_size * inner_size) + inner;
                    float* slice_dx = dX + (outer * dim_size * inner_size) + inner;

                    long double dot = 0.0L;
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        dot += static_cast<long double>( slice_p[i * inner_size] ) * static_cast<long double>( slice_dy[i * inner_size] );
                    }

                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        long double y = static_cast<long double>( slice_p[i * inner_size] );
                        long double grad = y * (static_cast<long double>( slice_dy[i * inner_size] ) - dot);
                        slice_dx[i * inner_size] += static_cast<float>( grad );
                    }
                }
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::SoftmaxOp;
        }

        std::string getName() const override
        {
            return "Cpu::SoftmaxOp";
        }

        const SoftmaxConfig& getConfig() const
        {
            return config_;
        }

    private:
        SoftmaxConfig config_;
        std::shared_ptr<CpuExecutionContext> context_;

        // Cached dimension values computed once in build() for hot-path dispatch
        int64_t cached_axis_{ -1 };
        int64_t cached_outer_size_{ 0 };
        int64_t cached_dim_size_{ 0 };
        int64_t cached_inner_size_{ 0 };
        bool enable_omp_{ false };
    };

    export class CpuSoftmaxOpRegistrar
    {
    public:
        static void registerOperations()
        {
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, TensorDataType::FP32, TensorDataType::FP32>(
                "SoftmaxOp",
                []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32, TensorDataType::FP32>>
                {
                    const auto& softmaxConfig = dynamic_cast<const SoftmaxConfig&>(config);
                    return std::make_shared<CpuSoftmaxOp>( context, softmaxConfig );
                }
            );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}