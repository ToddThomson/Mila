/**
 * @file CudaSoftmaxOp.ixx
 * @brief CUDA implementation of Softmax operation (TensorDataType-based).
 */

module;
#include <cuda_fp16.h>
#include "Kernels/Softmax.cuh"

export module Compute.CudaSoftmaxOp;

import std;
import Dnn.Components.SoftmaxConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.OperationType;
import Dnn.Component;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute::Cuda::Softmax
{
    using namespace Mila::Dnn;

    /**
     * @brief Namespace for CUDA softmax implementation details.
     *
     * Contains specialized templates for different data types (float, half)
     * and kernel dispatch logic.
     */
    namespace Detail
    {
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_softmax_impl;

        template <>
        struct cuda_softmax_impl<float>
        {
            cuda_softmax_impl() = default;

            static inline void forward_optimized( float* Y, const float* X, int N, int C, cudaStream_t stream )
            {
                cuda_softmax_forward<float>( Y, X, N, C, stream );
            }

            static inline void forward_general(
                float* Y, const float* X,
                int outer_size, int dim_size, int inner_size,
                cudaStream_t stream )
            {
                cuda_softmax_forward_general<float>( Y, X, outer_size, dim_size, inner_size, stream );
            }

            static inline void backward_optimized(
                float* dX, const float* dY, const float* Y, int N, int C, cudaStream_t stream )
            {
                cuda_softmax_backward<float>( dX, dY, Y, N, C, stream );
            }

            static inline void backward_general(
                float* dX, const float* dY, const float* Y,
                int outer_size, int dim_size, int inner_size,
                cudaStream_t stream )
            {
                cuda_softmax_backward_general<float>( dX, dY, Y, outer_size, dim_size, inner_size, stream );
            }
        };

        // REVIEW: The FP16 implementations need to be refactored to use bf16, the Mila half type standard

        template <>
        struct cuda_softmax_impl<half>
        {
            cuda_softmax_impl() = default;

            static inline void forward_optimized( half* Y, const half* X, int N, int C, cudaStream_t stream )
            {
                cuda_softmax_forward<half>( Y, X, N, C, stream );
            }

            static inline void forward_general(
                half* Y, const half* X,
                int outer_size, int dim_size, int inner_size,
                cudaStream_t stream )
            {
                cuda_softmax_forward_general<half>( Y, X, outer_size, dim_size, inner_size, stream );
            }

            static inline void backward_optimized(
                half* dX, const half* dY, const half* Y, int N, int C, cudaStream_t stream )
            {
                cuda_softmax_backward<half>( dX, dY, Y, N, C, stream );
            }

            static inline void backward_general(
                half* dX, const half* dY, const half* Y,
                int outer_size, int dim_size, int inner_size,
                cudaStream_t stream )
            {
                cuda_softmax_backward_general<half>( dX, dY, Y, outer_size, dim_size, inner_size, stream );
            }
        };
    }

    /**
     * @brief CUDA implementation of Softmax operation using abstract TensorDataType API.
     *
     * Template parameter TPrecision selects the abstract tensor precision (e.g. FP32, FP16).
     * NativeType is the corresponding CUDA device representation for that precision.
     *
     * Design philosophy:
     * - Two-phase initialization: build() does all setup, forward()/backward() are pure dispatch
     * - Softmax has no trainable parameters (stateless operation)
     * - All dimension computation and kernel selection happens once in build()
     * - Forward/backward are hot-path methods with minimal overhead
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaSoftmaxOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using OperationBaseType = Operation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaSoftmaxOp( IExecutionContext* context, const SoftmaxConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaSoftmaxOp" ) ), config_( config ), impl_()
        {
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
            // Softmax is stateless - no parameters
            if (weight != nullptr || bias != nullptr)
            {
                throw std::invalid_argument( "CudaSoftmaxOp::setParameters - Softmax has no trainable parameters" );
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
         *  4. Determine optimal kernel variant (optimized vs general)
         *  5. Cache kernel selection flag for hot-path dispatch
         *
         * After build(), the operation is ready for zero-overhead forward/backward dispatch.
         */
        void build( const BuildContext& config ) override
        {
            const auto& input_shape = config.inputShape();
            const auto& shape = input_shape;
            const std::int64_t ndim = static_cast<std::int64_t>(shape.size());

            if (ndim == 0)
            {
                throw std::invalid_argument( "CudaSoftmaxOp::build - input must have rank >= 1" );
            }

            std::int64_t axis = config_.getAxis();

            if (axis < 0)
                axis = ndim + axis;

            if (axis < 0 || axis >= ndim)
            {
                throw std::invalid_argument( "CudaSoftmaxOp::build - axis out of bounds" );
            }

            cached_axis_ = axis;

            std::int64_t outer_size = 1;
            for (std::int64_t i = 0; i < axis; ++i)
                outer_size *= shape[i];

            std::int64_t dim_size = shape[axis];

            std::int64_t inner_size = 1;
            for (std::int64_t i = axis + 1; i < ndim; ++i)
                inner_size *= shape[i];

            cached_outer_size_ = static_cast<int>( outer_size );
            cached_dim_size_ = static_cast<int>( dim_size );
            cached_inner_size_ = static_cast<int>( inner_size );

            use_optimized_kernel_ = (axis == ndim - 1 || inner_size == 1);

            OperationBaseType::build( config );
        }

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to CUDA kernel.
         *
         * All setup, validation, and dimension computation was done in build().
         * This method extracts raw pointers and dispatches directly to the appropriate
         * kernel variant using pre-computed cached dimensions and kernel selection.
         *
         * Zero redundant work - maximum performance.
         */
        void forward( const ITensor& input, ITensor& output ) const
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            if (use_optimized_kernel_)
            {
                Detail::cuda_softmax_impl<NativeType>::forward_optimized(
                    Y, X,
                    cached_outer_size_,
                    cached_dim_size_,
                    stream
                );
            }
            else
            {
                Detail::cuda_softmax_impl<NativeType>::forward_general(
                    Y, X,
                    cached_outer_size_,
                    cached_dim_size_,
                    cached_inner_size_,
                    stream
                );
            }
        }

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to CUDA kernel.
         *
         * Writes dX = Y * (dY - dot(Y, dY)) along the softmax axis, overwriting @p input_grad.
         *
         * @param output The forward output Y, not the forward input.
         * @param output_grad Gradient with respect to Y.
         * @param input_grad Receives the gradient with respect to the forward input.
         */
        void backward(
            const ITensor& output,
            const ITensor& output_grad,
            ITensor& input_grad ) const
        {
            const NativeType* Y = static_cast<const NativeType*>(output.rawData());
            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* dX = static_cast<NativeType*>(input_grad.rawData());

            cudaStream_t stream = context_->getStream();

            if (use_optimized_kernel_)
            {
                Detail::cuda_softmax_impl<NativeType>::backward_optimized(
                    dX, dY, Y,
                    cached_outer_size_,
                    cached_dim_size_,
                    stream
                );
            }
            else
            {
                Detail::cuda_softmax_impl<NativeType>::backward_general(
                    dX, dY, Y,
                    cached_outer_size_,
                    cached_dim_size_,
                    cached_inner_size_,
                    stream
                );
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::SoftmaxOp;
        }

        std::string getName() const override
        {
            return "Cuda::SoftmaxOp";
        }

        const SoftmaxConfig& getConfig() const
        {
            return config_;
        }

    private:

        SoftmaxConfig config_;
        CudaExecutionContext* context_;
        Detail::cuda_softmax_impl<NativeType> impl_;

        // Cached dimension values computed once in build() for hot-path dispatch
        std::int64_t cached_axis_{ -1 };
        int cached_outer_size_{ 0 };   // product of dimensions before axis
        int cached_dim_size_{ 0 };     // size of the softmax axis dimension
        int cached_inner_size_{ 0 };   // product of dimensions after axis
        bool use_optimized_kernel_{ false };  // kernel variant selection
    };
}
