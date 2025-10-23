/**
 * @file CudaSoftmaxOp.ixx
 * @brief Implementation of the CUDA-based softmax operation for neural networks.
 */

module;
#include <vector>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include "Kernels/CudaOps.h"

export module Compute.CudaSoftmaxOp;

import Dnn.Modules.Softmax;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ConfigurationBase;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CudaExecutionContext;
import Compute.CudaDeviceResources;
import Compute.OperationType;
import Compute.OperationAttributes;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief Namespace for CUDA softmax implementation details.
     *
     * This namespace contains the implementation details for the CUDA softmax operation,
     * including specialized templates for different data types (float, half).
     */
    namespace Detail
    {
        // Primary template - will cause a compile error if no specialization exists
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_softmax_impl;

        // Specialization for float
        template <>
        struct cuda_softmax_impl<float>
        {
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
        };

        // Specialization for half
        template <>
        struct cuda_softmax_impl<half>
        {
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
        };
    }

    /**
     * @brief CUDA implementation of the softmax operation for neural networks.
     *
     * Uses the abstract TensorDataType enum for precision and an ExecutionContext for CUDA resources.
     */
    export template<TensorDataType TPrecision>
        requires ValidFloatTensorDataType<TPrecision>
    class CudaSoftmaxOp : public UnaryOperation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaSoftmaxOp( std::shared_ptr<CudaExecutionContext> context, const SoftmaxConfig& config )
            : context_( context ), config_( config )
        {
            if (!context_)
            {
                throw std::invalid_argument( "CudaExecutionContext cannot be null." );
            }
            config_.validate();
        }

        void forward(
            const ITensor& input,
            const Parameters& parameters,
            ITensor& output,
            OutputState& output_state ) const override
        {
            // Validate device placement
            if (input.getDeviceType() != DeviceType::Cuda || output.getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaSoftmaxOp: Input and output tensors must be on CUDA device." );
            }

            if (input.size() != output.size())
            {
                throw std::invalid_argument( "CudaSoftmaxOp: Input and output tensors must have the same size." );
            }

            // Access native data pointers
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            // Axis handling
            int64_t axis = config_.getAxis();

            const auto& shape = input.shape();
            const int64_t ndim = static_cast<int64_t>(shape.size());
            if (axis < 0) axis = ndim + axis;
            if (axis < 0 || axis >= ndim)
            {
                throw std::runtime_error( "Softmax axis out of bounds" );
            }

            int64_t outer_size = 1;
            for (int64_t i = 0; i < axis; ++i) outer_size *= shape[i];

            int64_t dim_size = shape[axis];

            int64_t inner_size = 1;
            for (int64_t i = axis + 1; i < ndim; ++i) inner_size *= shape[i];

            cudaStream_t stream = context_->getStream();

            if (axis == ndim - 1 || inner_size == 1)
            {
                int N = static_cast<int>( outer_size );
                int C = static_cast<int>( dim_size );
                Detail::cuda_softmax_impl<NativeType>::forward_optimized( Y, X, N, C, stream );
            }
            else
            {
                Detail::cuda_softmax_impl<NativeType>::forward_general(
                    Y, X,
                    static_cast<int>(outer_size),
                    static_cast<int>(dim_size),
                    static_cast<int>(inner_size),
                    stream );
            }
        }

        void backward(
            const ITensor& grad_output,
            const ITensor& input,
            const Parameters& parameters,
            const OutputState& output_state,
            ITensor& grad_input,
            Parameters& grad_parameters ) const override
        {
            // Validate device placement
            if (grad_output.getDeviceType() != DeviceType::Cuda || input.getDeviceType() != DeviceType::Cuda || grad_input.getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaSoftmaxOp::backward: tensors must be on CUDA device." );
            }

            const NativeType* Y = static_cast<const NativeType*>(input.rawData()); // note: input contains logits; depending on convention adjust as needed
            const NativeType* dY = static_cast<const NativeType*>(grad_output.rawData());
            NativeType* dX = static_cast<NativeType*>(grad_input.rawData());

            int N = static_cast<int>(input.size());
            int axis = config_.getAxis();

            cudaStream_t stream = context_->getStream();

            // Kernel expected to be implemented in Kernels/CudaOps.h / .cu
            // Signature: cuda_softmax_backward( dX, dY, Y, N, axis, stream );
            //cuda_softmax_backward( dX, dY, Y, N, axis, stream );
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
        std::shared_ptr<CudaExecutionContext> context_;
    };

    /**
     * @brief Class responsible for registering the CudaSoftmaxOp operation.
     */
    export class CudaSoftmaxOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "SoftmaxOp";

            // FP32
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& softmaxConfig = static_cast<const SoftmaxConfig&>(config);
                    return std::make_shared<CudaSoftmaxOp<TensorDataType::FP32>>( context, softmaxConfig );
                }
            );

            // FP16
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16>(
                opName,
                []( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                {
                    const auto& softmaxConfig = static_cast<const SoftmaxConfig&>(config);
                    return std::make_shared<CudaSoftmaxOp<TensorDataType::FP16>>( context, softmaxConfig );
                }
            );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}