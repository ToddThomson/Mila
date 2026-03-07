/**
 * @file CudaRmsNormOp.ixx
 * @brief CUDA implementation of RMS Normalization operation.
 */

module;
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <sstream>
#include <cuda_fp16.h>
#include "Kernels/RmsNorm.cuh"

// DEBUG: Remove these when not needed.
#include <iostream> 

export module Compute.CudaRmsNormOp;
import :Dispatch;

import Dnn.Components.RmsNorm;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
import Dnn.TensorPartitioning;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.DeviceType;

// DEBUG:
import Compute.Device;
import Compute.DeviceId;
import Dnn.TensorOps;
import Compute.CpuMemoryResource;

import Compute.IExecutionContext;
import Compute.ExecutionContext;
//import Compute.CudaDeviceResources;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;
import Compute.Precision;
import Utils.Logger;

namespace Mila::Dnn::Compute::Cuda::RmsNorm
{
    using namespace Mila::Dnn;

    /**
     * @brief CUDA implementation of RMS Normalization.
     *
     * Normalizes activations using root-mean-square across a specified axis,
     * then applies an optional affine transform with learnable weight and bias.
     *
     * This class mirrors the LayerNorm op structure but calls RMS-specific kernels.
     *
     * @tparam TPrecision Abstract tensor precision (FP32, FP16, etc.)
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaRmsNormOp : public UnaryOperation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaRmsNormOp( IExecutionContext* context, const RmsNormConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaRmsNormOp" ) )
            , config_( config )
            , impl_()
        {
            config_.validate();
        }

        /**
         * @brief Bind component-owned parameter tensors.
         *
         * Caches native device pointers for zero-overhead hot-path access.
         * Weight is required; bias is optional based on configuration.
         *
         * @param weight Scaling parameter applied after normalization (required)
         * @param bias Shift parameter applied after normalization (optional)
         */
        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            if ( !weight )
            {
                throw std::invalid_argument( "CudaRmsNormOp::setParameters - weight parameter is required" );
            }

            if ( weight->getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaRmsNormOp::setParameters - weight must be a CUDA tensor" );
            }

            weight_ = static_cast<NativeType*>(weight->rawData());

            if ( config_.hasBias() )
            {
                if ( !bias )
                {
                    throw std::invalid_argument( "CudaRmsNormOp::setParameters - bias parameter expected but null was provided" );
                }

                if ( bias->getDeviceType() != DeviceType::Cuda )
                {
                    throw std::invalid_argument( "CudaRmsNormOp::setParameters - bias must be a CUDA tensor" );
                }

                bias_ = static_cast<NativeType*>(bias->rawData());
            }
            else
            {
                bias_ = nullptr;
            }
        }

        /**
         * @brief Bind component-owned parameter gradient tensors for training.
         *
         * Caches native device gradient pointers for backward pass writes.
         *
         * @param weight_grad Gradient accumulator for weight parameter (required)
         * @param bias_grad Gradient accumulator for bias parameter (optional)
         */
        void setGradients( ITensor* weight_grad, ITensor* bias_grad ) override
        {
            if ( !weight_grad )
            {
                throw std::invalid_argument( "CudaRmsNormOp::setGradients - weight gradient is required" );
            }

            if ( weight_grad->getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaRmsNormOp::setGradients - weight gradient must be a CUDA tensor" );
            }

            weight_grad_ = static_cast<NativeType*>(weight_grad->rawData());

            if ( config_.hasBias() )
            {
                if ( !bias_grad )
                {
                    throw std::invalid_argument( "CudaRmsNormOp::setGradients - bias gradient expected but null was provided" );
                }

                if ( bias_grad->getDeviceType() != DeviceType::Cuda )
                {
                    throw std::invalid_argument( "CudaRmsNormOp::setGradients - bias gradient must be a CUDA tensor" );
                }

                bias_grad_ = static_cast<NativeType*>(bias_grad->rawData());
            }
            else
            {
                bias_grad_ = nullptr;
            }
        }

        /**
         * @brief Prepare operation for execution with concrete input shape.
         *
         * Computes normalization axis, partitions tensor dimensions, and allocates
         * forward-pass statistics storage required by backward.
         */
        void build( const shape_t& input_shape ) override
        {
            if ( weight_ == nullptr )
            {
                throw std::runtime_error( "CudaRmsNormOp::build requires parameters bound via setParameters() before build()" );
            }

            if ( config_.hasBias() && bias_ == nullptr )
            {
                throw std::runtime_error( "CudaRmsNormOp::build - bias expected by config but not bound via setParameters()" );
            }

            if ( !config_.getNormalizedShape().empty() )
            {
                if ( input_shape.size() < config_.getNormalizedShape().size() )
                {
                    throw std::invalid_argument( "CudaRmsNormOp::build - input rank is less than normalized_shape rank" );
                }

                size_t offset = input_shape.size() - config_.getNormalizedShape().size();

                for ( size_t i = 0; i < config_.getNormalizedShape().size(); ++i )
                {
                    if ( input_shape[ offset + i ] != config_.getNormalizedShape()[ i ] )
                    {
                        throw std::invalid_argument( "CudaRmsNormOp::build - input trailing dimensions don't match normalized_shape" );
                    }
                }
            }
            else if ( !config_.getAxis().has_value() )
            {
                throw std::invalid_argument( "CudaRmsNormOp::build - configuration must specify normalized_shape or axis before build()" );
            }

            const auto& shape = input_shape;
            const int64_t ndim = static_cast<int64_t>(shape.size());

            int64_t axis = -1;

            if ( config_.getAxis().has_value() )
            {
                axis = config_.getAxis().value();
            }
            else
            {
                axis = static_cast<int64_t>(shape.size()) - static_cast<int64_t>(config_.getNormalizedShape().size());
            }

            if ( axis < 0 )
            {
                axis += ndim;
            }

            if ( axis < 0 || axis >= ndim )
            {
                throw std::invalid_argument( "CudaRmsNormOp::build - computed axis out of range" );
            }

            norm_axis_ = axis;

            int64_t outer = 1;
            for ( int64_t i = 0; i < axis; ++i )
            {
                outer *= static_cast<int64_t>( shape[ i ] );
            }

            int64_t inner = 1;
            for ( int64_t i = axis + 1; i < ndim; ++i )
            {
                inner *= static_cast<int64_t>( shape[ i ] );
            }

            const int64_t dim = static_cast<int64_t>( shape[ axis ] );
            const int64_t num_slices = outer * inner;

            outer_size_ = static_cast<int>( outer );
            inner_size_ = static_cast<int>( inner );
            norm_dim_ = static_cast<int>( dim );

            auto device = context_->getDeviceId();

            rstd_tensor_ = std::make_shared<TensorType>( device, shape_t{ num_slices } );
            rstd_tensor_->setName( "rstd" );
            rstd_ = static_cast<NativeType*>(rstd_tensor_->rawData());

            UnaryOperationBase::build( input_shape );
        }

        /**
         * @brief Execute forward pass (hot path).
         *
         * Computes RMS-normalized output and caches forward-pass statistics required for backward().
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            cudaStream_t stream = context_->getStream();

            Detail::cuda_rmsnorm_impl<NativeType>::forward(
                Y, X,
                weight_, bias_,
                rstd_,
                outer_size_, inner_size_, norm_dim_,
                config_.getEpsilon(),
                stream );

            context_->synchronize();
        }

        /**
         * @brief Execute backward pass (hot path).
         *
         * Computes input gradient and accumulates parameter gradients using
         * forward-pass statistics cached during forward().
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* dX = static_cast<NativeType*>(input_grad.rawData());

            cudaStream_t stream = context_->getStream();

            Detail::cuda_rmsnorm_impl<NativeType>::backward(
                dX, weight_grad_, bias_grad_,
                dY, X, weight_,
                rstd_,
                outer_size_, inner_size_, norm_dim_,
                stream );
        }

        OperationType getOperationType() const override
        {
            return OperationType::RmsNormOp;
        }

        std::string getName() const override
        {
            return "Cuda::RmsNormOp";
        }

        const RmsNormConfig& getConfig() const
        {
            return config_;
        }

    private:
        RmsNormConfig config_;
        CudaExecutionContext* context_;
        Detail::cuda_rmsnorm_impl<NativeType> impl_;

        NativeType* weight_{ nullptr };
        NativeType* bias_{ nullptr };
        NativeType* weight_grad_{ nullptr };
        NativeType* bias_grad_{ nullptr };

        std::shared_ptr<TensorType> rstd_tensor_;
        NativeType* rstd_{ nullptr };

        int64_t norm_axis_{ -1 };
        int outer_size_{ 0 };
        int inner_size_{ 0 };
        int norm_dim_{ 0 };
    };

    export class CudaRmsNormOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "RmsNormOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& rnConfig = static_cast<const RmsNormConfig&>(config);
                    return std::make_shared<CudaRmsNormOp<TensorDataType::FP32>>( context, rnConfig );
                } );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16, TensorDataType::FP16>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                {
                    const auto& rnConfig = static_cast<const RmsNormConfig&>(config);
                    return std::make_shared<CudaRmsNormOp<TensorDataType::FP16>>( context, rnConfig );
                } );
        }
    };
}