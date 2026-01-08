/**
 * @file CudaLayerNormOp.ixx
 * @brief CUDA implementation of Layer Normalization operation.
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
#include "Kernels/LayerNorm.cuh"

#include <iostream> // DEBUG

export module Compute.CudaLayerNormOp;

import Dnn.Components.LayerNorm;
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
import Compute.CudaDeviceResources;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;
import Compute.Precision;
import Utils.Logger;

namespace Mila::Dnn::Compute::Cuda::LayerNorm
{
    namespace Detail
    {
        /**
         * @brief CUDA kernel dispatcher for LayerNorm operations.
         *
         * Specialized for float (FP32) and half (FP16) native CUDA types.
         */
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_layernorm_impl;

        template <>
        struct cuda_layernorm_impl<float>
        {
            cuda_layernorm_impl() = default;

            static inline void forward(
                float* Y, const float* X,
                const float* weight, const float* bias,
                float* mean, float* rstd,
                int outer_size, int inner_size, int norm_dim,
                float epsilon,
                cudaStream_t stream )
            {
                cuda_layernorm_forward_fp32( Y, mean, rstd, X, weight, bias, outer_size, inner_size, norm_dim, epsilon, stream );
            }

            static inline void backward(
                float* dX, float* dweight, float* dbias,
                const float* dY, const float* X, const float* weight,
                const float* mean, const float* rstd,
                int outer_size, int inner_size, int norm_dim,
                cudaStream_t stream )
            {
                cuda_layernorm_backward_fp32( dX, dweight, dbias, dY, X, weight, mean, rstd, outer_size, inner_size, norm_dim, stream );
            }
        };

        template <>
        struct cuda_layernorm_impl<half>
        {
            cuda_layernorm_impl() = default;

            static inline void forward(
                half* Y, const half* X,
                const half* weight, const half* bias,
                half* mean, half* rstd,
                int outer_size, int inner_size, int norm_dim,
                float epsilon,
                cudaStream_t stream )
            {
                cuda_layernorm_forward_fp16( Y, mean, rstd, X, weight, bias, outer_size, inner_size, norm_dim, epsilon, stream );
            }

            static inline void backward(
                half* dX, half* dweight, half* dbias,
                const half* dY, const half* X, const half* weight,
                const half* mean, const half* rstd,
                int outer_size, int inner_size, int norm_dim,
                cudaStream_t stream )
            {
                cuda_layernorm_backward_fp16( dX, dweight, dbias, dY, X, weight, mean, rstd, outer_size, inner_size, norm_dim, stream );
            }
        };
    }

    using namespace Mila::Dnn;

    /**
     * @brief CUDA implementation of Layer Normalization.
     *
     * Normalizes activations along a specified axis by computing mean and variance,
     * then applying an affine transformation with learnable weight and bias parameters.
     *
     * Design philosophy:
     * - Two-phase initialization: build() allocates resources, forward()/backward() dispatch to kernels
     * - Component owns weight/bias parameters, operation caches device pointers
     * - Operation owns ephemeral forward-pass statistics (mean/rstd) required for backward
     * - All dimension computation happens once in build() for zero-overhead hot-path execution
     *
     * @tparam TPrecision Abstract tensor precision (FP32, FP16, etc.)
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaLayerNormOp : public UnaryOperation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaLayerNormOp( IExecutionContext* context, const LayerNormConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaLayerNormOp" ) )
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
         *
         * @throws std::invalid_argument If weight is null or not a CUDA tensor
         * @throws std::invalid_argument If bias is required by config but null or not a CUDA tensor
         *
         * @note Must be called before build()
         */
        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            if ( !weight )
            {
                throw std::invalid_argument( "CudaLayerNormOp::setParameters - weight parameter is required" );
            }

            if ( weight->getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaLayerNormOp::setParameters - weight must be a CUDA tensor" );
            }

            weight_ = static_cast<NativeType*>( weight->rawData() );

            if ( config_.hasBias() )
            {
                if ( !bias )
                {
                    throw std::invalid_argument( "CudaLayerNormOp::setParameters - bias parameter expected but null was provided" );
                }

                if ( bias->getDeviceType() != DeviceType::Cuda )
                {
                    throw std::invalid_argument( "CudaLayerNormOp::setParameters - bias must be a CUDA tensor" );
                }

                bias_ = static_cast<NativeType*>( bias->rawData() );
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
         * Weight gradient is required; bias gradient is optional based on configuration.
         *
         * @param weight_grad Gradient accumulator for weight parameter (required)
         * @param bias_grad Gradient accumulator for bias parameter (optional)
         *
         * @throws std::invalid_argument If weight_grad is null or not a CUDA tensor
         * @throws std::invalid_argument If bias_grad is required by config but null or not a CUDA tensor
         *
         * @note Must be called before training (backward pass)
         */
        void setGradients( ITensor* weight_grad, ITensor* bias_grad ) override
        {
            if ( !weight_grad )
            {
                throw std::invalid_argument( "CudaLayerNormOp::setGradients - weight gradient is required" );
            }

            if ( weight_grad->getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaLayerNormOp::setGradients - weight gradient must be a CUDA tensor" );
            }

            weight_grad_ = static_cast<NativeType*>( weight_grad->rawData() );

            if ( config_.hasBias() )
            {
                if ( !bias_grad )
                {
                    throw std::invalid_argument( "CudaLayerNormOp::setGradients - bias gradient expected but null was provided" );
                }

                if ( bias_grad->getDeviceType() != DeviceType::Cuda )
                {
                    throw std::invalid_argument( "CudaLayerNormOp::setGradients - bias gradient must be a CUDA tensor" );
                }

                bias_grad_ = static_cast<NativeType*>( bias_grad->rawData() );
            }
            else
            {
                bias_grad_ = nullptr;
            }
        }

        /**
         * @brief Prepare operation for execution with concrete input shape.
         *
         * Cold-path initialization: computes normalization axis, partitions tensor dimensions,
         * and allocates forward-pass statistics storage.
         *
         * Dimension partitioning:
         * - norm_axis: The axis along which normalization is applied
         * - outer_size: Product of all dimensions before norm_axis
         * - inner_size: Product of all dimensions after norm_axis
         * - norm_dim: Size of the dimension at norm_axis
         *
         * Example: For shape [2, 3, 4, 5] with axis=2:
         * - norm_axis = 2
         * - outer_size = 2 * 3 = 6
         * - inner_size = 5
         * - norm_dim = 4
         *
         * Forward-pass statistics (mean, rstd) are allocated with size outer_size * inner_size
         * to store one mean/rstd value per normalized slice.
         *
         * @param input_shape Shape of input tensor to be normalized
         *
         * @throws std::runtime_error If parameters not bound via setParameters()
         * @throws std::invalid_argument If input shape incompatible with configuration
         * @throws std::invalid_argument If computed normalization axis is out of range
         *
         * @note After build(), forward() and backward() become pure dispatch with zero overhead
         */
        void build( const shape_t& input_shape ) override
        {
            if ( weight_ == nullptr )
            {
                throw std::runtime_error( "CudaLayerNormOp::build requires parameters bound via setParameters() before build()" );
            }

            if ( config_.hasBias() && bias_ == nullptr )
            {
                throw std::runtime_error( "CudaLayerNormOp::build - bias expected by config but not bound via setParameters()" );
            }

            if ( !config_.getNormalizedShape().empty() )
            {
                if ( input_shape.size() < config_.getNormalizedShape().size() )
                {
                    throw std::invalid_argument( "CudaLayerNormOp::build - input rank is less than normalized_shape rank" );
                }

                size_t offset = input_shape.size() - config_.getNormalizedShape().size();

                for ( size_t i = 0; i < config_.getNormalizedShape().size(); ++i )
                {
                    if ( input_shape[ offset + i ] != config_.getNormalizedShape()[ i ] )
                    {
                        throw std::invalid_argument( "CudaLayerNormOp::build - input trailing dimensions don't match normalized_shape" );
                    }
                }
            }
            else if ( !config_.getAxis().has_value() )
            {
                throw std::invalid_argument( "CudaLayerNormOp::build - configuration must specify normalized_shape or axis before build()" );
            }

            const auto& shape = input_shape;
            const int64_t ndim = static_cast<int64_t>( shape.size() );

            int64_t axis = -1;

            if ( config_.getAxis().has_value() )
            {
                axis = config_.getAxis().value();
            }
            else
            {
                axis = static_cast<int64_t>( shape.size() ) - static_cast<int64_t>( config_.getNormalizedShape().size() );
            }

            if ( axis < 0 )
            {
                axis += ndim;
            }

            if ( axis < 0 || axis >= ndim )
            {
                throw std::invalid_argument( "CudaLayerNormOp::build - computed axis out of range" );
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

            mean_tensor_ = std::make_shared<TensorType>( device, shape_t{ num_slices } );
            mean_tensor_->setName( "mean" );
            mean_ = static_cast<NativeType*>( mean_tensor_->rawData() );

            rstd_tensor_ = std::make_shared<TensorType>( device, shape_t{ num_slices } );
            rstd_tensor_->setName( "rstd" );
            rstd_ = static_cast<NativeType*>( rstd_tensor_->rawData() );

            UnaryOperationBase::build( input_shape );
        }

        /**
         * @brief Execute forward pass (hot path).
         *
         * Computes normalized output and caches forward-pass statistics (mean, rstd)
         * required for backward gradient computation.
         *
         * @param input Input tensor to normalize
         * @param output Normalized output tensor (same shape as input)
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const NativeType* X = static_cast<const NativeType*>( input.rawData() );
            NativeType* Y = static_cast<NativeType*>( output.rawData() );

            cudaStream_t stream = context_->getStream();

            Detail::cuda_layernorm_impl<NativeType>::forward(
                Y, X,
                weight_, bias_,
                mean_, rstd_,
                outer_size_, inner_size_, norm_dim_,
                config_.getEpsilon(),
                stream );

            context_->synchronize();

        #ifndef NDEBUG
            // Debug: validate LayerNorm output for NaN/Inf or large magnitudes.
            //try
            //{
            //    // Ensure device work is visible to host
            //    context_->synchronize();

            //    Tensor<TensorDataType::FP32, CpuMemoryResource> host_out( Device::Cpu(), output.shape() );
            //    host_out.setName( this->getName() + ".forward_output.host_copy" );

            //    // perform ordered copy D2H using the component's execution context
            //    copy( static_cast<const TensorType&>(output), host_out /* this->getExecutionContext() */);

            //    const auto* ptr = host_out.data();
            //    size_t n = host_out.size();

            //    double sum_abs = 0.0;
            //    double max_abs = 0.0;
            //    size_t nan_ct = 0;
            //    for ( size_t i = 0; i < n; ++i )
            //    {
            //        double v = static_cast<double>( ptr[ i ] );
            //        if ( !std::isfinite( v ) ) ++nan_ct;
            //        double av = std::abs( v );
            //        sum_abs += av;
            //        if ( av > max_abs ) max_abs = av;
            //    }

            //    double mean_abs = n ? (sum_abs / static_cast<double>(n)) : 0.0;

            //    constexpr double kLnForwardLimit = 100.0; // debug threshold
            //    if ( nan_ct != 0 || max_abs > kLnForwardLimit || !std::isfinite( mean_abs ) )
            //    {
            //        std::clog << this->getName() << " [DEBUG] LayerNorm forward anomaly: mean_abs="
            //            << std::scientific << mean_abs << " max_abs=" << max_abs
            //            << " nan_count=" << nan_ct << " total=" << n << std::defaultfloat << std::endl;

            //        // print small sample (first channel slice b=0,t=0..min(16,channels))
            //        size_t channels = host_out.shape().back();
            //        size_t base = 0;
            //        std::clog << this->getName() << " [DEBUG] sample first 16 elements: ";
            //        for ( size_t i = 0; i < std::min<size_t>( channels, 16 ); ++i )
            //        {
            //            std::clog << std::scientific << static_cast<double>( ptr[ base + i ] ) << " ";
            //        }
            //        std::clog << std::defaultfloat << std::endl;

            //        // Suggest calling the heavier diagnostic from caller (Transformer) or throw to abort here:
            //        // throw std::runtime_error( "LayerNorm forward produced suspicious values - diagnostics logged" );
            //    }
            //}
            //catch ( const std::exception& e )
            //{
            //    std::cerr << this->getName() << " [DEBUG] LayerNorm forward diagnostic failed: " << e.what() << std::endl;
            //}
        #endif

            //context_->synchronize();
            //{
            //    using HostTensorType = Tensor<TensorDataType::FP32, CpuMemoryResource>;

            //    HostTensorType host_output( Device::Cpu(), output.shape() );
            //    host_output.setName( this->getName() + ".dbg.output" );
            //    copy( static_cast<const TensorType&>(output), host_output );
            //    Utils::Logger::info( this->getName() + ": dbg.output:\n" + host_output.toString( true ) );
            //}
        }

        /**
         * @brief Execute backward pass (hot path).
         *
         * Computes input gradient and accumulates parameter gradients using
         * forward-pass statistics cached during forward().
         *
         * @param input Original forward-pass input (required for gradient computation)
         * @param output_grad Gradient of loss with respect to output
         * @param input_grad Gradient of loss with respect to input (computed)
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            const NativeType* X = static_cast<const NativeType*>( input.rawData() );
            const NativeType* dY = static_cast<const NativeType*>( output_grad.rawData() );
            NativeType* dX = static_cast<NativeType*>( input_grad.rawData() );

            cudaStream_t stream = context_->getStream();

            Detail::cuda_layernorm_impl<NativeType>::backward(
                dX, weight_grad_, bias_grad_,
                dY, X, weight_,
                mean_, rstd_,
                outer_size_, inner_size_, norm_dim_,
                stream );
        }

        OperationType getOperationType() const override
        {
            return OperationType::LayerNormOp;
        }

        std::string getName() const override
        {
            return "Cuda::LayerNormOp";
        }

        const LayerNormConfig& getConfig() const
        {
            return config_;
        }

    private:
        LayerNormConfig config_;
        CudaExecutionContext* context_;
        Detail::cuda_layernorm_impl<NativeType> impl_;

        NativeType* weight_{ nullptr };
        NativeType* bias_{ nullptr };
        NativeType* weight_grad_{ nullptr };
        NativeType* bias_grad_{ nullptr };

        std::shared_ptr<TensorType> mean_tensor_;
        std::shared_ptr<TensorType> rstd_tensor_;
        NativeType* mean_{ nullptr };
        NativeType* rstd_{ nullptr };

        int64_t norm_axis_{ -1 };
        int outer_size_{ 0 };
        int inner_size_{ 0 };
        int norm_dim_{ 0 };
    };

    export class CudaLayerNormOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string opName = "LayerNormOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& lnConfig = static_cast<const LayerNormConfig&>( config );
                    return std::make_shared<CudaLayerNormOp<TensorDataType::FP32>>( context, lnConfig );
                } );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16, TensorDataType::FP16>(
                opName,
                []( IExecutionContext* context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                {
                    const auto& lnConfig = static_cast<const LayerNormConfig&>( config );
                    return std::make_shared<CudaLayerNormOp<TensorDataType::FP16>>( context, lnConfig );
                } );
        }
    };
}