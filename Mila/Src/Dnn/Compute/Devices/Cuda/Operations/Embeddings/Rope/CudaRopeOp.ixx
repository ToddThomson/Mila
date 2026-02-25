/**
 * @file CudaRopeOp.ixx
 * @brief CUDA backend stub for RoPE (rotary positional embedding) encoder transform.
 *
 * This implementation provides a minimal CUDA-based UnaryOperation that performs
 * an identity copy for the forward and backward paths. It follows the patterns
 * used by other CUDA ops so it integrates with the module/registration system.
 *
 * NOTE: Kernelized RoPE implementations (applying sin/cos rotations) are not
 * provided here. This file acts as a placeholder implementation that is
 * correct, safe and will compile and run while a full kernel implementation
 * is added later.
 */

module;
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdint>

export module Compute.CudaRopeOp;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.ComponentConfig;
import Compute.Precision;
import Compute.UnaryOperation;
import Compute.OperationType;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.OperationRegistrarHelpers;

namespace Mila::Dnn::Compute::Cuda::Rope
{
    using namespace Mila::Dnn;

    /**
     * @brief Minimal CUDA RoPE op that currently performs device-to-device copy.
     *
     * This is a placeholder implementation that preserves the module/registration
     * patterns used across the CUDA ops. A full RoPE kernel can replace the
     * memcpy-based implementation without changing the external API.
     */
    export template< Dnn::TensorDataType TPrecision = Dnn::TensorDataType::FP32 >
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaRopeEncoderOp : public UnaryOperation<DeviceType::Cuda, TPrecision, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        // Expose ConfigType so registrar helpers can statically cast ComponentConfig
        using ConfigType = ComponentConfig;

        /**
         * @brief Construct with a CUDA execution context and generic config.
         *
         * The config is accepted for compatibility with the registrar helper.
         */
        explicit CudaRopeEncoderOp( IExecutionContext* context, const ComponentConfig& /*cfg*/ )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaRopeEncoderOp" ) )
        {
            if ( !context_ )
            {
                throw std::invalid_argument( "CudaRopeEncoderOp requires a non-null CUDA execution context" );
            }
        }

        ~CudaRopeEncoderOp() override = default;

        /**
         * @brief Build caches the total element count for efficient memcpy dispatch.
         *
         * Validates that the input has rank 3: [B, T, C] expected for token/sequence embeddings.
         */
        void build( const shape_t& input_shape ) override
        {
            if ( input_shape.size() != 3 )
            {
                throw std::invalid_argument( "CudaRopeEncoderOp::build - input must have rank 3 [B, T, C]" );
            }

            total_elements_ = 1;
            for ( const auto& d : input_shape )
            {
                total_elements_ *= static_cast<int>(d);
            }
        }

        /**
         * @brief Forward: currently performs device-to-device copy (identity).
         *
         * Replace this implementation with the actual RoPE kernel when available.
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            NativeType* Y = static_cast<NativeType*>(output.rawData());

            if ( !X || !Y )
            {
                throw std::runtime_error( "CudaRopeEncoderOp::forward - null tensor data pointer" );
            }

            if ( total_elements_ <= 0 )
            {
                throw std::runtime_error( "CudaRopeEncoderOp::forward - op not built or invalid shape" );
            }

            const size_t bytes = static_cast<size_t>(total_elements_) * sizeof( NativeType );

            cudaStream_t stream = context_->getStream();

            // Perform asynchronous device-to-device copy on the context's stream.
            cudaError_t err = cudaMemcpyAsync( Y, X, bytes, cudaMemcpyDeviceToDevice, stream );
            if ( err != cudaSuccess )
            {
                throw std::runtime_error( std::string( "CudaRopeEncoderOp::forward - cudaMemcpyAsync failed: " ) + cudaGetErrorString( err ) );
            }
        }

        /**
         * @brief Backward: copies output gradient into input grad (identity).
         *
         * Real RoPE backward should implement the inverse of forward rotation.
         */
        void backward(
            const ITensor& input, // unused
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            const NativeType* dY = static_cast<const NativeType*>(output_grad.rawData());
            NativeType* dX = static_cast<NativeType*>(input_grad.rawData());

            if ( !dY || !dX )
            {
                throw std::runtime_error( "CudaRopeEncoderOp::backward - null gradient tensor data pointer" );
            }

            if ( total_elements_ <= 0 )
            {
                throw std::runtime_error( "CudaRopeEncoderOp::backward - op not built or invalid shape" );
            }

            const size_t bytes = static_cast<size_t>(total_elements_) * sizeof( NativeType );

            cudaStream_t stream = context_->getStream();

            cudaError_t err = cudaMemcpyAsync( dX, dY, bytes, cudaMemcpyDeviceToDevice, stream );
            if ( err != cudaSuccess )
            {
                throw std::runtime_error( std::string( "CudaRopeEncoderOp::backward - cudaMemcpyAsync failed: " ) + cudaGetErrorString( err ) );
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::RopeOp;
        }

        std::string getName() const override
        {
            return std::string( "Cuda::" ) + std::string( operationTypeToString( getOperationType() ) );
        }

    private:
        CudaExecutionContext* context_{ nullptr };
        int total_elements_{ 0 };
    };

    /**
     * @brief Registrar that exposes the Rope op for FP32 and FP16 compute types.
     */
    export class CudaRopeOpRegistrar
    {
    public:
        static void registerOperations()
        {
            const std::string_view opName = Compute::OperationNames::Rope;

            registerUnaryOpType<DeviceType::Cuda,
                CudaRopeEncoderOp<TensorDataType::FP32>,
                TensorDataType::FP32, TensorDataType::FP32>( opName );

            registerUnaryOpType<DeviceType::Cuda,
                CudaRopeEncoderOp<TensorDataType::FP16>,
                TensorDataType::FP16, TensorDataType::FP16>( opName );
        }
    };
}