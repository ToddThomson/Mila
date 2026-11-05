/**
 * @file MetalTensorOps.Structural.ixx
 * @brief Splitting a Metal tensor along its last axis.
 */

export module Compute.MetalTensorOps:Structural;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalTensorDataTypeMap;
import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal
{
    export struct StructuralOps
    {
        /// Encode [B, T, D0 + D1] into [B, T, D0] and [B, T, D1]. @throws std::invalid_argument When the shapes do not add up.
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void split(
            const Tensor<TDataType, TMemoryResource>& input,
            Tensor<TDataType, TMemoryResource>& output_a,
            Tensor<TDataType, TMemoryResource>& output_b,
            IExecutionContext* exec_context = nullptr )
        {
            splitInto( input, std::array<Tensor<TDataType, TMemoryResource>*, 2>{ &output_a, &output_b }, exec_context );
        }

        /// Encode [B, T, D0 + D1 + D2] into three tensors. @throws std::invalid_argument When the shapes do not add up.
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void split(
            const Tensor<TDataType, TMemoryResource>& input,
            Tensor<TDataType, TMemoryResource>& output_a,
            Tensor<TDataType, TMemoryResource>& output_b,
            Tensor<TDataType, TMemoryResource>& output_c,
            IExecutionContext* exec_context = nullptr )
        {
            splitInto( input, std::array<Tensor<TDataType, TMemoryResource>*, 3>{ &output_a, &output_b, &output_c }, exec_context );
        }

    private:

        template<TensorDataType TDataType, typename TMemoryResource, std::size_t TOutputs>
        static void splitInto(
            const Tensor<TDataType, TMemoryResource>& input,
            const std::array<Tensor<TDataType, TMemoryResource>*, TOutputs>& outputs,
            IExecutionContext* exec_context )
        {
            const shape_t& shape = input.shape();

            if ( shape.size() != 3 )
            {
                throw std::invalid_argument( "split: all tensors must be rank-3" );
            }

            dim_t columns = 0;

            for ( const auto* const output : outputs )
            {
                const shape_t& part = output->shape();

                if ( part.size() != 3 || part[0] != shape[0] || part[1] != shape[1] )
                {
                    throw std::invalid_argument( "split: every output must match the input in B and T" );
                }

                columns += part[2];
            }

            if ( columns != shape[2] )
            {
                throw std::invalid_argument( "split: the outputs' last axes must add up to the input's" );
            }

            if ( exec_context )
            {
                encode( input, outputs, *validateExecutionContext_<DeviceType::Metal>( exec_context, "Metal split" ) );

                return;
            }

            MetalExecutionContext context( input.getDeviceId() );
            encode( input, outputs, context );
            context.synchronize();
        }

        template<TensorDataType TDataType, typename TMemoryResource, std::size_t TOutputs>
        static void encode(
            const Tensor<TDataType, TMemoryResource>& input,
            const std::array<Tensor<TDataType, TMemoryResource>*, TOutputs>& outputs,
            MetalExecutionContext& context )
        {
            const std::string kernel = std::format( "split_columns_{}", MetalTensorDataTypeMap<TDataType>::kernel_type );
            const void* const input_address = static_cast<const ITensor&>( input ).rawData();

            std::uint32_t first_column = 0;

            for ( auto* const output : outputs )
            {
                const auto columns = static_cast<std::uint32_t>( narrowToKernelIndex( output->shape()[2] ) );
                const std::array<std::uint32_t, 3> shape{
                    static_cast<std::uint32_t>( narrowToKernelIndex( input.shape()[2] ) ), first_column, columns };
                const std::array<const void*, 2> buffers{ input_address, static_cast<ITensor&>( *output ).rawData() };

                context.dispatch(
                    context.device().kernels().pipeline( kernel ), buffers, std::as_bytes( std::span( shape ) ),
                    static_cast<std::size_t>( narrowToKernelIndex( output->size() ) ) );

                first_column += columns;
            }
        }
    };
}
