/**
 * @file CudaGatedDeltaRuleOp.ixx
 * @brief CUDA implementation of the recurrent gated delta rule.
 *
 * The state is FP32 regardless of the activation precision -- the reference runs the
 * recurrence in float32 (`mamba_ssm_dtype`), and a state that is both carried across every
 * token and repeatedly accumulated into is the last place to economize on mantissa.
 */

module;
#include "Kernels/GatedDeltaRule.cuh"
#include <cuda_bf16.h>
#include <stdexcept>
#include <format>
#include <cstdint>
#include <string>

export module Compute.CudaGatedDeltaRuleOp;
import :Dispatch;

import Dnn.Components.GatedDeltaRuleConfig;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.OperationType;
import Dnn.Component;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute::Cuda::DeltaNet
{
    export template <TensorDataType TPrecision>
    class CudaGatedDeltaRuleOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;
        using ConfigType = GatedDeltaRuleConfig;

        CudaGatedDeltaRuleOp( IExecutionContext* context, const GatedDeltaRuleConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaGatedDeltaRuleOp" ) ),
              config_( config )
        {
            if ( !context_ )
            {
                throw std::invalid_argument(
                    "CudaGatedDeltaRuleOp requires a non-null CUDA execution context" );
            }

            config_.validate();

            num_key_heads_ = static_cast<int>( config_.getNumKeyHeads() );
            num_value_heads_ = static_cast<int>( config_.getNumValueHeads() );
            head_key_dim_ = static_cast<int>( config_.getHeadKeyDim() );
            head_value_dim_ = static_cast<int>( config_.getHeadValueDim() );
        }

        /// A_log and dt_bias, both [num_value_heads]. The rule derives g and beta from them.
        void setParameters( ITensor* A_log, ITensor* dt_bias )
        {
            A_log_ = A_log;
            dt_bias_ = dt_bias;
        }

        /**
         * @brief Advance the recurrence over a chunk, updating @p state in place.
         *
         * @param q,k   [B, T, num_key_heads * head_key_dim] -- NOT expanded to value heads.
         * @param v     [B, T, num_value_heads * head_value_dim].
         * @param a,b   [B, T, num_value_heads] -- the raw projections, not g and beta.
         * @param state [B, num_value_heads, head_key_dim, head_value_dim], FP32.
         * @param out   [B, T, num_value_heads * head_value_dim].
         */
        void forward(
            const ITensor& q, const ITensor& k, const ITensor& v,
            const ITensor& a, const ITensor& b,
            ITensor& state, ITensor& out ) const
        {
            if ( !A_log_ || !dt_bias_ )
            {
                throw std::runtime_error( "CudaGatedDeltaRuleOp::forward - parameters not set" );
            }

            const auto& q_shape = q.shape();

            if ( q_shape.size() != 3 )
            {
                throw std::runtime_error(
                    "CudaGatedDeltaRuleOp::forward - q must be rank 3 [B, T, key_width]" );
            }

            const int batch = static_cast<int>( q_shape[ 0 ] );
            const int steps = static_cast<int>( q_shape[ 1 ] );

            if ( q_shape[ 2 ] != config_.getKeyWidth() || k.shape() != q_shape )
            {
                throw std::runtime_error(
                    "CudaGatedDeltaRuleOp::forward - q/k must both be [B, T, key_width]" );
            }

            if ( v.shape()[ 2 ] != config_.getValueWidth() || out.shape() != v.shape() )
            {
                throw std::runtime_error(
                    "CudaGatedDeltaRuleOp::forward - v/out must be [B, T, value_width]" );
            }

            if ( a.shape()[ 2 ] != config_.getNumValueHeads() || b.shape() != a.shape() )
            {
                throw std::runtime_error(
                    "CudaGatedDeltaRuleOp::forward - a/b must be [B, T, num_value_heads]" );
            }

            if ( state.size() != batch * config_.getStateElementsPerBatch() )
            {
                throw std::runtime_error(
                    "CudaGatedDeltaRuleOp::forward - state does not match the built geometry" );
            }

            Detail::cuda_gated_delta_rule_impl<NativeType>::forward(
                static_cast<NativeType*>( out.rawData() ),
                static_cast<const NativeType*>( q.rawData() ),
                static_cast<const NativeType*>( k.rawData() ),
                static_cast<const NativeType*>( v.rawData() ),
                static_cast<const NativeType*>( a.rawData() ),
                static_cast<const NativeType*>( b.rawData() ),
                static_cast<const NativeType*>( A_log_->rawData() ),
                static_cast<const NativeType*>( dt_bias_->rawData() ),
                static_cast<float*>( state.rawData() ),
                batch, steps, num_key_heads_, num_value_heads_,
                head_key_dim_, head_value_dim_,
                context_->getStream() );
        }

        void build( const BuildContext& /*context*/ ) override
        {
        }

        OperationType getOperationType() const override
        {
            return OperationType::GatedDeltaRuleOp;
        }

        std::string getName() const override
        {
            return "Cuda::GatedDeltaRuleOp";
        }

    private:
        CudaExecutionContext* context_{ nullptr };
        GatedDeltaRuleConfig config_;

        ITensor* A_log_{ nullptr };
        ITensor* dt_bias_{ nullptr };

        int num_key_heads_{ 0 };
        int num_value_heads_{ 0 };
        int head_key_dim_{ 0 };
        int head_value_dim_{ 0 };
    };
}
