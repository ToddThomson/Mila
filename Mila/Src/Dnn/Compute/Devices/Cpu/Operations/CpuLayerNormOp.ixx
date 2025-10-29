/**
 * @file CpuLayerNormOp.ixx
 * @brief CPU implementation of Layer Normalization operation (TensorDataType-based).
 *
 * Ported to the ExecutionContext / TensorDataType UnaryOperation interface following
 * the pattern used by CpuGeluOp.
 */

module;
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#ifdef USE_OMP
#include <omp.h>
#endif
#include <cstdint>

export module Compute.CpuLayerNormOp;

import Dnn.Modules.LayerNorm;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
import Dnn.TensorPartitioning;
import Dnn.ConfigurationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.CpuExecutionContext;
import Compute.OperationType;
import Compute.OperationAttributes;
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
     * @brief CPU implementation of Layer Normalization using abstract TensorDataType API.
     *
     * Template parameter TPrecision selects the abstract tensor precision (e.g. FP32).
     * HostType is the corresponding CPU host representation for that precision.
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cpu>
    class CpuLayerNormOp : public UnaryOperation<DeviceType::Cpu, TPrecision>
    {
    public:
        using MR = CpuMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cpu, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;
        using HostType = typename TensorHostTypeMap<TPrecision>::host_type;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;

        CpuLayerNormOp( std::shared_ptr<CpuExecutionContext> context, const LayerNormConfig& config )
            : config_( config ), context_( context )
        {
            if (!context_)
            {
                throw std::runtime_error( "CpuLayerNormOp requires a CPU execution context" );
            }

            config_.validate();
        }

        /**
         * Forward:
         * - input: ITensor containing input values
         * - parameters: [weight, bias] as TensorType shared_ptrs (bias optional)
         * - output: ITensor to write normalized result
         * - output_state: must contain two allocated tensors for mean and rstd
         */
        void forward(
            const ITensor& input,
            const Parameters& parameters,
            ITensor& output,
            OutputState& output_state ) const override
        {
            const HostType* X = static_cast<const HostType*>(input.rawData());
            HostType* Y = static_cast<HostType*>(output.rawData());

            if (!X || !Y)
            {
                throw std::runtime_error( "CpuLayerNormOp::forward - null tensor data pointer" );
            }

            const HostType* weight = nullptr;
            const HostType* bias = nullptr;
            
            if (!parameters.empty() && parameters[0])
            {
                weight = static_cast<const HostType*>(parameters[0]->data());
            }
            
            if (parameters.size() > 1 && parameters[1])
            {
                bias = static_cast<const HostType*>(parameters[1]->data());
            }

            if (output_state.size() < 2 || !output_state[0] || !output_state[1])
            {
                throw std::invalid_argument( "CpuLayerNormOp::forward requires output_state[0]=mean and output_state[1]=rstd tensors." );
            }

            HostType* mean = static_cast<HostType*>(output_state[0]->data());
            HostType* rstd = static_cast<HostType*>(output_state[1]->data());

            const auto& shape = input.shape();
            const int64_t ndim = shape.size();

            if ( ndim < 1 )
            {
                throw std::runtime_error( "CpuLayerNormOp::forward - input must have rank >= 1" );
            }

            // Option 1: Using axis (single dimension normalization)
            if (config_.getAxis().has_value())
            {
                auto partition = computeAxisPartition(
                    shape,
                    config_.getAxis().value(),
                    "CpuLayerNormOp::forward"
                );
            }

            auto axis = -1;
            if ( axis < 0 )
                axis += ndim;
            
            int64_t outer_size = 1;
            for (int64_t i = 0; i < axis; ++i) outer_size *= static_cast<int64_t>( shape[i] );

            const int64_t dim_size = static_cast<int64_t>( shape[axis] );

            int64_t inner_size = 1;
            for (int64_t i = axis + 1; i < ndim; ++i)
                inner_size *= static_cast<int64_t>( shape[i] );

            // Validate output_state shape matches expected number of slices
            const int64_t expected_slices = outer_size * inner_size;
            const auto& mean_shape = output_state[0]->shape();
            int64_t mean_elems = 1;
            
            for (auto d : mean_shape)
                mean_elems *= static_cast<int64_t>(d);
            
            if ( mean_elems != expected_slices )
            {
                throw std::runtime_error( "CpuLayerNormOp::forward - output_state mean tensor has unexpected size" );
            }

#pragma omp parallel for collapse(2) if( (size_t)outer_size * (size_t)inner_size > 100 )
            for (int64_t outer = 0; outer < outer_size; ++outer)
            {
                for (int64_t inner = 0; inner < inner_size; ++inner)
                {
                    const HostType* slice_in = X + (outer * dim_size * inner_size) + inner;
                    HostType* slice_out = Y + (outer * dim_size * inner_size) + inner;

                    // compute mean over the reduction axis
                    long double m = 0.0L;
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        m += static_cast<long double>( slice_in[i * inner_size] );
                    }
                    m /= static_cast<long double>( dim_size );

                    // compute variance
                    long double v = 0.0L;
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        long double diff = static_cast<long double>( slice_in[i * inner_size] ) - m;
                        v += diff * diff;
                    }
                    v /= static_cast<long double>( dim_size );

                    long double s = 1.0L / std::sqrt( v + static_cast<long double>( config_.getEpsilon() ) );

                    // normalize and apply weight/bias (weight/bias indexed along reduction dim)
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        long double n = s * (static_cast<long double>( slice_in[i * inner_size] ) - m);
                        if ( weight )
                        {
                            n *= static_cast<long double>( weight[i] );
                        }
                        if ( bias )
                        {
                            n += static_cast<long double>( bias[i] );
                        }
                        slice_out[i * inner_size] = static_cast<HostType>( n );
                    }

                    const int64_t slice_index = outer * inner_size + inner;
                    mean[slice_index] = static_cast<HostType>( m );
                    rstd[slice_index] = static_cast<HostType>( s );
                }
            }
        }

        /**
         * Backward:
         * - input: ITensor of original inputs (same as forward input)
         * - output_grad: gradient w.r.t. output (dout)
         * - parameters: [weight, bias] (weight required for gradient w.r.t. input)
         * - output_state: [mean, rstd] produced by forward
         * - input_grad: output gradient buffer to accumulate dinput
         * - parameter_grads: vector to accumulate [dweight, dbias] (optional)
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            const Parameters& parameters,
            const OutputState& output_state,
            ITensor& input_grad,
            Parameters& parameter_grads ) const override
        {
            const HostType* inp = static_cast<const HostType*>(input.rawData());
            const HostType* dout = static_cast<const HostType*>(output_grad.rawData());
            HostType* dinp = static_cast<HostType*>(input_grad.rawData());

            if (!inp || !dout || !dinp)
            {
                throw std::runtime_error( "CpuLayerNormOp::backward - null tensor data pointer" );
            }

            const HostType* weight = nullptr;
            if (!parameters.empty() && parameters[0])
            {
                weight = static_cast<const HostType*>(parameters[0]->data());
            }
            else
            {
                throw std::runtime_error( "CpuLayerNormOp::backward requires weight parameter." );
            }

            HostType* dweight = nullptr;
            HostType* dbias = nullptr;
            
            if (parameter_grads.size() > 0 && parameter_grads[0])
            {
                dweight = static_cast<HostType*>(parameter_grads[0]->data());
            }
            
            if (parameter_grads.size() > 1 && parameter_grads[1])
            {
                dbias = static_cast<HostType*>(parameter_grads[1]->data());
            }

            if (output_state.size() < 2 || !output_state[0] || !output_state[1])
            {
                throw std::invalid_argument( "CpuLayerNormOp::backward requires output_state[0]=mean and output_state[1]=rstd tensors." );
            }

            const HostType* mean = static_cast<const HostType*>(output_state[0]->data());
            const HostType* rstd = static_cast<const HostType*>(output_state[1]->data());

            const auto& shape = input.shape();
            const int64_t ndim = static_cast<int64_t>( shape.size() );

            if ( ndim < 1 )
            {
                throw std::runtime_error( "CpuLayerNormOp::backward - input must have rank >= 1" );
            }

            if (config_.getAxis().has_value())
            {
                auto partition = computeAxisPartition(
                    shape,
                    config_.getAxis().value(),
                    "CpuLayerNormOp::forward"
                );
            }

            int64_t axis = -1;

            if ( axis < 0 )
                axis += ndim;
            
            if ( axis < 0 || axis >= ndim )
            {
                throw std::runtime_error( "CpuLayerNormOp::backward - axis out of range" );
            }

            int64_t outer_size = 1;
            for (int64_t i = 0; i < axis; ++i) outer_size *= static_cast<int64_t>( shape[i] );

            const int64_t dim_size = static_cast<int64_t>( shape[axis] );

            int64_t inner_size = 1;
            for (int64_t i = axis + 1; i < ndim; ++i) inner_size *= static_cast<int64_t>( shape[i] );

#pragma omp parallel for collapse(2) if( (size_t)outer_size * (size_t)inner_size > 100 )
            for (int64_t outer = 0; outer < outer_size; ++outer)
            {
                for (int64_t inner = 0; inner < inner_size; ++inner)
                {
                    const HostType* inp_slice = inp + (outer * dim_size * inner_size) + inner;
                    const HostType* dout_slice = dout + (outer * dim_size * inner_size) + inner;
                    HostType* dinp_slice = dinp + (outer * dim_size * inner_size) + inner;

                    const int64_t slice_index = outer * inner_size + inner;
                    long double mean_slice = static_cast<long double>( mean[slice_index] );
                    long double rstd_slice = static_cast<long double>( rstd[slice_index] );

                    // accumulate intermediate sums
                    long double dnorm_mean = 0.0L;
                    long double dnorm_norm_mean = 0.0L;

                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        long double norm_bti = (static_cast<long double>( inp_slice[i * inner_size] ) - mean_slice) * rstd_slice;
                        long double dnorm_i = static_cast<long double>( weight[i] ) * static_cast<long double>( dout_slice[i * inner_size] );
                        dnorm_mean += dnorm_i;
                        dnorm_norm_mean += dnorm_i * norm_bti;
                    }
                    dnorm_mean /= static_cast<long double>( dim_size );
                    dnorm_norm_mean /= static_cast<long double>( dim_size );

                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        long double norm_bti = (static_cast<long double>( inp_slice[i * inner_size] ) - mean_slice) * rstd_slice;
                        long double dnorm_i = static_cast<long double>( weight[i] ) * static_cast<long double>( dout_slice[i * inner_size] );

                        if (dbias)
                        {
                            // accumulate dbias (sum over slices)
#pragma omp atomic
                            dbias[i] += static_cast<HostType>( dout_slice[i * inner_size] );
                        }

                        if (dweight)
                        {
                            // accumulate dweight (sum over slices)
#pragma omp atomic
                            dweight[i] += static_cast<HostType>( norm_bti * static_cast<long double>( dout_slice[i * inner_size] ) );
                        }

                        long double dval = dnorm_i;
                        dval -= dnorm_mean;
                        dval -= norm_bti * dnorm_norm_mean;
                        dval *= rstd_slice;

#pragma omp atomic
                        dinp_slice[i * inner_size] += static_cast<HostType>( dval );
                    }
                }
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::LayerNormOp;
        }

        std::string getName() const override
        {
            return "Cpu::LayerNormOp";
        }

    private:
        LayerNormConfig config_;
        std::shared_ptr<CpuExecutionContext> context_;
    };

    // Register CPU LayerNorm (FP32)
    export class CpuLayerNormOpRegistrar
    {
    public:
        static void registerOperations()
        {
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>(
                "LayerNormOp",
                []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>>
                {
                    const auto& lnConfig = static_cast<const LayerNormConfig&>(config);
                    return std::make_shared<CpuLayerNormOp<TensorDataType::FP32>>( context, lnConfig );
                }
            );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}