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

export module Compute.CpuLayerNormOp;

import Dnn.Modules.LayerNorm;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
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
            if (context_ && context_->getDevice()->getDeviceType() != DeviceType::Cpu)
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
            if (shape.size() < 3)
            {
                throw std::runtime_error( "CpuLayerNormOp::forward - expected input rank >= 3 [B,T,C]" );
            }

            int B = static_cast<int>(shape[0]);
            int T = static_cast<int>(shape[1]);
            int C = static_cast<int>(shape[2]);

            HostType eps = static_cast<HostType>(config_.getEpsilon());

#pragma omp parallel for collapse(2) if( (size_t)B * (size_t)T * (size_t)C > 1000 )
            for (int b = 0; b < B; ++b)
            {
                for (int t = 0; t < T; ++t)
                {
                    int offset = b * T * C + t * C;

                    // compute mean
                    long double m = 0.0L;
                    for (int i = 0; i < C; ++i)
                    {
                        m += static_cast<long double>( X[offset + i] );
                    }
                    m = m / static_cast<long double>( C );

                    // compute variance
                    long double v = 0.0L;
                    for (int i = 0; i < C; ++i)
                    {
                        long double xshift = static_cast<long double>( X[offset + i] ) - m;
                        v += xshift * xshift;
                    }
                    v = v / static_cast<long double>( C );

                    long double s = 1.0L / std::sqrt( v + static_cast<long double>( eps ) );

                    // normalize and apply weight/bias
                    for (int i = 0; i < C; ++i)
                    {
                        HostType n = static_cast<HostType>( s * (static_cast<long double>( X[offset + i] ) - m) );
                        long double o = static_cast<long double>( n );
                        if (weight)
                        {
                            o *= static_cast<long double>( weight[i] );
                        }
                        if (bias)
                        {
                            o += static_cast<long double>( bias[i] );
                        }
                        Y[offset + i] = static_cast<HostType>(o);
                    }

                    mean[b * T + t] = static_cast<HostType>(m);
                    rstd[b * T + t] = static_cast<HostType>(s);
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
            if (shape.size() < 3)
            {
                throw std::runtime_error( "CpuLayerNormOp::backward - expected input rank >= 3 [B,T,C]" );
            }

            int B = static_cast<int>(shape[0]);
            int T = static_cast<int>(shape[1]);
            int C = static_cast<int>(shape[2]);

#pragma omp parallel for collapse(2) if( (size_t)B * (size_t)T * (size_t)C > 1000 )
            for (int b = 0; b < B; ++b)
            {
                for (int t = 0; t < T; ++t)
                {
                    int base = b * T * C + t * C;
                    const HostType* dout_bt = dout + base;
                    const HostType* inp_bt = inp + base;
                    HostType* dinp_bt = dinp + base;
                    long double mean_bt = static_cast<long double>( mean[b * T + t] );
                    long double rstd_bt = static_cast<long double>( rstd[b * T + t] );

                    // accumulate intermediate sums
                    long double dnorm_mean = 0.0L;
                    long double dnorm_norm_mean = 0.0L;

                    for (int i = 0; i < C; ++i)
                    {
                        long double norm_bti = (static_cast<long double>( inp_bt[i] ) - mean_bt) * rstd_bt;
                        long double dnorm_i = static_cast<long double>( weight[i] ) * static_cast<long double>( dout_bt[i] );
                        dnorm_mean += dnorm_i;
                        dnorm_norm_mean += dnorm_i * norm_bti;
                    }
                    dnorm_mean /= static_cast<long double>( C );
                    dnorm_norm_mean /= static_cast<long double>( C );

                    for (int i = 0; i < C; ++i)
                    {
                        long double norm_bti = (static_cast<long double>( inp_bt[i] ) - mean_bt) * rstd_bt;
                        long double dnorm_i = static_cast<long double>( weight[i] ) * static_cast<long double>( dout_bt[i] );

                        if (dbias)
                        {
                            // accumulate dbias (sum over B and T)
#pragma omp atomic
                            dbias[i] += static_cast<HostType>( dout_bt[i] );
                        }

                        if (dweight)
                        {
                            // accumulate dweight (sum over B and T)
#pragma omp atomic
                            dweight[i] += static_cast<HostType>(norm_bti * static_cast<long double>(dout_bt[i]));
                        }

                        long double dval = dnorm_i;
                        dval -= dnorm_mean;
                        dval -= norm_bti * dnorm_norm_mean;
                        dval *= rstd_bt;

#pragma omp atomic
                        dinp_bt[i] += static_cast<HostType>(dval);
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