/**
 * @file CpuLinearOp.ixx
 * @brief CPU implementation of the Fully Connected operation (TensorDataType-based).
 *
 * Ported to the ExecutionContext / TensorDataType UnaryOperation interface following
 * the pattern used by CpuGeluOp.
 */

module;
#include <math.h>
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <sstream>
#ifdef USE_OMP
#include <omp.h>
#endif
#include <functional>

export module Compute.CpuLinearOp;

import Dnn.Modules.Linear;
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
     * @brief CPU implementation of the Fully Connected / Linear operation using abstract TensorDataType.
     *
     * Forward: Y = X * W^T + b
     * Backward:
     *  - dX += dY * W
     *  - dW += dY^T * X
     *  - db += sum(dY)
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cpu>
    class CpuLinearOp : public UnaryOperation<DeviceType::Cpu, TPrecision>
    {
    public:
        using MR = CpuMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cpu, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;
        using HostType = typename TensorHostTypeMap<TPrecision>::host_type;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;

        CpuLinearOp( std::shared_ptr<CpuExecutionContext> context, const LinearConfig& config )
            : config_( config ), context_( context )
        {
            if (context_ && context_->getDevice()->getDeviceType() != DeviceType::Cpu)
            {
                throw std::runtime_error( "CpuLinearOp requires a CPU execution context" );
            }

            config_.validate();
        }

		// ====================================================================
		// Parameters
		// ====================================================================

        /**
         * @brief Set weight and optional bias parameters.
         *
         * @param weight Weight matrix of shape (out_features, in_features)
         * @param bias Optional bias vector of shape (out_features)
         */
        void setParameters( const TensorType* weight, const TensorType* bias )
        {
            if (!weight)
            {
                throw std::invalid_argument( "Weight cannot be null" );
            }

            weight_ptr_ = weight->data();
            bias_ptr_ = bias ? bias->data() : nullptr;
            has_bias_ = (bias_ptr_ != nullptr);

            // Validate weight dimensions match config
            auto weight_shape = weight->shape();
            if (weight_shape.size() != 2)
            {
                throw std::invalid_argument( "Weight must be 2D" );
            }
            if (weight_shape[0] != config_.getOutputFeatures() )
            {
                throw std::invalid_argument(
                    "Weight dim 0 must match config outputfeatures"
                );
            }
            if (input_features_ > 0 && weight_shape[1] != input_features_)
            {
                throw std::invalid_argument(
                    "Weight dim 1 must match input_features"
                );
            }
        }

		// ====================================================================
		// Lifecycle
		// ====================================================================

        void build( const shape_t& input_shape ) override
        {
            if (this->is_built_ && input_shape == cached_input_shape_)
            {
                return;  // Already built for this shape
            }

            if (input_shape.empty())
            {
                throw std::invalid_argument( "Input shape cannot be empty" );
            }

            // Extract dimensions
            // Input: (batch_dims..., in_features)
            input_features_ = input_shape.back();

            if (config_.getInputFeatures() > 0 && config_.getInputFeatures() != input_features_)
            {
                throw std::invalid_argument(
                    "Input features " + std::to_string( in_features_ ) +
                    " don't match config " + std::to_string( config_.in_features )
                );
            }

            // Compute batch size (flatten all but last dimension)
            batch_size_ = std::accumulate(
                input_shape.begin(),
                input_shape.end() - 1,
                dim_t{ 1 },
                std::multiplies{}
            );

            output_features_ = config_.getOutputFeatures();

            cached_input_shape_ = input_shape;
            
            this->is_built_ = true;
        }

		// ====================================================================
		// Computation
		// ====================================================================

        void forward( const ITensor& input, ITensor& output ) const override
        {
            if (!this->is_built_)
            {
                throw std::runtime_error( "LinearOp not built - call build() first" );
            }
            const HostType* X = static_cast<const HostType*>(input.rawData());
            HostType* Y = static_cast<HostType*>(output.rawData());

            if (!X || !Y)
            {
                throw std::runtime_error( "CpuLinearOp::forward - null tensor data pointer" );
            }

            const auto& in_shape = input.shape();
            if (in_shape.size() < 2)
            {
                throw std::runtime_error( "CpuLinearOp::forward - expected input rank >= 2" );
            }

            // compute outer size = product of leading dims except last (features)
            int C = static_cast<int>(in_shape.back());
            size_t outer_size = 1;
            for (size_t i = 0; i + 1 < in_shape.size(); ++i) outer_size *= in_shape[i];

            // Output feature dimension from output tensor shape
            const auto& out_shape = output.shape();
            if (out_shape.size() < 2)
            {
                throw std::runtime_error( "CpuLinearOp::forward - expected output rank >= 2" );
            }
            int OC = static_cast<int>(out_shape.back());

            // weight expected shape: [OC, C] (row-major)
            // input layout: outer index -> contiguous C features
            // output layout: outer index -> contiguous OC features

            // loop-unroll optimization retained from previous implementation
            const int LOOP_UNROLL = 8;
            if (outer_size % LOOP_UNROLL != 0)
            {
                // fallback naive
#pragma omp parallel for
                for (size_t idx = 0; idx < outer_size; ++idx)
                {
                    const size_t in_base = idx * static_cast<size_t>( C );
                    const size_t out_base = idx * static_cast<size_t>( OC );
                    for (int o = 0; o < OC; ++o)
                    {
                        long double acc = 0.0L;
                        for (int i = 0; i < C; ++i)
                        {
                            acc += static_cast<long double>( X[in_base + i] ) * static_cast<long double>( W[o * C + i] );
                        }
                        if (B) acc += static_cast<long double>( B[o] );
                        Y[out_base + o] = static_cast<HostType>( acc );
                    }
                }
                return;
            }

#pragma omp parallel for
            for (int out_idx = 0; out_idx < static_cast<int>( outer_size ); out_idx += LOOP_UNROLL)
            {
                for (int o = 0; o < OC; ++o)
                {
                    HostType result[LOOP_UNROLL];
                    for (int i_idx = 0; i_idx < LOOP_UNROLL; ++i_idx)
                    {
                        result[i_idx] = (B ? B[o] : static_cast<HostType>( 0 ));
                    }

                    for (int i = 0; i < C; ++i)
                    {
                        HostType w = W[o * C + i];
                        for (int i_idx = 0; i_idx < LOOP_UNROLL; ++i_idx)
                        {
                            int idx = out_idx + i_idx;
                            result[i_idx] += X[idx * C + i] * w;
                        }
                    }

                    for (int i_idx = 0; i_idx < LOOP_UNROLL; ++i_idx)
                    {
                        int idx = out_idx + i_idx;
                        Y[idx * OC + o] = result[i_idx];
                    }
                }
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
            const HostType* X = static_cast<const HostType*>(input.rawData());
            const HostType* dY = static_cast<const HostType*>(grad_output.rawData());
            HostType* dX = static_cast<HostType*>(grad_input.rawData());

            if (!X || !dY || !dX)
            {
                throw std::runtime_error( "CpuLinearOp::backward - null tensor data pointer" );
            }

            if (parameters.empty() || !parameters[0])
            {
                throw std::invalid_argument( "CpuLinearOp::backward requires weight parameter" );
            }

            const HostType* W = static_cast<const HostType*>(parameters[0]->data());

            const auto& in_shape = input.shape();
            int C = static_cast<int>(in_shape.back());
            size_t outer_size = 1;
            for (size_t i = 0; i + 1 < in_shape.size(); ++i) outer_size *= in_shape[i];

            const auto& out_shape = grad_output.shape();
            int OC = static_cast<int>(out_shape.back());

            // Prepare gradients for parameters if provided
            HostType* dW = nullptr;
            HostType* dB = nullptr;
            if (grad_parameters.size() > 0 && grad_parameters[0])
            {
                dW = static_cast<HostType*>(grad_parameters[0]->data());
            }
            if (grad_parameters.size() > 1 && grad_parameters[1])
            {
                dB = static_cast<HostType*>(grad_parameters[1]->data());
            }

#pragma omp parallel for collapse(1)
            for (size_t idx = 0; idx < outer_size; ++idx)
            {
                const size_t in_base = idx * static_cast<size_t>( C );
                const size_t out_base = idx * static_cast<size_t>( OC );

                // compute dX for this outer index
                for (int i = 0; i < C; ++i)
                {
                    long double acc = 0.0L;
                    for (int o = 0; o < OC; ++o)
                    {
                        acc += static_cast<long double>( W[o * C + i] ) * static_cast<long double>( dY[out_base + o] );
                    }
#pragma omp atomic
                    dX[in_base + i] += static_cast<HostType>( acc );
                }

                // accumulate dW and dB
                for (int o = 0; o < OC; ++o)
                {
                    long double dy = static_cast<long double>( dY[out_base + o] );
                    if (dB)
                    {
#pragma omp atomic
                        dB[o] += static_cast<HostType>( dy );
                    }
                    if (dW)
                    {
                        for (int i = 0; i < C; ++i)
                        {
#pragma omp atomic
                            dW[o * C + i] += static_cast<HostType>( static_cast<long double>( X[in_base + i] ) * dy );
                        }
                    }
                }
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::LinearOp;
        }

        std::string getName() const override
        {
            return "Cpu::LinearOp";
        }

    private:
        LinearConfig config_;
        std::shared_ptr<CpuExecutionContext> context_;
    };

    export class CpuLinearOpRegistrar
    {
    public:
        static void registerOperations()
        {
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>(
                "LinearOp",
                []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CpuLinearOp<TensorDataType::FP32>>( context, linearConfig );
                }
            );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}