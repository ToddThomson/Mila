/**
 * @file CpuSoftmaxOp.ixx
 * @brief CPU implementation of the softmax operation (TensorDataType-based).
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#ifdef USE_OMP
#include <omp.h>
#endif
#include <cstdint>
#include <limits>

export module Compute.CpuSoftmaxOp;

import Dnn.Modules.Softmax;
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

using namespace Mila::Dnn;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU implementation of Softmax using the abstract TensorDataType API.
     *
     * This implementation mirrors the structure used in CpuGeluOp. It performs
     * numerically-stable softmax across the configured axis. Backward computes
     * dX = Y * (dY - sum(Y * dY)) for each axis slice.
     */
    export template<TensorDataType TPrecision>
        class CpuSoftmaxOp : public UnaryOperation<DeviceType::Cpu, TPrecision>
    {
    public:
        using MR = CpuMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cpu, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;
        using NativeType = typename CpuTensorDataTypeTraits::template native_type<TPrecision>;
        using HostType = typename TensorHostTypeMap<TPrecision>::host_type;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;

        CpuSoftmaxOp( std::shared_ptr<CpuExecutionContext> context, const SoftmaxConfig& config )
            : context_( context ), config_( config )
        {
            if (context_)
            {
                throw std::runtime_error( "CpuSoftmaxOp requires a CPU execution context" );
            }

            config_.validate();
        }

        void forward(
            const ITensor& input,
            [[maybe_unused]] const Parameters& parameters,
            ITensor& output,
            [[maybe_unused]] OutputState& output_state ) const override
        {
            const HostType* in_data = static_cast<const HostType*>(input.rawData());
            HostType* out_data = static_cast<HostType*>(output.rawData());

            if (!in_data || !out_data)
            {
                throw std::runtime_error( "CpuSoftmaxOp::forward - null tensor data pointer" );
            }
            auto& shape = input.shape();

            auto partition = computeAxisPartition(
                shape,
                config_.getAxis(),
                "CpuSoftmaxOp::forward"
            );

            int64_t axis = config_.getAxis();
            
            const int64_t ndim = static_cast<int64_t>(shape.size());
            
            if (axis < 0)
                axis = ndim + axis;

            if (axis < 0 || axis >= ndim)
            {
                throw std::runtime_error( "Softmax axis out of bounds" );
            }

            int64_t outer_size = 1;
            for (int64_t i = 0; i < axis; ++i) 
                outer_size *= shape[i];
            
            const int64_t dim_size = shape[axis];
            int64_t inner_size = 1;
            
            
            
            
            for (int64_t i = axis + 1; i < ndim; ++i) inner_size *= shape[i];

            // Iterate over slices [outer, inner] and compute softmax along dim_size
#pragma omp parallel for collapse(2) if(outer_size * inner_size > 100)
            for (int64_t outer = 0; outer < outer_size; ++outer)
            {
                for (int64_t inner = 0; inner < inner_size; ++inner)
                {
                    const HostType* slice_in = in_data + (outer * dim_size * inner_size) + inner;
                    HostType* slice_out = out_data + (outer * dim_size * inner_size) + inner;

                    // find max for numerical stability
                    HostType max_val = -std::numeric_limits<HostType>::infinity();
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        HostType v = slice_in[i * inner_size];
                        if (v > max_val) max_val = v;
                    }

                    // exp and sum
                    long double sum = 0.0L;
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        long double val = std::expl( static_cast<long double>( slice_in[i * inner_size] - max_val ) );
                        slice_out[i * inner_size] = static_cast<HostType>( val );
                        sum += val;
                    }

                    long double inv_sum = 1.0L / sum;
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        slice_out[i * inner_size] = static_cast<HostType>( static_cast<long double>( slice_out[i * inner_size] ) * inv_sum );
                    }
                }
            }
        }

        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            [[maybe_unused]] const Parameters& parameters,
            [[maybe_unused]] const OutputState& output_state,
            ITensor& input_grad,
            [[maybe_unused]] Parameters& parameter_grads ) const override
        {
            const HostType* probs = static_cast<const HostType*>(input.rawData()); // assuming 'input' contains softmax probabilities
            const HostType* dY = static_cast<const HostType*>(output_grad.rawData());
            HostType* dX = static_cast<HostType*>(input_grad.rawData());

            if (!probs || !dY || !dX)
            {
                throw std::runtime_error( "CpuSoftmaxOp::backward - null tensor data pointer" );
            }

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
            const int64_t dim_size = shape[axis];
            int64_t inner_size = 1;
            for (int64_t i = axis + 1; i < ndim; ++i) inner_size *= shape[i];

#pragma omp parallel for collapse(2) if(outer_size * inner_size > 100)
            for (int64_t outer = 0; outer < outer_size; ++outer)
            {
                for (int64_t inner = 0; inner < inner_size; ++inner)
                {
                    const HostType* slice_p = probs + (outer * dim_size * inner_size) + inner;
                    const HostType* slice_dy = dY + (outer * dim_size * inner_size) + inner;
                    HostType* slice_dx = dX + (outer * dim_size * inner_size) + inner;

                    // compute dot = sum_j y_j * dY_j
                    long double dot = 0.0L;
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        dot += static_cast<long double>( slice_p[i * inner_size] ) * static_cast<long double>( slice_dy[i * inner_size] );
                    }

                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        long double y = static_cast<long double>( slice_p[i * inner_size] );
                        long double grad = y * (static_cast<long double>( slice_dy[i * inner_size] ) - dot);
                        slice_dx[i * inner_size] += static_cast<HostType>( grad );
                    }
                }
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::SoftmaxOp;
        }

        std::string getName() const override
        {
            return "Cpu::SoftmaxOp";
        }

    private:
        SoftmaxConfig config_;
        std::shared_ptr<CpuExecutionContext> context_;
    };

    // Register CPU Softmax (FP32)
    export class CpuSoftmaxOpRegistrar
    {
    public:
        static void registerOperations()
        {
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>(
                "SoftmaxOp",
                []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
                    const ConfigurationBase& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>>
                {
                    const auto& softmaxConfig = static_cast<const SoftmaxConfig&>(config);
                    return std::make_shared<CpuSoftmaxOp<TensorDataType::FP32>>( context, softmaxConfig );
                }
            );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}