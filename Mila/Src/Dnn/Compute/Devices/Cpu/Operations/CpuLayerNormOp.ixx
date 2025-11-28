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

import Dnn.Components.LayerNorm;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
import Dnn.TensorPartitioning;
import Dnn.ComponentConfig;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.CpuExecutionContext;
import Compute.OperationType;
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
     * float is the corresponding CPU host representation for that precision.
     */
    class CpuLayerNormOp : public UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using MR = CpuMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorType = Tensor<TensorDataType::FP32, MR>;
        //using float = typename TensorfloatMap<TPrecision>::host_type;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;

        CpuLayerNormOp( std::shared_ptr<CpuExecutionContext> context, const LayerNormConfig& config )
            : context_( context ), config_( config )
        {
            if (!context_)
            {
                throw std::runtime_error( "CpuLayerNormOp requires a CPU execution context" );
            }

            config_.validate();
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        /**
         * @brief Set parameter tensor references (module remains owner).
         *
         * The operation caches native data pointers for hot-path access. The
         * weight tensor is required; bias is bound only when the LayerNorm
         * config indicates a bias is present.
         *
         * Note: build() requires parameters to be bound before it is called.
         */
        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            if (!weight)
            {
                throw std::invalid_argument( "CpuLayerNormOp::setParameters - weight parameter is required" );
            }

            weight_ = static_cast<float*>(weight->rawData());

            if (config_.hasBias())
            {
                if (!bias)
                {
                    throw std::invalid_argument( "CpuLayerNormOp::setParameters - bias parameter expected but null was provided" );
                }

                bias_ = static_cast<float*>(bias->rawData());
            }
            else
            {
                bias_ = nullptr;
            }
        }

        /**
         * @brief Set parameter gradient tensor references for training.
         *
         * The operation caches native gradient pointers for hot-path write access
         * during backward(). Weight gradient is required; bias gradient is bound
         * only when the LayerNorm config indicates a bias is present.
         *
         * @param weight_grad Gradient tensor for weight parameter
         * @param bias_grad Gradient tensor for bias parameter (optional based on config)
         *
         * @throws std::invalid_argument If weight_grad is null
         * @throws std::invalid_argument If bias_grad is null when config requires bias
         */
        void setGradients( ITensor* weight_grad, ITensor* bias_grad ) override
        {
            if (!weight_grad)
            {
                throw std::invalid_argument( "CpuLayerNormOp::setParameterGradients - weight gradient is required" );
            }

            weight_grad_ = static_cast<float*>(weight_grad->rawData());

            if (config_.hasBias())
            {
                if (!bias_grad)
                {
                    throw std::invalid_argument( "CpuLayerNormOp::setParameterGradients - bias gradient expected but null was provided" );
                }

                bias_grad_ = static_cast<float*>(bias_grad->rawData());
            }
            else
            {
                bias_grad_ = nullptr;
            }
        }

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /**
         * @brief Build the operation for a concrete input shape.
         *
         * Requires:
         *  - setParameters() has already been called so weight/bias pointers are available
         *  - configuration (axis or normalized_shape) is final
         *
         * Allocates backend-owned runtime storage for mean/rstd sized to the
         * outer grouping implied by the input shape and normalized axes.
         */
        void build( const shape_t& input_shape ) override
        {
            // Ensure parameters were bound before build (module must call setParameters first).
            if (weight_ == nullptr)
            {
                throw std::runtime_error( "CpuLayerNormOp::build requires parameters bound via setParameters() before build()." );
            }

            if (config_.hasBias() && bias_ == nullptr)
            {
                throw std::runtime_error( "CpuLayerNormOp::build - bias expected by config but not bound via setParameters()." );
            }

            // Validate input_shape matches configuration
            if (!config_.getNormalizedShape().empty())
            {
                // If normalized_shape configured, ensure input_shape trailing dims match.
                if (input_shape.size() < config_.getNormalizedShape().size())
                {
                    throw std::invalid_argument( "CpuLayerNormOp::build - input rank is less than normalized_shape rank" );
                }

                size_t offset = input_shape.size() - config_.getNormalizedShape().size();
                for (size_t i = 0; i < config_.getNormalizedShape().size(); ++i)
                {
                    if (input_shape[offset + i] != config_.getNormalizedShape()[i])
                    {
                        throw std::invalid_argument( "CpuLayerNormOp::build - input trailing dimensions don't match normalized_shape" );
                    }
                }
            }
            else if (!config_.getAxis().has_value())
            {
                // Neither normalized_shape nor axis available -> cannot determine statistics layout
                throw std::invalid_argument( "CpuLayerNormOp::build - configuration must specify normalized_shape or axis before build()" );
            }

            // Compute outer/inner grouping to size mean/rstd workspace
            const auto& shape = input_shape;
            const int64_t ndim = static_cast<int64_t>(shape.size());

            int64_t axis = -1;
            if (config_.getAxis().has_value())
            {
                axis = config_.getAxis().value();
            }
            else
            {
                // normalize axis to trailing dims: choose first trailing dim index
                axis = static_cast<int64_t>(shape.size()) - static_cast<int64_t>(config_.getNormalizedShape().size());
            }

            if (axis < 0)
                axis += ndim;

            if (axis < 0 || axis >= ndim)
            {
                throw std::invalid_argument( "CpuLayerNormOp::build - computed axis out of range" );
            }

            int64_t outer_size = 1;
            for (int64_t i = 0; i < axis; ++i)
                outer_size *= static_cast<int64_t>( shape[i] );

            int64_t inner_size = 1;
            for (int64_t i = axis + 1; i < ndim; ++i)
                inner_size *= static_cast<int64_t>( shape[i] );

            const int64_t expected_slices = outer_size * inner_size;

            // Allocate backend-owned mean/rstd storage sized per outer grouping.
            mean_storage_.assign( static_cast<size_t>( expected_slices ), float( 0 ) );
            rstd_storage_.assign( static_cast<size_t>( expected_slices ), float( 0 ) );

            mean_ = mean_storage_.data();
            rstd_ = rstd_storage_.data();

            // Mark built
            UnaryOperationBase::build( input_shape );
        }

        // ====================================================================
        // Computation
        // ====================================================================

        /**
         * Forward:
         * - input: ITensor containing input values
         * - output: ITensor to write normalized result
         *
         * Uses cached parameter raw pointers (weight_, bias_) and backend-owned
         * mean/rstd storage allocated during build().
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const float* X = static_cast<const float*>(input.rawData());
            float* Y = static_cast<float*>(output.rawData());

            if (!X || !Y)
            {
                throw std::runtime_error( "CpuLayerNormOp::forward - null tensor data pointer" );
            }

            const float* weight = weight_;
            const float* bias = bias_;

            float* mean = mean_;
            float* rstd = rstd_;

            const auto& shape = input.shape();
            const int64_t ndim = shape.size();

            if (ndim < 1)
            {
                throw std::runtime_error( "CpuLayerNormOp::forward - input must have rank >= 1" );
            }

            if (config_.getAxis().has_value())
            {
                auto partition = computeAxisPartition(
                    shape,
                    config_.getAxis().value(),
                    "CpuLayerNormOp::forward"
                );
                (void)partition;
            }

            auto axis = -1;
            if (axis < 0)
                axis += ndim;

            int64_t outer_size = 1;
            for (int64_t i = 0; i < axis; ++i) outer_size *= static_cast<int64_t>( shape[i] );

            const int64_t dim_size = static_cast<int64_t>( shape[axis] );

            int64_t inner_size = 1;
            for (int64_t i = axis + 1; i < ndim; ++i)
                inner_size *= static_cast<int64_t>( shape[i] );

#pragma omp parallel for collapse(2) if( (size_t)outer_size * (size_t)inner_size > 100 )
            for (int64_t outer = 0; outer < outer_size; ++outer)
            {
                for (int64_t inner = 0; inner < inner_size; ++inner)
                {
                    const float* slice_in = X + (outer * dim_size * inner_size) + inner;
                    float* slice_out = Y + (outer * dim_size * inner_size) + inner;

                    long double m = 0.0L;
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        m += static_cast<long double>( slice_in[i * inner_size] );
                    }
                    m /= static_cast<long double>( dim_size );

                    long double v = 0.0L;
                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        long double diff = static_cast<long double>( slice_in[i * inner_size] ) - m;
                        v += diff * diff;
                    }
                    v /= static_cast<long double>( dim_size );

                    long double s = 1.0L / std::sqrt( v + static_cast<long double>( config_.getEpsilon() ) );

                    for (int64_t i = 0; i < dim_size; ++i)
                    {
                        long double n = s * (static_cast<long double>( slice_in[i * inner_size] ) - m);
                        if (weight)
                        {
                            n *= static_cast<long double>( weight[i] );
                        }
                        if (bias)
                        {
                            n += static_cast<long double>( bias[i] );
                        }
                        slice_out[i * inner_size] = static_cast<float>(n);
                    }

                    const int64_t slice_index = outer * inner_size + inner;
                    if (mean) mean[slice_index] = static_cast<float>(m);
                    if (rstd) rstd[slice_index] = static_cast<float>(s);
                }
            }
        }

        /**
         * Backward:
         * - input: ITensor of original inputs (same as forward input)
         * - output_grad: gradient w.r.t. output (dout)
         * - input_grad: gradient w.r.t. input (accumulated)
         *
         * Parameter gradients are written directly to the pointers provided
         * via setParameterGradients() (weight_grad_, bias_grad_).
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            const float* inp = static_cast<const float*>(input.rawData());
            const float* dout = static_cast<const float*>(output_grad.rawData());
            float* dinp = static_cast<float*>(input_grad.rawData());

            if (!inp || !dout || !dinp)
            {
                throw std::runtime_error( "CpuLayerNormOp::backward - null tensor data pointer" );
            }

            const float* weight = weight_;
            float* dweight = weight_grad_;
            float* dbias = bias_grad_;

            const float* mean = mean_;
            const float* rstd = rstd_;

            const auto& shape = input.shape();
            const int64_t ndim = static_cast<int64_t>(shape.size());

            if (ndim < 1)
            {
                throw std::runtime_error( "CpuLayerNormOp::backward - input must have rank >= 1" );
            }

            if (config_.getAxis().has_value())
            {
                auto partition = computeAxisPartition(
                    shape,
                    config_.getAxis().value(),
                    "CpuLayerNormOp::backward"
                );
                (void)partition;
            }

            int64_t axis = -1;

            if (axis < 0)
                axis += ndim;

            if (axis < 0 || axis >= ndim)
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
                    const float* inp_slice = inp + (outer * dim_size * inner_size) + inner;
                    const float* dout_slice = dout + (outer * dim_size * inner_size) + inner;
                    float* dinp_slice = dinp + (outer * dim_size * inner_size) + inner;

                    const int64_t slice_index = outer * inner_size + inner;
                    long double mean_slice = mean ? static_cast<long double>( mean[slice_index] ) : 0.0L;
                    long double rstd_slice = rstd ? static_cast<long double>( rstd[slice_index] ) : 1.0L;

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
#pragma omp atomic
                            dbias[i] += static_cast<float>( dout_slice[i * inner_size] );
                        }

                        if (dweight)
                        {
#pragma omp atomic
                            dweight[i] += static_cast<float>(norm_bti * static_cast<long double>(dout_slice[i * inner_size]));
                        }

                        long double dval = dnorm_i;
                        dval -= dnorm_mean;
                        dval -= norm_bti * dnorm_norm_mean;
                        dval *= rstd_slice;

#pragma omp atomic
                        dinp_slice[i * inner_size] += static_cast<float>(dval);
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

        // Cached native parameter pointers (module owns underlying tensors)
        float* weight_{ nullptr };
        float* bias_{ nullptr };

        // Cached native parameter gradient pointers (module owns underlying tensors)
        float* weight_grad_{ nullptr };
        float* bias_grad_{ nullptr };

        // Backend-owned runtime statistics storage
        std::vector<float> mean_storage_;
        std::vector<float> rstd_storage_;
        float* mean_{ nullptr };
        float* rstd_{ nullptr };
    };

    // Register CPU LayerNorm (FP32)
    export class CpuLayerNormOpRegistrar
    {
    public:
        static void registerOperations()
        {
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, TensorDataType::FP32, TensorDataType::FP32>(
                "LayerNormOp",
                []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>>
                {
                    const auto& lnConfig = static_cast<const LayerNormConfig&>(config);
                    return std::make_shared<CpuLayerNormOp>( context, lnConfig );
                }
            );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}