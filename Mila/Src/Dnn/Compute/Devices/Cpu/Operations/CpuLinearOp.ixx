/**
 * @file CpuLinearOp.ixx
 * @brief CPU implementation of Linear (fully connected) operation (TensorDataType-based).
 *
 * Ported to the ExecutionContext / TensorDataType UnaryOperation interface
 * following the two-phase initialization pattern used by CpuLayerNormOp.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <numeric>
#include <functional>
#include <cstdint>
#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuLinearOp;

import Dnn.Components.Linear;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
import Dnn.ComponentConfig;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
//import Compute.CpuExecutionContext;
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
     * @brief CPU implementation of Linear operation using abstract TensorDataType API.
     *
     * Template parameter TPrecision selects the abstract tensor precision (e.g. FP32).
     * float is the corresponding CPU host representation for that precision.
     *
     * Design philosophy:
     * - Two-phase initialization: build() does all setup, forward()/backward() are pure dispatch
     * - Module owns weight/bias parameters and binds them via setParameters()
     * - All dimension computation happens once in build()
     * - Forward/backward are hot-path methods with minimal overhead
     * - Implements: y = x * W^T + b where W is (out_features, in_features)
     *
     * Forward: Y = X * W^T + b
     * Backward:
     *  - dX += dY * W
     *  - dW += dY^T * X
     *  - db += sum(dY)
     */
    export class CpuLinearOp : public UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using MR = CpuMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorType = Tensor<TensorDataType::FP32, MR>;
        //using Parameters = std::vector<std::shared_ptr<TensorType>>;
        //using OutputState = std::vector<std::shared_ptr<TensorType>>;
        //using float = typename TensorfloatMap<TPrecision>::host_type;
        using CpuExecutionContext = ExecutionContext<DeviceType::Cpu>;

        CpuLinearOp( std::shared_ptr<CpuExecutionContext> context, const LinearConfig& config )
            : context_( context ), config_( config )
        {
            if (!context_)
            {
                throw std::runtime_error( "CpuLinearOp requires a CPU execution context" );
            }

            config_.validate();
        }

        // ====================================================================
		// Parameters and Gradients
        // ====================================================================

        /**
         * @brief Set parameter tensor references (module remains owner).
         *
         * The operation caches native host pointers for hot-path access. The
         * weight tensor is required; bias is bound only when the Linear
         * config indicates a bias is present.
         *
         * Note: build() requires parameters to be bound before it is called.
         */
        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            if (!weight)
            {
                throw std::invalid_argument( "CpuLinearOp::setParameters - weight parameter is required" );
            }

            weight_ = static_cast<const float*>(weight->rawData());

            // Validate weight is 2D
            const auto& weight_shape = weight->shape();
            if (weight_shape.size() != 2)
            {
                throw std::invalid_argument( "CpuLinearOp::setParameters - weight must be 2D tensor" );
            }

            // Store weight dimensions for validation
            weight_out_features_ = weight_shape[0];
            weight_in_features_ = weight_shape[1];

            if (config_.hasBias())
            {
                if (!bias)
                {
                    throw std::invalid_argument( "CpuLinearOp::setParameters - bias parameter expected but null was provided" );
                }

                bias_ = static_cast<const float*>(bias->rawData());
            }
            else
            {
                bias_ = nullptr;
            }
        }

        void setGradients( ITensor* weight_grad, ITensor* bias_grad ) override
        {
            if (!weight_grad)
            {
                throw std::invalid_argument( "CpuLinearOp::setParameterGradients - weight gradient is required" );
            }

            weight_grad_ = static_cast<float*>(weight_grad->rawData());

            if (config_.hasBias())
            {
                if (!bias_grad)
                {
                    throw std::invalid_argument( "CpuLinearOp::setParameterGradients - bias gradient expected but null was provided" );
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
         * This is the COLD PATH where all setup, validation, and computation happens ONCE.
         * After build() completes, forward() and backward() become pure dispatch methods.
         *
         * Responsibilities:
         *  1. Validate parameters were bound via setParameters()
         *  2. Validate input shape compatibility with weight dimensions
         *  3. Compute and cache batch size and feature dimensions
         *  4. Determine optimal loop unrolling strategy
         *  5. Cache OMP parallelization threshold
         *
         * After build(), the operation is ready for zero-overhead forward/backward dispatch.
         */
        void build( const shape_t& input_shape ) override
        {
            if (weight_ == nullptr)
            {
                throw std::runtime_error( "CpuLinearOp::build requires parameters bound via setParameters() before build()." );
            }

            if (config_.hasBias() && bias_ == nullptr)
            {
                throw std::runtime_error( "CpuLinearOp::build - bias expected by config but not bound via setParameters()." );
            }

            if (input_shape.empty())
            {
                throw std::invalid_argument( "CpuLinearOp::build - input shape cannot be empty" );
            }

            // Extract dimensions: input is (..., in_features)
            in_features_ = input_shape.back();

            // Validate weight dimensions match configuration
            if (weight_out_features_ != config_.getOutputFeatures())
            {
                std::ostringstream oss;
                oss << "CpuLinearOp::build - weight output features mismatch. Expected "
                    << config_.getOutputFeatures() << ", got " << weight_out_features_;
                throw std::invalid_argument( oss.str() );
            }

            if (weight_in_features_ != in_features_)
            {
                std::ostringstream oss;
                oss << "CpuLinearOp::build - weight input features mismatch. Expected "
                    << in_features_ << ", got " << weight_in_features_;
                throw std::invalid_argument( oss.str() );
            }

            // Compute batch size (flatten all dimensions except last)
            batch_size_ = 1;
            for (size_t i = 0; i + 1 < input_shape.size(); ++i)
            {
                batch_size_ *= input_shape[i];
            }

            out_features_ = config_.getOutputFeatures();

            // Determine loop unrolling strategy
            use_loop_unroll_ = (batch_size_ % LOOP_UNROLL == 0);

            // Cache OMP parallelization threshold
            enable_omp_ = (batch_size_ > 100);

            UnaryOperationBase::build( input_shape );
        }

        // ====================================================================
        // Computation
        // ====================================================================

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to CPU kernel.
         *
         * All setup, validation, and dimension computation was done in build().
         * This method extracts raw pointers and dispatches directly to the
         * optimized matrix multiplication kernel using pre-computed cached dimensions.
         *
         * Algorithm: Y = X * W^T + b
         * Zero redundant work - maximum performance.
         */
        void forward( const ITensor& input, ITensor& output ) const override
        {
            const float* X = static_cast<const float*>(input.rawData());
            float* Y = static_cast<float*>(output.rawData());

            const float* W = weight_;
            const float* B = bias_;

            if (use_loop_unroll_)
            {
                forwardUnrolled( X, Y, W, B, batch_size_, in_features_, out_features_ );
            }

            else
            {
                forwardNaive( X, Y, W, B, batch_size_, in_features_, out_features_ );
            }
        }

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to CPU kernel.
         *
         * Similar to forward(), this method does minimal work and dispatches
         * directly to the backward kernel using cached dimensions from build().
         *
         * Gradients are accumulated into the tensors provided via setParameterGradients().
         *
         * Algorithm:
         *  - dX += dY * W
         *  - dW += dY^T * X
         *  - db += sum(dY)
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) const override
        {
            const float* X = static_cast<const float*>(input.rawData());
            const float* dY = static_cast<const float*>(output_grad.rawData());
            float* dX = static_cast<float*>(input_grad.rawData());

            const float* W = weight_;
            float* dW = weight_grad_;
            float* dB = bias_grad_;

#pragma omp parallel for if(enable_omp_)
            for (int64_t idx = 0; idx < batch_size_; ++idx)
            {
                const int64_t in_base = idx * in_features_;
                const int64_t out_base = idx * out_features_;

                // Compute dX for this batch element
                for (int64_t i = 0; i < in_features_; ++i)
                {
                    long double acc = 0.0L;
                    for (int64_t o = 0; o < out_features_; ++o)
                    {
                        acc += static_cast<long double>( W[o * in_features_ + i] ) *
                            static_cast<long double>( dY[out_base + o] );
                    }

#pragma omp atomic
                    dX[in_base + i] += static_cast<float>( acc );
                }

                // Accumulate dW and dB
                for (int64_t o = 0; o < out_features_; ++o)
                {
                    long double dy = static_cast<long double>( dY[out_base + o] );

                    if (dB)
                    {
#pragma omp atomic
                        dB[o] += static_cast<float>( dy );
                    }

                    if (dW)
                    {
                        for (int64_t i = 0; i < in_features_; ++i)
                        {
#pragma omp atomic
                            dW[o * in_features_ + i] += static_cast<float>(
                                static_cast<long double>( X[in_base + i] ) * dy );
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

        const LinearConfig& getConfig() const
        {
            return config_;
        }

    private:
        static constexpr int LOOP_UNROLL = 8;

        LinearConfig config_;
        std::shared_ptr<CpuExecutionContext> context_;

        const float* weight_{ nullptr };
        const float* bias_{ nullptr };

		// Cached native host parameter gradient pointers
		float* weight_grad_{ nullptr };
		float* bias_grad_{ nullptr };

        // Weight dimensions for validation
        int64_t weight_out_features_{ 0 };
        int64_t weight_in_features_{ 0 };

        // Dimension values computed once in build() for hot-path dispatch
        int64_t batch_size_{ 0 };
        int64_t in_features_{ 0 };
        int64_t out_features_{ 0 };
        
        bool use_loop_unroll_{ false };
        bool enable_omp_{ false };

        /**
         * @brief Naive forward implementation without loop unrolling.
         *
         * Used when batch size doesn't align with unroll factor.
         */
        void forwardNaive(
            const float* X, float* Y,
            const float* W, const float* B,
            int64_t batch_size, int64_t in_features, int64_t out_features ) const
        {
#pragma omp parallel for if(enable_omp_)
            for (int64_t idx = 0; idx < batch_size; ++idx)
            {
                const int64_t in_base = idx * in_features;
                const int64_t out_base = idx * out_features;

                for (int64_t o = 0; o < out_features; ++o)
                {
                    long double acc = 0.0L;

                    for (int64_t i = 0; i < in_features; ++i)
                    {
                        acc += static_cast<long double>( X[in_base + i] ) *
                            static_cast<long double>( W[o * in_features + i] );
                    }

                    if (B)
                        acc += static_cast<long double>( B[o] );

                    Y[out_base + o] = static_cast<float>( acc );
                }
            }
        }

        /**
         * @brief Optimized forward implementation with loop unrolling.
         *
         * Processes LOOP_UNROLL batch elements simultaneously for better cache utilization.
         */
        void forwardUnrolled(
            const float* X, float* Y,
            const float* W, const float* B,
            int64_t batch_size, int64_t in_features, int64_t out_features ) const
        {
#pragma omp parallel for if(enable_omp_)
            for (int64_t out_idx = 0; out_idx < batch_size; out_idx += LOOP_UNROLL)
            {
                for (int64_t o = 0; o < out_features; ++o)
                {
                    float result[LOOP_UNROLL];

                    // Initialize with bias
                    for (int i_idx = 0; i_idx < LOOP_UNROLL; ++i_idx)
                    {
                        result[i_idx] = B ? B[o] : static_cast<float>( 0 );
                    }

                    // Accumulate weighted inputs
                    for (int64_t i = 0; i < in_features; ++i)
                    {
                        float w = W[o * in_features + i];

                        for (int i_idx = 0; i_idx < LOOP_UNROLL; ++i_idx)
                        {
                            int64_t idx = out_idx + i_idx;
                            result[i_idx] += X[idx * in_features + i] * w;
                        }
                    }

                    // Store results
                    for (int i_idx = 0; i_idx < LOOP_UNROLL; ++i_idx)
                    {
                        int64_t idx = out_idx + i_idx;
                        Y[idx * out_features + o] = result[i_idx];
                    }
                }
            }
        }
    };

    export class CpuLinearOpRegistrar
    {
    public:
        static void registerOperations()
        {
            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cpu, TensorDataType::FP32, TensorDataType::FP32>(
                "LinearOp",
                []( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
                    const ComponentConfig& config ) -> std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>>
                {
                    const auto& linearConfig = static_cast<const LinearConfig&>(config);
                    return std::make_shared<CpuLinearOp>( context, linearConfig );
                }
            );
        }

        static inline bool isRegistered = []() {
            registerOperations();
            return true;
            }();
    };
}