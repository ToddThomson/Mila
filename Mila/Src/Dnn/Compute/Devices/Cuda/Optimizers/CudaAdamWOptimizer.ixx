/**
 * @file CudaAdamWOptimizer.ixx
 * @brief CUDA implementation of AdamW optimizer.
 *
 * Bridges the Mila optimizer interface to the AdamW CUDA kernel (adamw.cuh).
 * Manages per-parameter state tensors (momentum, variance) and performs
 * parameter updates using the optimized CUDA kernel implementation.
 */

module;
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <cstdint>
#include <cuda_runtime.h>
#include "Kernels/CudaOptimizers.h"

export module Compute.CudaAdamWOptimizer;

import Dnn.Optimizers.AdamWConfig;
import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorInitializers;
import Compute.OptimizerBase;
import Compute.DeviceType;
import Compute.CudaDeviceMemoryResource;
import Compute.ExecutionContext;
import Compute.CudaExecutionContext;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute
{
	using OptimizerConfig = Mila::Dnn::Optimizers::AdamWConfig;

    /**
     * @brief CUDA-specific AdamW optimizer implementation.
     *
     * Implements the AdamW algorithm using optimized CUDA kernels from adamw.cuh.
     * Maintains per-parameter state tensors (first moment, second moment) on the GPU
     * and performs asynchronous parameter updates via CUDA streams.
     *
     * AdamW algorithm:
     * - m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          (first moment)
     * - v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        (second moment)
     * - m_hat = m_t / (1 - beta1^t)                        (bias correction)
     * - v_hat = v_t / (1 - beta2^t)                        (bias correction)
     * - theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta_{t-1})
     *
     * Features:
     * - Decoupled weight decay (AdamW variant)
     * - Bias correction for moments
     * - Stochastic rounding for mixed precision
     * - Optional master parameters for FP16/BF16 training
     * - Asynchronous execution via CUDA streams
     *
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cuda>
    class CudaAdamWOptimizer : public Optimizer<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using ExecutionContextType = ExecutionContext<DeviceType::Cuda>;

        /**
         * @brief Construct CUDA AdamW optimizer.
         *
         * @param exec_context CUDA execution context for stream and device management
         * @param config AdamW optimizer configuration
         *
         * @throws std::invalid_argument if exec_context is null
         */
        explicit CudaAdamWOptimizer( std::shared_ptr<ExecutionContextType> context, const OptimizerConfig& config )
            : exec_context_( context ), config_( config ), grad_scale_{ 1.0f }, step_count_{ 0 }
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "CudaAdamWOptimizer: ExecutionContext cannot be null" );
            }

            validateHyperparameters();
        }

        ~CudaAdamWOptimizer() override = default;

        // ====================================================================
        // Optimizer Interface Implementation
        // ====================================================================

        /**
         * @brief Register a parameter-gradient pair for optimization.
         *
         * The optimizer does not take ownership of the parameter/gradient tensors.
         * The caller (typically a Module) must ensure the tensors remain valid for
         * the lifetime of the optimizer.
         *
         * Allocates momentum and variance state tensors on the GPU matching
         * the parameter shape. State tensors are zero-initialized.
         *
         * @param param Parameter tensor to optimize (non-owning, must be on CUDA device)
         * @param grad Gradient tensor (non-owning, must match param shape and device)
         *
         * @throws std::invalid_argument if param or grad is null
         * @throws std::invalid_argument if param and grad shapes don't match
         * @throws std::invalid_argument if param or grad is not a CUDA tensor
         * @throws std::invalid_argument if param or grad data type doesn't match optimizer precision
         * @throws std::runtime_error if state allocation fails
         */
        void addParameter( ITensor* param, ITensor* grad ) override
        {
            if (!param || !grad)
            {
                throw std::invalid_argument( "CudaAdamWOptimizer: parameter and gradient cannot be null" );
            }

            if (param->getDeviceType() != DeviceType::Cuda || grad->getDeviceType() != DeviceType::Cuda)
            {
                throw std::invalid_argument( "CudaAdamWOptimizer: parameters must be CUDA tensors" );
            }

            if (param->shape() != grad->shape())
            {
                std::ostringstream oss;
                oss << "CudaAdamWOptimizer: parameter and gradient shape mismatch. "
                    << "Parameter shape: " << shapeToString( param->shape() )
                    << ", Gradient shape: " << shapeToString( grad->shape() );
                throw std::invalid_argument( oss.str() );
            }

            if (param->getDataType() != TPrecision || grad->getDataType() != TPrecision)
            {
                std::ostringstream oss;
                oss << "CudaAdamWOptimizer: parameter/gradient data type mismatch. "
                    << "Expected precision: " << static_cast<int>(TPrecision)
                    << ", Parameter type: " << static_cast<int>(param->getDataType())
                    << ", Gradient type: " << static_cast<int>(grad->getDataType());
                throw std::invalid_argument( oss.str() );
            }

            // Store non-owning pointers to parameters and gradients
            params_.push_back( param );
            grads_.push_back( grad );

            // Cache raw data pointers for hot-path kernel dispatch
            param_data_.push_back( reinterpret_cast<NativeType*>(param->rawData()) );
            grad_data_.push_back( reinterpret_cast<NativeType*>(grad->rawData()) );

            // Create optimizer-owned state tensors (always FP32 for numerical stability)
            auto device = exec_context_->getDevice();
            auto shape = param->shape();

            auto m_state = std::make_shared<Tensor<TensorDataType::FP32, MR>>( device, shape );
            m_state->setName( param->getName() + ".m" );
            zeros( *m_state );

            auto v_state = std::make_shared<Tensor<TensorDataType::FP32, MR>>( device, shape );
            v_state->setName( param->getName() + ".v" );
            zeros( *v_state );

            m_states_.push_back( m_state );
            v_states_.push_back( v_state );

            // Cache raw pointers to state tensors for kernel dispatch
            m_data_.push_back( reinterpret_cast<float*>(m_state->rawData()) );
            v_data_.push_back( reinterpret_cast<float*>(v_state->rawData()) );

            // For mixed precision, optionally create master parameters
            if constexpr (TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16)
            {
                auto master_param = std::make_shared<Tensor<TensorDataType::FP32, MR>>( device, shape );
                master_param->setName( param->getName() + ".master" );

                // Initialize master param from current param values
                // TODO: Implement copy with type conversion
                // For now, initialize to zero
                zeros( *master_param );

                master_params_.push_back( master_param );
                master_param_data_.push_back( reinterpret_cast<float*>(master_param->rawData()) );
            }
        }

        /**
         * @brief Perform one AdamW optimization step.
         *
         * Updates all registered parameters asynchronously on the GPU using
         * the AdamW CUDA kernel. Execution happens on the execution context's
         * CUDA stream.
         *
         * @throws std::runtime_error if no parameters have been registered
         * @throws std::runtime_error if CUDA kernel launch fails
         *
         * @note Asynchronous - returns immediately without GPU synchronization
         * @note Increments internal step counter for bias correction
         */
        void step() override
        {
            if (params_.empty())
            {
                throw std::runtime_error( "CudaAdamWOptimizer: no parameters registered" );
            }

            step_count_++;

            cudaStream_t stream = exec_context_->getStream();

            // Generate seed for stochastic rounding (based on step count)
            unsigned int seed = static_cast<unsigned int>(step_count_);

            // Pull hyperparameters from config for this step
            const float lr = config_.getLearningRate();
            const float b1 = config_.getBeta1();
            const float b2 = config_.getBeta2();
            const float eps = config_.getEpsilon();
            const float wd = config_.getWeightDecay();

            // Update each parameter group using cached raw pointers
            for (size_t i = 0; i < params_.size(); ++i)
            {
                size_t num_params = params_[i]->size();

                NativeType* param_ptr = param_data_[i];
                NativeType* grad_ptr = grad_data_[i];
                
                float* m_ptr = m_data_[i];
                float* v_ptr = v_data_[i];

                float* master_param_ptr = nullptr;
                
                if constexpr (TPrecision == TensorDataType::FP16 || TPrecision == TensorDataType::BF16)
                {
                    if (i < master_param_data_.size())
                    {
                        master_param_ptr = master_param_data_[i];
                    }
                }

                // Call AdamW CUDA kernel
                adamw_update(
                    param_ptr,
                    master_param_ptr,
                    grad_ptr,
                    m_ptr,
                    v_ptr,
                    num_params,
                    0,  // w_stride (single tensor, not batched)
                    0,  // g_stride
                    0,  // s_stride
                    1,  // num_slices (single tensor per call)
                    lr,
                    b1,
                    b2,
                    static_cast<int>(step_count_),
                    eps,
                    wd,
                    grad_scale_,
                    seed + static_cast<unsigned int>(i),  // Unique seed per parameter
                    stream
                );
            }

            // Note: No synchronization - caller can sync via exec_context_->synchronize() if needed
        }

        /**
         * @brief Zero all gradient tensors.
         *
         * Asynchronously clears all registered gradient tensors on the GPU.
         *
         * @throws std::runtime_error if no parameters have been registered
         */
        void zeroGrad() override
        {
            if (grads_.empty())
            {
                throw std::runtime_error( "CudaAdamWOptimizer: no gradients to zero." );
            }

            for (auto* grad : grads_)
            {
                // The grad must be valid!
				auto total_size = grad->size() * grad->elementSize();
                cudaMemsetAsync( grad->rawData(), 0, total_size, exec_context_->getStream() );
            }
        }

        /**
         * @brief Get current learning rate.
         */
        float getLearningRate() const override
        {
            return config_.getLearningRate();
        }

        /**
         * @brief Set learning rate for future steps.
         *
         * @param learning_rate New learning rate (must be positive)
         * @throws std::invalid_argument if learning_rate <= 0
         */
        void setLearningRate( float learning_rate ) override
        {
            if (learning_rate <= 0.0f)
            {
                throw std::invalid_argument( "CudaAdamWOptimizer: learning rate must be positive" );
            }

            config_.withLearningRate( learning_rate );
        }

        // ====================================================================
        // AdamW-Specific Configuration
        // ====================================================================

        /**
         * @brief Get current step count.
         */
        size_t getStepCount() const noexcept
        {
            return step_count_;
        }

        /**
         * @brief Get beta1 parameter.
         */
        float getBeta1() const noexcept
        {
            return config_.getBeta1();
        }

        /**
         * @brief Get beta2 parameter.
         */
        float getBeta2() const noexcept
        {
            return config_.getBeta2();
        }

        /**
         * @brief Get epsilon parameter.
         */
        float getEpsilon() const noexcept
        {
            return config_.getEpsilon();
        }

        /**
         * @brief Get weight decay parameter.
         */
        float getWeightDecay() const noexcept
        {
            return config_.getWeightDecay();
        }

        /**
         * @brief Set weight decay coefficient.
         *
         * @param weight_decay New weight decay (must be non-negative)
         * @throws std::invalid_argument if weight_decay < 0
         */
        void setWeightDecay( float weight_decay )
        {
            if (weight_decay < 0.0f)
            {
                throw std::invalid_argument( "CudaAdamWOptimizer: weight decay must be non-negative" );
            }

            config_.withWeightDecay( weight_decay );
        }

        /**
         * @brief Get number of registered parameter groups.
         */
        size_t getParameterCount() const noexcept
        {
            return params_.size();
        }

    private:
        std::shared_ptr<ExecutionContextType> exec_context_;
		OptimizerConfig config_;

        float grad_scale_{ 1.0f };
        
        size_t step_count_;

        // Non-owning pointers to parameters and gradients (module owns these)
        std::vector<ITensor*> params_;
        std::vector<ITensor*> grads_;

        // Cached raw data pointers for hot-path kernel dispatch
        std::vector<NativeType*> param_data_;
        std::vector<NativeType*> grad_data_;

        // Optimizer-owned state tensors (always FP32 for numerical stability)
        std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, MR>>> m_states_;
        std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, MR>>> v_states_;
        std::vector<float*> m_data_;  // Cached raw pointers
        std::vector<float*> v_data_;  // Cached raw pointers

        // Optional master parameters for mixed precision (FP16/BF16)
        std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, MR>>> master_params_;
        std::vector<float*> master_param_data_;  // Cached raw pointers

        /**
         * @brief Validate optimizer hyperparameters.
         */
        void validateHyperparameters() const
        {
            const float lr = config_.getLearningRate();
            const float b1 = config_.getBeta1();
            const float b2 = config_.getBeta2();
            const float eps = config_.getEpsilon();
            const float wd = config_.getWeightDecay();

            if (lr <= 0.0f)
            {
                throw std::invalid_argument( "CudaAdamWOptimizer: learning rate must be positive" );
            }

            if (b1 <= 0.0f || b1 >= 1.0f)
            {
                throw std::invalid_argument( "CudaAdamWOptimizer: beta1 must be in (0, 1)" );
            }

            if (b2 <= 0.0f || b2 >= 1.0f)
            {
                throw std::invalid_argument( "CudaAdamWOptimizer: beta2 must be in (0, 1)" );
            }

            if (eps <= 0.0f)
            {
                throw std::invalid_argument( "CudaAdamWOptimizer: epsilon must be positive" );
            }

            if (wd < 0.0f)
            {
                throw std::invalid_argument( "CudaAdamWOptimizer: weight decay must be non-negative" );
            }
        }

        /**
         * @brief Convert shape to string for error messages.
         */
        static std::string shapeToString( const shape_t& shape )
        {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < shape.size(); ++i)
            {
                oss << shape[i];
                if (i + 1 < shape.size())
                    oss << ", ";
            }
            oss << "]";
            return oss.str();
        }
    };
}