/**
 * @file CpuAdamWOptimizer.ixx
 * @brief CPU implementation of AdamW optimizer.
 *
 * Implements the AdamW optimization algorithm for CPU tensors using
 * scalar loops. Maintains per-parameter state tensors (momentum, variance)
 * and performs parameter updates in-place on CPU memory.
 */

module;
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <cstdint>
#include <cmath>
#include <cstring>

export module Compute.CpuAdamWOptimizer;

import Dnn.Optimizers.AdamWConfig;
import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
import Dnn.TensorInitializers;
import Compute.OptimizerBase;
import Compute.DeviceType;
import Compute.CpuMemoryResource;
import Compute.ExecutionContext;
import Compute.CpuExecutionContext;

namespace Mila::Dnn::Compute
{
	using AdamWConfig = Mila::Dnn::Optimizers::AdamWConfig;

    /**
     * @brief CPU-specific AdamW optimizer implementation.
     *
     * Implements the AdamW algorithm using scalar CPU loops.
     * Maintains per-parameter state tensors (first moment, second moment)
     * and performs synchronous parameter updates.
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
     * - FP32 state for numerical stability
     * - Synchronous execution
     *
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, DeviceType::Cpu>
    class CpuAdamWOptimizer : public Optimizer<DeviceType::Cpu, TPrecision>
    {
    public:
        using MR = CpuMemoryResource;
        using TensorType = Tensor<TPrecision, MR>;
        using HostType = typename TensorHostTypeMap<TPrecision>::host_type;
        using ExecutionContextType = ExecutionContext<DeviceType::Cpu>;

        /**
         * @brief Construct CPU AdamW optimizer.
         *
         * @param exec_context CPU execution context
         * @param learning_rate Initial learning rate (typical: 1e-3 to 1e-4)
         * @param beta1 Exponential decay rate for first moment (typical: 0.9)
         * @param beta2 Exponential decay rate for second moment (typical: 0.999)
         * @param epsilon Small constant for numerical stability (typical: 1e-8)
         * @param weight_decay Weight decay coefficient (typical: 0.01)
         *
         * @throws std::invalid_argument if exec_context is null
         * @throws std::invalid_argument if learning_rate <= 0
         * @throws std::invalid_argument if beta1, beta2 not in (0, 1)
         * @throws std::invalid_argument if epsilon <= 0
         */
        explicit CpuAdamWOptimizer( std::shared_ptr<ExecutionContextType> context, const AdamWConfig& config )
            : context_( context ), config_( config )
        {
            if (!context_)
            {
                throw std::invalid_argument( "CpuAdamWOptimizer: ExecutionContext cannot be null" );
            }

			config_.validate();

			// The learning rate can be set and accessed
			learning_rate_ = config_.getLearningRate();
        }

        ~CpuAdamWOptimizer() override = default;

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
         * Allocates momentum and variance state tensors on CPU matching
         * the parameter shape. State tensors are zero-initialized.
         *
         * @param param Parameter tensor to optimize (non-owning, must be on CPU)
         * @param grad Gradient tensor (non-owning, must match param shape and device)
         *
         * @throws std::invalid_argument if param or grad is null
         * @throws std::invalid_argument if param and grad shapes don't match
         * @throws std::invalid_argument if param or grad is not a CPU tensor
         * @throws std::invalid_argument if param or grad data type doesn't match optimizer precision
         * @throws std::runtime_error if state allocation fails
         */
        void addParameter( ITensor* param, ITensor* grad ) override
        {
            if (!param || !grad)
            {
                throw std::invalid_argument( "CpuAdamWOptimizer: parameter and gradient cannot be null" );
            }

            if (param->getDeviceType() != DeviceType::Cpu || grad->getDeviceType() != DeviceType::Cpu)
            {
                throw std::invalid_argument( "CpuAdamWOptimizer: parameters must be CPU tensors" );
            }

            if (param->shape() != grad->shape())
            {
                std::ostringstream oss;
                oss << "CpuAdamWOptimizer: parameter and gradient shape mismatch. "
                    << "Parameter shape: " << shapeToString( param->shape() )
                    << ", Gradient shape: " << shapeToString( grad->shape() );
                throw std::invalid_argument( oss.str() );
            }

            if (param->getDataType() != TPrecision || grad->getDataType() != TPrecision)
            {
                std::ostringstream oss;
                oss << "CpuAdamWOptimizer: parameter/gradient data type mismatch. "
                    << "Expected precision: " << static_cast<int>(TPrecision)
                    << ", Parameter type: " << static_cast<int>(param->getDataType())
                    << ", Gradient type: " << static_cast<int>(grad->getDataType());
                throw std::invalid_argument( oss.str() );
            }

            // Store non-owning pointers to parameters and gradients
            params_.push_back( param );
            grads_.push_back( grad );

            // Cache raw data pointers for hot-path access
            param_data_.push_back( reinterpret_cast<HostType*>(param->rawData()) );
            grad_data_.push_back( reinterpret_cast<const HostType*>(grad->rawData()) );

            // Create optimizer-owned state tensors (always FP32 for numerical stability)
            auto device = context_->getDevice();
            auto shape = param->shape();

            auto m_state = std::make_shared<Tensor<TensorDataType::FP32, MR>>( device, shape );
            m_state->setName( param->getName() + ".m" );
            zeros( *m_state );

            auto v_state = std::make_shared<Tensor<TensorDataType::FP32, MR>>( device, shape );
            v_state->setName( param->getName() + ".v" );
            zeros( *v_state );

            m_states_.push_back( m_state );
            v_states_.push_back( v_state );

            // Cache raw pointers to state tensors for hot-path access
            m_data_.push_back( reinterpret_cast<float*>(m_state->rawData()) );
            v_data_.push_back( reinterpret_cast<float*>(v_state->rawData()) );
        }

        /**
         * @brief Perform one AdamW optimization step.
         *
         * Updates all registered parameters on CPU using scalar loops.
         * Execution is synchronous.
         *
         * @throws std::runtime_error if no parameters have been registered
         *
         * @note Synchronous - blocks until all updates complete
         * @note Increments internal step counter for bias correction
         */
        void step() override
        {
            if (params_.empty())
            {
                throw std::runtime_error( "CpuAdamWOptimizer: no parameters registered" );
            }

            step_count_++;

            // Compute bias correction factors (same for all parameters)
            const float beta1_correction = 1.0f - std::pow( config_.getBeta1(), static_cast<float>(step_count_));
            const float beta2_correction = 1.0f - std::pow( config_.getBeta2(), static_cast<float>(step_count_) );

            // Update each parameter group using cached raw pointers
            for (size_t i = 0; i < params_.size(); ++i)
            {
                updateParameter(
                    param_data_[i],
                    grad_data_[i],
                    m_data_[i],
                    v_data_[i],
                    params_[i]->size(),
                    beta1_correction,
                    beta2_correction
                );
            }
        }

        /**
         * @brief Zero all gradient tensors.
         *
         * Clears all registered gradient tensors on CPU.
         *
         * @throws std::runtime_error if no parameters have been registered
         */
        void zeroGrad() override
        {
            if (grads_.empty())
            {
                throw std::runtime_error( "CpuAdamWOptimizer: no gradients to zero" );
            }

            for (auto* grad : grads_)
            {
                // Zero gradient using memset
                std::memset( grad->rawData(), 0, grad->size() * grad->elementSize() );
            }
        }

        /**
         * @brief Get current learning rate.
         */
        float getLearningRate() const override
        {
            return learning_rate_;
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
                throw std::invalid_argument( "CpuAdamWOptimizer: learning rate must be positive" );
            }

            learning_rate_ = learning_rate;
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
         * @brief Get number of registered parameter groups.
         */
        size_t getParameterCount() const noexcept
        {
            return params_.size();
        }

    private:
        
        std::shared_ptr<ExecutionContextType> context_;
        AdamWConfig config_;

        float learning_rate_;
        
        //float beta1_;
        //float beta2_;
        //float epsilon_;
        //float weight_decay_;

        size_t step_count_{ 0 };

        // Non-owning pointers to parameters and gradients (module owns these)
        std::vector<ITensor*> params_;
        std::vector<ITensor*> grads_;

        // Cached raw data pointers for hot-path access
        std::vector<HostType*> param_data_;
        std::vector<const HostType*> grad_data_;

        // Optimizer-owned state tensors (always FP32 for numerical stability)
        std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, MR>>> m_states_;
        std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, MR>>> v_states_;
        std::vector<float*> m_data_;  // Cached raw pointers
        std::vector<float*> v_data_;  // Cached raw pointers

        

        /**
         * @brief Update a single parameter using AdamW algorithm.
         *
         * Performs the AdamW update for a single parameter tensor using scalar loops.
         * Implements the complete AdamW algorithm including:
         * - First moment (momentum) update
         * - Second moment (RMSprop) update
         * - Bias correction
         * - Parameter update with decoupled weight decay
         *
         * @param param_data Parameter data pointer
         * @param grad_data Gradient data pointer
         * @param m_data First moment state pointer
         * @param v_data Second moment state pointer
         * @param num_params Number of scalar parameters
         * @param beta1_correction Bias correction for first moment (1 - beta1^t)
         * @param beta2_correction Bias correction for second moment (1 - beta2^t)
         */
        void updateParameter(
            HostType* param_data,
            const HostType* grad_data,
            float* m_data,
            float* v_data,
            size_t num_params,
            float beta1_correction,
            float beta2_correction )
        {
            // Precompute constants outside loop
            const float one_minus_beta1 = 1.0f - getBeta1();
            const float one_minus_beta2 = 1.0f - getBeta2();

            for (size_t idx = 0; idx < num_params; ++idx)
            {
                // Get gradient value (convert to float for computation)
                const float grad = static_cast<float>( grad_data[idx] );

                // Update first moment (momentum): m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                float m = getBeta1() * m_data[idx] + one_minus_beta1 * grad;
                m_data[idx] = m;

                // Update second moment (RMSprop): v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                float v = getBeta2() * v_data[idx] + one_minus_beta2 * (grad * grad);
                v_data[idx] = v;

                // Apply bias correction
                const float m_hat = m / beta1_correction;
                const float v_hat = v / beta2_correction;

                // Get current parameter value
                const float old_param = static_cast<float>(param_data[idx]);

                // Compute parameter update
                // AdamW: theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta_{t-1})
                const float adaptive_lr = m_hat / (std::sqrt( v_hat ) + getEpsilon() );
                const float weight_decay_term = getWeightDecay() * old_param;
                const float new_param = old_param - learning_rate_ * (adaptive_lr + weight_decay_term);

                // Write updated parameter (convert back to HostType)
                param_data[idx] = static_cast<HostType>(new_param);
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

    // Convenience aliases
    //export template<TensorDataType TPrecision>/*
    //    using CpuAdamW = CpuAdamWOptimizer<TPrecision>;*/
}