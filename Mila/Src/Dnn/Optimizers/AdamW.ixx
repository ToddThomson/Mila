/**
 * @file AdamWOptimizer.ixx
 * @brief Device-agnostic AdamW optimizer.
 *
 * Provides a unified interface for the AdamW optimization algorithm that
 * automatically dispatches to CPU or CUDA implementations based on the
 * device type template parameter.
 *
 * Usage:
 * @code
 * // Automatically uses CUDA implementation
 * auto cuda_optimizer = std::make_shared<AdamWOptimizer<DeviceType::Cuda, TensorDataType::FP32>>(
 *     exec_context, learning_rate, beta1, beta2, epsilon, weight_decay);
 *
 * // Automatically uses CPU implementation
 * auto cpu_optimizer = std::make_shared<AdamWOptimizer<DeviceType::Cpu, TensorDataType::FP32>>(
 *     exec_context, learning_rate, beta1, beta2, epsilon, weight_decay);
 * @endcode
 */

module;
#include <memory>
#include <type_traits>

export module Dnn.Optimizers.AdamW;
export import :Config;

import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Compute.OptimizerBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CpuExecutionContext;
import Compute.CudaExecutionContext;
import Compute.CpuAdamWOptimizer;
import Compute.CudaAdamWOptimizer;

namespace Mila::Dnn::Optimizers
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Device-agnostic AdamW optimizer.
     *
     * Wrapper class that dispatches to the appropriate device-specific
     * AdamW implementation (CPU or CUDA) based on the TDeviceType template
     * parameter. Provides a uniform interface regardless of device.
     *
     * This class uses template specialization to select the correct implementation
     * at compile time, ensuring zero runtime overhead for device dispatch.
     *
     * AdamW Algorithm Features:
     * - Adaptive learning rates per parameter
     * - Momentum (first moment estimation)
     * - RMSprop-style variance (second moment estimation)
     * - Bias correction for moments
     * - Decoupled weight decay (AdamW vs Adam)
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Tensor precision (TensorDataType::FP32, FP16, BF16)
     *
     * @see CpuAdamWOptimizer
     * @see CudaAdamWOptimizer
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class AdamWOptimizer : public Optimizer<TDeviceType, TPrecision>
    {
    public:
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using ImplType = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaAdamWOptimizer<TPrecision>, CpuAdamWOptimizer<TPrecision>>;

        /**
         * @brief Construct AdamW optimizer.
         *
         * Creates the appropriate device-specific implementation (CPU or CUDA)
         * based on the TDeviceType template parameter.
         *
         * @param exec_context Execution context for device resources
         * @param learning_rate Initial learning rate (typical: 1e-3 to 1e-4)
         * @param beta1 First moment decay rate (typical: 0.9)
         * @param beta2 Second moment decay rate (typical: 0.999)
         * @param epsilon Numerical stability constant (typical: 1e-8)
         * @param weight_decay Weight decay coefficient (typical: 0.01)
         *
         * @throws std::invalid_argument if exec_context is null
         * @throws std::invalid_argument if hyperparameters are invalid
         *
         * Example:
         * @code
         * auto exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>(0);
         * auto optimizer = std::make_shared<AdamWOptimizer<DeviceType::Cuda, TensorDataType::FP32>>(
         *     exec_ctx,
         *     0.001f,  // learning_rate
         *     0.9f,    // beta1
         *     0.999f,  // beta2
         *     1e-8f,   // epsilon
         *     0.01f    // weight_decay
         * );
         * @endcode
         */
        explicit AdamWOptimizer(
            std::shared_ptr<ExecutionContextType> exec_context,
            float learning_rate,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float epsilon = 1e-8f,
            float weight_decay = 0.01f )
        {
            if constexpr (TDeviceType == DeviceType::Cuda)
            {
                // CUDA implementation includes grad_scale parameter
                impl_ = std::make_shared<ImplType>(
                    exec_context,
                    learning_rate,
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay,
                    1.0f  // grad_scale (default, can be exposed if needed)
                );
            }
            else
            {
                // CPU implementation
                impl_ = std::make_shared<ImplType>(
                    exec_context,
                    learning_rate,
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay
                );
            }
        }

        ~AdamWOptimizer() override = default;

        // ====================================================================
        // Optimizer Interface Implementation (Delegation)
        // ====================================================================

        /**
         * @brief Register a parameter-gradient pair for optimization.
         *
         * Delegates to the device-specific implementation.
         * 
         * @param param Pointer to parameter tensor (non-owning)
         * @param grad Pointer to gradient tensor (non-owning)
         */
        void addParameter( ITensor* param, ITensor* grad ) override
        {
            impl_->addParameter( param, grad );
        }

        /**
         * @brief Perform one optimization step.
         *
         * Delegates to the device-specific implementation.
         */
        void step() override
        {
            impl_->step();
        }

        /**
         * @brief Zero all gradient tensors.
         *
         * Delegates to the device-specific implementation.
         */
        void zeroGrad() override
        {
            impl_->zeroGrad();
        }

        /**
         * @brief Get current learning rate.
         *
         * Delegates to the device-specific implementation.
         */
        float getLearningRate() const override
        {
            return impl_->getLearningRate();
        }

        /**
         * @brief Set learning rate for future steps.
         *
         * Delegates to the device-specific implementation.
         */
        void setLearningRate( float learning_rate ) override
        {
            impl_->setLearningRate( learning_rate );
        }

        // ====================================================================
        // AdamW-Specific Interface (Pass-through)
        // ====================================================================

        /**
         * @brief Get current step count.
         *
         * Returns the number of optimization steps performed.
         */
        size_t getStepCount() const noexcept
        {
            return impl_->getStepCount();
        }

        /**
         * @brief Get beta1 parameter.
         *
         * Returns the first moment decay rate.
         */
        float getBeta1() const noexcept
        {
            return impl_->getBeta1();
        }

        /**
         * @brief Get beta2 parameter.
         *
         * Returns the second moment decay rate.
         */
        float getBeta2() const noexcept
        {
            return impl_->getBeta2();
        }

        /**
         * @brief Get epsilon parameter.
         *
         * Returns the numerical stability constant.
         */
        float getEpsilon() const noexcept
        {
            return impl_->getEpsilon();
        }

        /**
         * @brief Get weight decay parameter.
         *
         * Returns the weight decay coefficient used in updates.
         */
        float getWeightDecay() const noexcept
        {
            return impl_->getWeightDecay();
        }

        /**
         * @brief Set weight decay coefficient.
         *
         * @param weight_decay New weight decay (must be non-negative)
         * @throws std::invalid_argument if weight_decay < 0
         */
        void setWeightDecay( float weight_decay )
        {
            impl_->setWeightDecay( weight_decay );
        }

        /**
         * @brief Get number of registered parameter groups.
         *
         * Returns the count of parameter tensors being optimized.
         */
        size_t getParameterCount() const noexcept
        {
            return impl_->getParameterCount();
        }

    private:
        std::shared_ptr<ImplType> impl_;
    };

    // ====================================================================
    // Convenience Type Aliases
    // ====================================================================

    /**
     * @brief CPU AdamW optimizer with FP32 precision.
     */
    export using CpuAdamWOptimizerFP32 = AdamWOptimizer<DeviceType::Cpu, TensorDataType::FP32>;

    /**
     * @brief CUDA AdamW optimizer with FP32 precision.
     */
    export using CudaAdamWOptimizerFP32 = AdamWOptimizer<DeviceType::Cuda, TensorDataType::FP32>;

    /**
     * @brief CUDA AdamW optimizer with FP16 precision.
     */
    export using CudaAdamWOptimizerFP16 = AdamWOptimizer<DeviceType::Cuda, TensorDataType::FP16>;

    /**
     * @brief CUDA AdamW optimizer with BF16 precision.
     */
    export using CudaAdamWOptimizerBF16 = AdamWOptimizer<DeviceType::Cuda, TensorDataType::BF16>;
}