/**
 * @file AdamW.ixx
 * @brief AdamW optimizer wrapper using fluent `AdamWConfig`.
 */

module;
#include <memory>
#include <type_traits>
#include <stdexcept>

export module Dnn.Optimizers.AdamW;

import Dnn.Optimizers.AdamWConfig;
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
     * Dispatches to the appropriate device-specific implementation (CPU or CUDA)
     * based on the `TDeviceType` template parameter. Uses `AdamWConfig` for
     * fluent configuration of hyperparameters.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Tensor precision (TensorDataType::FP32, FP16, BF16)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class AdamWOptimizer : public Optimizer<TDeviceType, TPrecision>
    {
    public:
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using OptimizerType = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaAdamWOptimizer<TPrecision>, CpuAdamWOptimizer<TPrecision>>;

        /**
         * @brief Construct AdamW optimizer from fluent `AdamWConfig`.
         *
         * @param exec_context Execution context for device resources
         * @param config Fluent AdamWConfig describing hyperparameters
         *
         * @throws std::invalid_argument if exec_context is null
         * @throws std::invalid_argument if config.validate() fails
         */
        explicit AdamWOptimizer( std::shared_ptr<ExecutionContextType> exec_context, const AdamWConfig& config )
			: context_( exec_context ), config_( config )
        {
            if (!exec_context)
            {
                throw std::invalid_argument( "AdamWOptimizer: ExecutionContext cannot be null" );
            }

            config.validate();

            const float lr = config.getLearningRate();
            const float beta1 = config.getBeta1();
            const float beta2 = config.getBeta2();
            const float eps = config.getEpsilon();
            const float wd = config.getWeightDecay();

            impl_ = std::make_shared<OptimizerType>(
                    context_, config_ );
        }

        ~AdamWOptimizer() override = default;

        // ====================================================================
        // Optimizer Interface Implementation (Delegation)
        // ====================================================================

        void addParameter( ITensor* param, ITensor* grad ) override
        {
            impl_->addParameter( param, grad );
        }

        void step() override
        {
            impl_->step();
        }

        void zeroGrad() override
        {
            impl_->zeroGrad();
        }

        float getLearningRate() const override
        {
            return impl_->getLearningRate();
        }

        void setLearningRate( float learning_rate ) override
        {
            impl_->setLearningRate( learning_rate );
        }

        // ====================================================================
        // AdamW-Specific Interface (Pass-through)
        // ====================================================================

        size_t getStepCount() const noexcept
        {
            return impl_->getStepCount();
        }

        float getBeta1() const noexcept
        {
            return impl_->getBeta1();
        }

        float getBeta2() const noexcept
        {
            return impl_->getBeta2();
        }

        float getEpsilon() const noexcept
        {
            return impl_->getEpsilon();
        }

        float getWeightDecay() const noexcept
        {
            return impl_->getWeightDecay();
        }

        void setWeightDecay( float weight_decay )
        {
            impl_->setWeightDecay( weight_decay );
        }

        size_t getParameterCount() const noexcept
        {
            return impl_->getParameterCount();
        }

    private:
		AdamWConfig config_;
		std::shared_ptr<ExecutionContextType> context_;
        std::shared_ptr<OptimizerType> impl_;
    };

    //export template<TensorDataType TPrecision>
    //    using CpuAdamWOptimizerFP32 = AdamWOptimizer<DeviceType::Cpu, TPrecision>;

    //export template<TensorDataType TPrecision>
    //    using CudaAdamWOptimizerFP32 = AdamWOptimizer<DeviceType::Cuda, TPrecision>;
}