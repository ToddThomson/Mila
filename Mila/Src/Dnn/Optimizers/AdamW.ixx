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
import Dnn.TensorDataTypeTraits;
import Compute.OptimizerBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
#ifdef MILA_HAS_CUDA
import Compute.CudaAdamWOptimizer;
#endif

namespace Mila::Dnn::Optimizers
{
    using namespace Mila::Dnn::Compute;

    namespace Detail
    {
        /**
         * @brief Selects the device-specific AdamW implementation for (device, precision).
         *
         * A missing specialization is a hard compile error naming the pair, matching the
         * OperationTraits convention used elsewhere.
         */
        template<DeviceType TDeviceType, TensorDataType TPrecision>
        struct AdamWImplFor;

#ifdef MILA_HAS_CUDA
        template<TensorDataType TPrecision>
        struct AdamWImplFor<DeviceType::Cuda, TPrecision>
        {
            using type = CudaAdamWOptimizer<TPrecision>;
        };
#endif
    }

    /**
     * @brief Device-agnostic AdamW optimizer.
     *
     * Dispatches to the device-specific implementation selected by `TDeviceType`;
     * CUDA is the only one. Uses `AdamWConfig` for fluent configuration of hyperparameters.
     *
     * @tparam TDeviceType Device type (DeviceType::Cuda)
     * @tparam TPrecision Tensor precision (TensorDataType::FP32, FP16, BF16)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class AdamWOptimizer : public Optimizer<TDeviceType, TPrecision>
    {
    public:
        
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using OptimizerType = typename Detail::AdamWImplFor<TDeviceType, TPrecision>::type;

        /**
         * @brief Construct AdamW optimizer from fluent `AdamWConfig`.
         *
         * @param exec_context Execution context for device resources
         * @param config Fluent AdamWConfig describing hyperparameters
         *
         * @throws std::invalid_argument if exec_context is null
         * @throws std::invalid_argument if config.validate() fails
         */
        explicit AdamWOptimizer( IExecutionContext* exec_context, const AdamWConfig& config )
			: context_( exec_context ), config_( config )
        {
            if (!exec_context)
            {
                throw std::invalid_argument( "AdamWOptimizer: ExecutionContext cannot be null" );
            }

            config.validate();

            impl_ = std::make_shared<OptimizerType>(
                    context_, config_ );
        }

        ~AdamWOptimizer() override = default;

        // ====================================================================
        // Optimizer Interface Implementation
        // ====================================================================

        void addParameter( ITensor* param, ITensor* grad ) override
        {
            impl_->addParameter( param, grad );
        }

        void step() override
        {
            impl_->step();
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
        // AdamW-Specific Interface
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
		IExecutionContext* context_;
        std::shared_ptr<OptimizerType> impl_;
    };
}