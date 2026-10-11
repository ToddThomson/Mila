/**
 * @file AdamW.Cuda.cpp
 * @brief The AdamW optimizer on CUDA, FP32: construction, registration, the update against a host reference,
 * bias correction, convergence and numerical stability.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>

import Mila;

namespace Dnn::Optimizers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Optimizers;

    // The device-agnostic wrapper is what Mila exports; it is the surface a consumer drives.
    using AdamWCuda = AdamWOptimizer<DeviceType::Cuda, TensorDataType::FP32>;

    class AdamWCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            exec_ctx_ = createExecutionContext( Device::Cuda( 0 ) );

            small_shape_ = { 2, 3 };
            medium_shape_ = { 10, 20 };
            large_shape_ = { 100, 200 };
            very_large_shape_ = { 1000, 1000 };

            default_lr_ = 0.001f;
            default_beta1_ = 0.9f;
            default_beta2_ = 0.999f;
            default_epsilon_ = 1e-8f;
            default_weight_decay_ = 0.01f;
        }

        void TearDown() override
        {
            // Synchronize before cleanup
            if (exec_ctx_)
            {
                exec_ctx_->synchronize();
            }

            exec_ctx_.reset();
        }

        /**
         * @brief Create a CUDA parameter tensor with initial values.
         */
        std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>> createParameter(
            const shape_t& shape,
            float init_value = 1.0f )
        {
            auto param = std::make_shared<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>(
                Device::Cuda( 0 ), shape );

            // Initialize via host tensor
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_tensor( Device::Cpu(), shape );
            auto host_data = host_tensor.data();

            for (dim_t i = 0; i < host_tensor.size(); ++i)
            {
                host_data[i] = init_value;
            }

            copy( host_tensor, *param );
            exec_ctx_->synchronize();

            param->setName( "test_param" );
            return param;
        }

        /**
         * @brief Create a CUDA gradient tensor with specified values.
         */
        std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>> createGradient(
            const shape_t& shape,
            float grad_value = 0.1f )
        {
            auto grad = std::make_shared<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>(
                Device::Cuda( 0 ), shape );

            // Initialize via host tensor
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_tensor( Device::Cpu(), shape );
            auto host_data = host_tensor.data();

            for (dim_t i = 0; i < host_tensor.size(); ++i)
            {
                host_data[i] = grad_value;
            }

            copy( host_tensor, *grad );
            exec_ctx_->synchronize();

            grad->setName( "test_grad" );
            return grad;
        }

        /**
         * @brief Copy CUDA tensor to host for verification.
         */
        Tensor<TensorDataType::FP32, CpuMemoryResource> copyToHost(
            const Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>& device_tensor )
        {
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_tensor( Device::Cpu(), device_tensor.shape() );
            copy( device_tensor, host_tensor );
            exec_ctx_->synchronize();

            return host_tensor;
        }

        /**
         * @brief Verify tensor contains no NaN or Inf values (on host after copy).
         */
        bool hasNaNorInf( const Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>& device_tensor )
        {
            auto host_tensor = copyToHost( device_tensor );
            const float* data = host_tensor.data();

            for (dim_t i = 0; i < host_tensor.size(); ++i)
            {
                if (std::isnan( data[i] ) || std::isinf( data[i] ))
                {
                    return true;
                }
            }

            return false;
        }

        /**
         * @brief Check if all elements in tensor are approximately equal to expected value.
         */
        bool allClose(
            const Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>& device_tensor,
            float expected,
            float tolerance = 1e-5f )
        {
            auto host_tensor = copyToHost( device_tensor );
            const float* data = host_tensor.data();

            for (dim_t i = 0; i < host_tensor.size(); ++i)
            {
                if (std::abs( data[i] - expected ) > tolerance)
                {
                    return false;
                }
            }

            return true;
        }

        std::unique_ptr<IExecutionContext> exec_ctx_;
        shape_t small_shape_;
        shape_t medium_shape_;
        shape_t large_shape_;
        shape_t very_large_shape_;

        float default_lr_;
        float default_beta1_;
        float default_beta2_;
        float default_epsilon_;
        float default_weight_decay_;
    };

    // ============================================================================
    // Construction and Configuration Tests
    // ============================================================================

    TEST_F( AdamWCudaTests, Construction_WithConfig )
    {
        auto config = AdamWConfig()
            .withLearningRate( 0.001f )
            .withBeta1( 0.9f )
            .withBeta2( 0.999f )
            .withEpsilon( 1e-8f )
            .withWeightDecay( 0.01f );

        // Use device-agnostic wrapper with AdamWConfig
        auto optimizer = std::make_shared<AdamWOptimizer<DeviceType::Cuda, TensorDataType::FP32>>(
            exec_ctx_.get(), config );
    }

    TEST_F( AdamWCudaTests, Construction_DefaultHyperparameters )
    {
        // Construct device-agnostic optimizer using AdamWConfig defaults
        AdamWConfig config; // default values as in AdamWConfig.ixx

        auto optimizer = std::make_shared<AdamWOptimizer<DeviceType::Cuda, TensorDataType::FP32>>(
            exec_ctx_.get(), config );

        EXPECT_FLOAT_EQ( optimizer->getLearningRate(), default_lr_ );
        EXPECT_FLOAT_EQ( optimizer->getBeta1(), default_beta1_ );
        EXPECT_FLOAT_EQ( optimizer->getBeta2(), default_beta2_ );
        EXPECT_FLOAT_EQ( optimizer->getEpsilon(), default_epsilon_ );
        EXPECT_FLOAT_EQ( optimizer->getWeightDecay(), default_weight_decay_ );
        EXPECT_EQ( optimizer->getStepCount(), 0u );
    }

    TEST_F( AdamWCudaTests, Construction_CustomHyperparameters )
    {
        float lr = 0.002f;
        float beta1 = 0.85f;
        float beta2 = 0.995f;
        float eps = 1e-7f;
        float wd = 0.02f;

        auto config = AdamWConfig()
            .withLearningRate( lr )
            .withBeta1( beta1 )
            .withBeta2( beta2 )
            .withEpsilon( eps )
            .withWeightDecay( wd );

        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        EXPECT_FLOAT_EQ( optimizer->getLearningRate(), lr );
        EXPECT_FLOAT_EQ( optimizer->getBeta1(), beta1 );
        EXPECT_FLOAT_EQ( optimizer->getBeta2(), beta2 );
        EXPECT_FLOAT_EQ( optimizer->getEpsilon(), eps );
        EXPECT_FLOAT_EQ( optimizer->getWeightDecay(), wd );
    }

    TEST_F( AdamWCudaTests, Error_NullExecutionContext )
    {
        IExecutionContext* null_ctx = nullptr;

        auto config = AdamWConfig().withLearningRate( default_lr_ );

        EXPECT_THROW(
            (std::make_shared<AdamWCuda>( null_ctx, config )),
            std::invalid_argument
        );
    }

    TEST_F( AdamWCudaTests, Error_InvalidLearningRate )
    {
        {
            auto config = AdamWConfig().withLearningRate( 0.0f );
            EXPECT_THROW(
                (std::make_shared<AdamWCuda>( exec_ctx_.get(), config )),
                std::invalid_argument
            );
        }

        {
            auto config = AdamWConfig().withLearningRate( -0.001f );
            EXPECT_THROW(
                (std::make_shared<AdamWCuda>( exec_ctx_.get(), config )),
                std::invalid_argument
            );
        }
    }

    TEST_F( AdamWCudaTests, Error_InvalidBeta1 )
    {
        {
            auto config = AdamWConfig().withLearningRate( default_lr_ ).withBeta1( 0.0f );
            EXPECT_THROW(
                (std::make_shared<AdamWCuda>( exec_ctx_.get(), config )),
                std::invalid_argument
            );
        }

        {
            auto config = AdamWConfig().withLearningRate( default_lr_ ).withBeta1( 1.0f );
            EXPECT_THROW(
                (std::make_shared<AdamWCuda>( exec_ctx_.get(), config )),
                std::invalid_argument
            );
        }
    }

    TEST_F( AdamWCudaTests, Error_InvalidBeta2 )
    {
        {
            auto config = AdamWConfig()
                .withLearningRate( default_lr_ )
                .withBeta1( default_beta1_ )
                .withBeta2( 0.0f );
            EXPECT_THROW(
                (std::make_shared<AdamWCuda>( exec_ctx_.get(), config )),
                std::invalid_argument
            );
        }

        {
            auto config = AdamWConfig()
                .withLearningRate( default_lr_ )
                .withBeta1( default_beta1_ )
                .withBeta2( 1.0f );
            EXPECT_THROW(
                (std::make_shared<AdamWCuda>( exec_ctx_.get(), config )),
                std::invalid_argument
            );
        }
    }

    TEST_F( AdamWCudaTests, Error_InvalidEpsilon )
    {
        {
            auto config = AdamWConfig()
                .withLearningRate( default_lr_ )
                .withBeta1( default_beta1_ )
                .withBeta2( default_beta2_ )
                .withEpsilon( 0.0f );
            EXPECT_THROW(
                (std::make_shared<AdamWCuda>( exec_ctx_.get(), config )),
                std::invalid_argument
            );
        }

        {
            auto config = AdamWConfig()
                .withLearningRate( default_lr_ )
                .withBeta1( default_beta1_ )
                .withBeta2( default_beta2_ )
                .withEpsilon( -1e-8f );
            EXPECT_THROW(
                (std::make_shared<AdamWCuda>( exec_ctx_.get(), config )),
                std::invalid_argument
            );
        }
    }

    TEST_F( AdamWCudaTests, Error_InvalidWeightDecay )
    {
        auto config = AdamWConfig()
            .withLearningRate( default_lr_ )
            .withBeta1( default_beta1_ )
            .withBeta2( default_beta2_ )
            .withEpsilon( default_epsilon_ )
            .withWeightDecay( -0.01f );

        EXPECT_THROW(
            (std::make_shared<AdamWCuda>( exec_ctx_.get(), config )),
            std::invalid_argument
        );
    }

    // ============================================================================
    // Parameter Registration Tests
    // ============================================================================

    TEST_F( AdamWCudaTests, AddParameter_SingleParameter )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_ );
        auto grad = createGradient( small_shape_ );

        EXPECT_NO_THROW( optimizer->addParameter( param.get(), grad.get() ) );
        EXPECT_EQ( optimizer->getParameterCount(), 1u );
    }

    TEST_F( AdamWCudaTests, AddParameter_MultipleParameters )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        for (int i = 0; i < 5; ++i)
        {
            auto param = createParameter( small_shape_ );
            auto grad = createGradient( small_shape_ );

            optimizer->addParameter( param.get(), grad.get() );
        }

        EXPECT_EQ( optimizer->getParameterCount(), 5u );
    }

    TEST_F( AdamWCudaTests, AddParameter_DifferentShapes )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param1 = createParameter( small_shape_ );
        auto grad1 = createGradient( small_shape_ );
        optimizer->addParameter( param1.get(), grad1.get() );

        auto param2 = createParameter( medium_shape_ );
        auto grad2 = createGradient( medium_shape_ );
        optimizer->addParameter( param2.get(), grad2.get() );

        EXPECT_EQ( optimizer->getParameterCount(), 2u );
    }

    TEST_F( AdamWCudaTests, Error_AddParameter_NullParameter )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto grad = createGradient( small_shape_ );

        EXPECT_THROW(
            optimizer->addParameter( nullptr, grad.get() ),
            std::invalid_argument
        );
    }

    TEST_F( AdamWCudaTests, Error_AddParameter_NullGradient )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_ );

        EXPECT_THROW(
            optimizer->addParameter( param.get(), nullptr ),
            std::invalid_argument
        );
    }

    TEST_F( AdamWCudaTests, Error_AddParameter_ShapeMismatch )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_ );
        auto grad = createGradient( medium_shape_ );

        EXPECT_THROW(
            optimizer->addParameter( param.get(), grad.get() ),
            std::invalid_argument
        );
    }

    // ============================================================================
    // Optimization Step Tests (CUDA Kernel Execution)
    // ============================================================================

    TEST_F( AdamWCudaTests, Step_SingleParameter_SingleIteration )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.1f );

        optimizer->addParameter( param.get(), grad.get() );

        // Store initial values
        auto initial_host = copyToHost( *param );
        std::vector<float> initial_values( initial_host.size() );
        std::copy_n( initial_host.data(), initial_host.size(), initial_values.begin() );

        EXPECT_NO_THROW( optimizer->step() );
        exec_ctx_->synchronize();

        EXPECT_EQ( optimizer->getStepCount(), 1u );

        // Parameters should have changed
        auto updated_host = copyToHost( *param );
        const float* updated = updated_host.data();

        bool params_changed = false;
        for (dim_t i = 0; i < updated_host.size(); ++i)
        {
            if (std::abs( updated[i] - initial_values[i] ) > 1e-6f)
            {
                params_changed = true;
                break;
            }
        }
        EXPECT_TRUE( params_changed ) << "Parameters should be updated after step()";

        // Parameters should have decreased (gradient descent with positive gradients)
        for (dim_t i = 0; i < updated_host.size(); ++i)
        {
            EXPECT_LT( updated[i], initial_values[i] )
                << "Parameters should decrease with positive gradients";
        }

        // No NaN or Inf
        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCudaTests, Step_MultipleIterations )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.1f );

        optimizer->addParameter( param.get(), grad.get() );

        // Run multiple optimization steps
        for (int iter = 1; iter <= 10; ++iter)
        {
            EXPECT_NO_THROW( optimizer->step() );
            exec_ctx_->synchronize();

            EXPECT_EQ( optimizer->getStepCount(), static_cast<size_t>(iter) );
            EXPECT_FALSE( hasNaNorInf( *param ) );
        }

        // Parameters should have significantly decreased
        auto updated_host = copyToHost( *param );
        const float* updated = updated_host.data();

        for (dim_t i = 0; i < updated_host.size(); ++i)
        {
            EXPECT_LT( updated[i], 1.0f ) << "Parameters should decrease after multiple iterations";
        }
    }

    TEST_F( AdamWCudaTests, Step_MultipleParameters )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>>> grads;

        for (int i = 0; i < 3; ++i)
        {
            auto param = createParameter( small_shape_, 1.0f );
            auto grad = createGradient( small_shape_, 0.1f * (i + 1) );
            optimizer->addParameter( param.get(), grad.get() );
            params.push_back( param );
            grads.push_back( grad );
        }

        EXPECT_NO_THROW( optimizer->step() );
        exec_ctx_->synchronize();

        // All parameters should have been updated
        for (const auto& param : params)
        {
            EXPECT_FALSE( hasNaNorInf( *param ) );

            auto host_tensor = copyToHost( *param );
            const float* data = host_tensor.data();

            bool changed = false;
            for (dim_t j = 0; j < host_tensor.size(); ++j)
            {
                if (std::abs( data[j] - 1.0f ) > 1e-6f)
                {
                    changed = true;
                    break;
                }
            }
            EXPECT_TRUE( changed ) << "All parameters should be updated";
        }
    }

    TEST_F( AdamWCudaTests, Step_LargeParameters )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( large_shape_, 1.0f );
        auto grad = createGradient( large_shape_, 0.01f );

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );
        exec_ctx_->synchronize();

        EXPECT_FALSE( hasNaNorInf( *param ) );
        EXPECT_EQ( optimizer->getStepCount(), 1u );
    }

    TEST_F( AdamWCudaTests, Step_VeryLargeParameters )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( very_large_shape_, 1.0f );
        auto grad = createGradient( very_large_shape_, 0.001f );

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );
        exec_ctx_->synchronize();

        EXPECT_FALSE( hasNaNorInf( *param ) );
        EXPECT_EQ( optimizer->getStepCount(), 1u );
    }

    TEST_F( AdamWCudaTests, Step_WithWeightDecay )
    {
        // High weight decay to see its effect
        auto config = AdamWConfig()
            .withLearningRate( 0.1f )
            .withBeta1( default_beta1_ )
            .withBeta2( default_beta2_ )
            .withEpsilon( default_epsilon_ )
            .withWeightDecay( 0.1f );

        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.0f );  // Zero gradient

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );
        exec_ctx_->synchronize();

        // With zero gradient and weight decay, parameters should decrease
        auto host_tensor = copyToHost( *param );
        const float* data = host_tensor.data();

        for (dim_t i = 0; i < host_tensor.size(); ++i)
        {
            EXPECT_LT( data[i], 1.0f ) << "Weight decay should reduce parameters even with zero gradient";
        }
    }

    TEST_F( AdamWCudaTests, Error_StepWithoutParameters )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        EXPECT_THROW( optimizer->step(), std::runtime_error );
    }

    // ============================================================================
    // Asynchronous Execution Tests
    // ============================================================================

    TEST_F( AdamWCudaTests, Async_MultipleStepsWithoutSync )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( medium_shape_, 1.0f );
        auto grad = createGradient( medium_shape_, 0.01f );

        optimizer->addParameter( param.get(), grad.get() );

        // Launch multiple steps without synchronization
        for (int i = 0; i < 5; ++i)
        {
            EXPECT_NO_THROW( optimizer->step() );  // Asynchronous
        }

        // Synchronize once at the end
        exec_ctx_->synchronize();

        EXPECT_EQ( optimizer->getStepCount(), 5u );
        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCudaTests, Async_StreamOrdering )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.1f );

        optimizer->addParameter( param.get(), grad.get() );

        std::vector<float> values_after_each_step;

        for (int step = 0; step < 5; ++step)
        {
            optimizer->step();
            exec_ctx_->synchronize();  // Sync after each step

            auto host_tensor = copyToHost( *param );
            values_after_each_step.push_back( host_tensor.data()[0] );
        }

        // Values should decrease monotonically
        for (size_t i = 1; i < values_after_each_step.size(); ++i)
        {
            EXPECT_LT( values_after_each_step[i], values_after_each_step[i - 1] )
                << "Parameters should decrease monotonically with positive gradients";
        }
    }

    // ============================================================================
    // Hyperparameter Update Tests
    // ============================================================================

    TEST_F( AdamWCudaTests, SetLearningRate )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        float new_lr = 0.002f;
        EXPECT_NO_THROW( optimizer->setLearningRate( new_lr ) );
        EXPECT_FLOAT_EQ( optimizer->getLearningRate(), new_lr );
    }

    TEST_F( AdamWCudaTests, Error_SetLearningRate_Invalid )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        EXPECT_THROW( optimizer->setLearningRate( 0.0f ), std::invalid_argument );
        EXPECT_THROW( optimizer->setLearningRate( -0.001f ), std::invalid_argument );
    }

    TEST_F( AdamWCudaTests, SetWeightDecay )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        float new_wd = 0.02f;
        EXPECT_NO_THROW( optimizer->setWeightDecay( new_wd ) );
        EXPECT_FLOAT_EQ( optimizer->getWeightDecay(), new_wd );
    }

    TEST_F( AdamWCudaTests, Error_SetWeightDecay_Invalid )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        EXPECT_THROW( optimizer->setWeightDecay( -0.01f ), std::invalid_argument );
    }

    TEST_F( AdamWCudaTests, LearningRateSchedule )
    {
        auto config = AdamWConfig().withLearningRate( 0.1f );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.1f );
        optimizer->addParameter( param.get(), grad.get() );

        // Simulate learning rate decay schedule
        for (int epoch = 0; epoch < 5; ++epoch)
        {
            float lr = 0.1f * std::pow( 0.9f, static_cast<float>( epoch ) );
            optimizer->setLearningRate( lr );
            EXPECT_FLOAT_EQ( optimizer->getLearningRate(), lr );

            optimizer->step();
            exec_ctx_->synchronize();

            EXPECT_FALSE( hasNaNorInf( *param ) );
        }
    }

    // ============================================================================
    // Edge Cases and Numerical Stability Tests
    // ============================================================================

    TEST_F( AdamWCudaTests, EdgeCase_ZeroGradients )
    {
        auto config = AdamWConfig()
            .withLearningRate( default_lr_ )
            .withBeta1( default_beta1_ )
            .withBeta2( default_beta2_ )
            .withEpsilon( default_epsilon_ )
            .withWeightDecay( 0.0f );

        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.0f );

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );
        exec_ctx_->synchronize();

        // With zero gradients and zero weight decay, parameters should not change
        EXPECT_TRUE( allClose( *param, 1.0f, 1e-6f ) );
        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCudaTests, EdgeCase_LargeGradients )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 100.0f );  // Very large gradient

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );
        exec_ctx_->synchronize();

        EXPECT_FALSE( hasNaNorInf( *param ) ) << "AdamW should handle large gradients without NaN/Inf";
    }

    TEST_F( AdamWCudaTests, EdgeCase_VerySmallGradients )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 1e-10f );  // Very small gradient

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );
        exec_ctx_->synchronize();

        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCudaTests, EdgeCase_MixedSignGradientStaysFinite )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 0.0f );
        auto grad = createGradient( small_shape_, 0.0f );

        Tensor<TensorDataType::FP32, CpuMemoryResource> host_grad( Device::Cpu(), small_shape_ );

        for ( dim_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = ( i % 2 == 0 ) ? 0.1f : -0.1f;
        }

        copy( host_grad, *grad );
        exec_ctx_->synchronize();

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );
        exec_ctx_->synchronize();

        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    // Three steps against the AdamW update computed on the host, with a distinct gradient per element and weight
    // decay on, so the moments, both bias corrections and the decay term are each checked rather than assumed.
    TEST_F( AdamWCudaTests, Step_MatchesTheHostReferenceUpdate )
    {
        constexpr float kLearningRate = 0.01f;
        constexpr float kWeightDecay = 0.1f;

        auto config = AdamWConfig().withLearningRate( kLearningRate ).withWeightDecay( kWeightDecay );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        const shape_t shape{ 4, 8 };
        auto param = createParameter( shape, 0.0f );
        auto grad = createGradient( shape, 0.0f );

        Tensor<TensorDataType::FP32, CpuMemoryResource> host_param( Device::Cpu(), shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_grad( Device::Cpu(), shape );

        for ( dim_t i = 0; i < host_param.size(); ++i )
        {
            host_param.data()[ i ] = 0.5f - 0.03f * static_cast<float>( i );
            host_grad.data()[ i ] = 0.2f * std::sin( 0.7f * static_cast<float>( i ) ) + 0.01f;
        }

        copy( host_param, *param );
        copy( host_grad, *grad );
        exec_ctx_->synchronize();

        optimizer->addParameter( param.get(), grad.get() );

        const double beta1 = optimizer->getBeta1();
        const double beta2 = optimizer->getBeta2();
        const double epsilon = optimizer->getEpsilon();

        std::vector<double> p( host_param.data(), host_param.data() + host_param.size() );
        std::vector<double> m( p.size(), 0.0 );
        std::vector<double> v( p.size(), 0.0 );

        for ( int t = 1; t <= 3; ++t )
        {
            optimizer->step();
            exec_ctx_->synchronize();

            const double beta1_correction = 1.0 - std::pow( beta1, t );
            const double beta2_correction = 1.0 - std::pow( beta2, t );

            for ( size_t i = 0; i < p.size(); ++i )
            {
                const double g = host_grad.data()[ i ];
                m[ i ] = beta1 * m[ i ] + ( 1.0 - beta1 ) * g;
                v[ i ] = beta2 * v[ i ] + ( 1.0 - beta2 ) * g * g;

                const double m_hat = m[ i ] / beta1_correction;
                const double v_hat = v[ i ] / beta2_correction;

                p[ i ] -= kLearningRate * ( m_hat / ( std::sqrt( v_hat ) + epsilon ) + kWeightDecay * p[ i ] );
            }

            auto updated = copyToHost( *param );

            for ( size_t i = 0; i < p.size(); ++i )
            {
                EXPECT_NEAR( updated.data()[ i ], static_cast<float>( p[ i ] ), 1e-5f + 1e-5f * static_cast<float>( std::fabs( p[ i ] ) ) )
                    << "step " << t << " element " << i;
            }
        }
    }

    TEST_F( AdamWCudaTests, BiasCorrection_EarlyStepsDiffer )
    {
        auto config = AdamWConfig().withLearningRate( 0.01f );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.1f );

        optimizer->addParameter( param.get(), grad.get() );

        std::vector<float> after_step;

        for ( int step = 0; step < 5; ++step )
        {
            optimizer->step();
            exec_ctx_->synchronize();
            after_step.push_back( copyToHost( *param ).data()[ 0 ] );
        }

        for ( size_t i = 1; i < after_step.size(); ++i )
        {
            EXPECT_NE( after_step[ i ], after_step[ i - 1 ] )
                << "bias correction should produce a distinct update at each early step";
        }
    }

    // Closed loop: the gradient of a known convex objective is recomputed each step, so the parameter must
    // reach the minimizer. Weight decay is off, so the target is the unique fixed point.
    TEST_F( AdamWCudaTests, Convergence_MinimizesQuadraticToTarget )
    {
        const shape_t shape{ 4 };
        const std::vector<float> target{ 0.5f, -1.0f, 2.0f, -0.25f };

        auto config = AdamWConfig().withLearningRate( 0.05f ).withWeightDecay( 0.0f );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( shape, 0.0f );
        auto grad = createGradient( shape, 0.0f );

        optimizer->addParameter( param.get(), grad.get() );

        Tensor<TensorDataType::FP32, CpuMemoryResource> host_grad( Device::Cpu(), shape );

        auto loss = [&]( const Tensor<TensorDataType::FP32, CpuMemoryResource>& x ) {
            float sum = 0.0f;

            for ( size_t i = 0; i < target.size(); ++i )
            {
                const float d = x.data()[ i ] - target[ i ];
                sum += d * d;
            }

            return sum;
        };

        const float initial_loss = loss( copyToHost( *param ) );

        for ( int iteration = 0; iteration < 600; ++iteration )
        {
            auto x = copyToHost( *param );

            for ( size_t i = 0; i < target.size(); ++i )
            {
                host_grad.data()[ i ] = 2.0f * ( x.data()[ i ] - target[ i ] );
            }

            copy( host_grad, *grad );
            exec_ctx_->synchronize();

            optimizer->step();
            exec_ctx_->synchronize();
        }

        const auto x = copyToHost( *param );

        EXPECT_LT( loss( x ), initial_loss ) << "loss must strictly decrease";

        for ( size_t i = 0; i < target.size(); ++i )
        {
            EXPECT_NEAR( x.data()[ i ], target[ i ], 1e-2f ) << "parameter " << i << " should converge to its target";
        }
    }

    TEST_F( AdamWCudaTests, NumericalStability_ManyIterations )
    {
        auto config = AdamWConfig().withLearningRate( 0.01f );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.01f );

        optimizer->addParameter( param.get(), grad.get() );

        // Run many iterations to test numerical stability
        for (int iter = 0; iter < 1000; ++iter)
        {
            optimizer->step();

            if (iter % 100 == 99)
            {
                exec_ctx_->synchronize();
                EXPECT_FALSE( hasNaNorInf( *param ) )
                    << "Numerical instability detected at iteration " << (iter + 1);
            }
        }

        exec_ctx_->synchronize();
        EXPECT_FALSE( hasNaNorInf( *param ) );
        EXPECT_EQ( optimizer->getStepCount(), 1000u );
    }

    // ============================================================================
    // CUDA-Specific Tests
    // ============================================================================

    TEST_F( AdamWCudaTests, Cuda_KernelLaunch )
    {
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( medium_shape_, 1.0f );
        auto grad = createGradient( medium_shape_, 0.1f );

        optimizer->addParameter( param.get(), grad.get() );

        // Kernel launch should succeed without errors
        EXPECT_NO_THROW( optimizer->step() );

        // Check CUDA error status
        cudaError_t err = cudaGetLastError();
        EXPECT_EQ( err, cudaSuccess ) << "CUDA error: " << cudaGetErrorString( err );

        exec_ctx_->synchronize();
        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCudaTests, Cuda_StochasticRounding )
    {
        // Stochastic rounding is used internally for mixed precision
        // This test verifies that the optimizer produces deterministic results
        // for the same inputs (seed is based on step count + parameter index)

        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.1f );

        optimizer->addParameter( param.get(), grad.get() );

        optimizer->step();
        exec_ctx_->synchronize();

        auto result1 = copyToHost( *param );

        // Reset and run again with same inputs
        auto param2 = createParameter( small_shape_, 1.0f );
        auto grad2 = createGradient( small_shape_, 0.1f );

        auto optimizer2 = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        optimizer2->addParameter( param2.get(), grad2.get() );
        optimizer2->step();
        exec_ctx_->synchronize();

        auto result2 = copyToHost( *param2 );

        // Results should be identical (deterministic with same seed)
        for (dim_t i = 0; i < result1.size(); ++i)
        {
            EXPECT_FLOAT_EQ( result1.data()[i], result2.data()[i] )
                << "Results should be deterministic at index " << i;
        }
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    TEST_F( AdamWCudaTests, Integration_TrainingLoop )
    {
        auto config = AdamWConfig().withLearningRate( 0.01f );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param1 = createParameter( small_shape_, 1.0f );
        auto grad1 = createGradient( small_shape_, 0.1f );
        optimizer->addParameter( param1.get(), grad1.get() );

        auto param2 = createParameter( medium_shape_, 0.5f );
        auto grad2 = createGradient( medium_shape_, 0.05f );
        optimizer->addParameter( param2.get(), grad2.get() );

        // Simulate training loop
        for (int epoch = 0; epoch < 10; ++epoch)
        {
            // Update gradients (simulate backward pass)
            float grad_scale = 1.0f - 0.1f * epoch;

            auto grad1_host = copyToHost( *grad1 );
            auto grad2_host = copyToHost( *grad2 );

            for (dim_t i = 0; i < grad1_host.size(); ++i)
            {
                grad1_host.data()[i] = 0.1f * grad_scale;
            }
            for (dim_t i = 0; i < grad2_host.size(); ++i)
            {
                grad2_host.data()[i] = 0.05f * grad_scale;
            }

            copy( grad1_host, *grad1 );
            copy( grad2_host, *grad2 );
            exec_ctx_->synchronize();

            optimizer->step();
            exec_ctx_->synchronize();

            EXPECT_FALSE( hasNaNorInf( *param1 ) );
            EXPECT_FALSE( hasNaNorInf( *param2 ) );
        }

        EXPECT_EQ( optimizer->getStepCount(), 10u );
    }

    TEST_F( AdamWCudaTests, Integration_MultiDeviceCompatibility )
    {
        // Verify optimizer works correctly when execution context manages device
        auto config = AdamWConfig().withLearningRate( default_lr_ );
        auto optimizer = std::make_shared<AdamWCuda>( exec_ctx_.get(), config );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.1f );

        optimizer->addParameter( param.get(), grad.get() );

        // Verify device placement
        EXPECT_EQ( param->getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( grad->getDeviceType(), DeviceType::Cuda );

        EXPECT_NO_THROW( optimizer->step() );
        exec_ctx_->synchronize();

        EXPECT_FALSE( hasNaNorInf( *param ) );
    }
}