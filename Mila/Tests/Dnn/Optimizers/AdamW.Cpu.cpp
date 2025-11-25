/**
 * @file AdamW.Cpu.cpp
 * @brief CPU-specific unit tests for AdamW optimizer.
 *
 * Tests the CPU implementation of the AdamW optimization algorithm including:
 * - Configuration and construction
 * - Parameter registration
 * - Single-step updates
 * - Multi-iteration training
 * - Hyperparameter updates
 * - Gradient zeroing
 * - Edge cases and error conditions
 */

#include <gtest/gtest.h>
#include <memory>
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

    /**
     * @brief Test fixture for CPU AdamW optimizer tests.
     */
    class AdamWCpuTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            exec_ctx_ = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

            small_shape_ = { 2, 3 };
            medium_shape_ = { 10, 20 };
            large_shape_ = { 100, 200 };

            default_lr_ = 0.001f;
            default_beta1_ = 0.9f;
            default_beta2_ = 0.999f;
            default_epsilon_ = 1e-8f;
            default_weight_decay_ = 0.01f;
        }

        void TearDown() override
        {
            exec_ctx_.reset();
        }

        /**
         * @brief Create a test parameter tensor with initial values.
         */
        std::shared_ptr<Tensor<TensorDataType::FP32, CpuMemoryResource>> createParameter(
            const shape_t& shape,
            float init_value = 1.0f )
        {
            auto param = std::make_shared<Tensor<TensorDataType::FP32, CpuMemoryResource>>(
                exec_ctx_->getDevice(), shape );

            auto data = param->data();
            for (size_t i = 0; i < param->size(); ++i)
            {
                data[i] = init_value;
            }

            param->setName( "test_param" );
            return param;
        }

        /**
         * @brief Create a gradient tensor with specified values.
         */
        std::shared_ptr<Tensor<TensorDataType::FP32, CpuMemoryResource>> createGradient(
            const shape_t& shape,
            float grad_value = 0.1f )
        {
            auto grad = std::make_shared<Tensor<TensorDataType::FP32, CpuMemoryResource>>(
                exec_ctx_->getDevice(), shape );

            auto data = grad->data();
            for (size_t i = 0; i < grad->size(); ++i)
            {
                data[i] = grad_value;
            }

            grad->setName( "test_grad" );
            return grad;
        }

        /**
         * @brief Verify tensor contains no NaN or Inf values.
         */
        bool hasNaNorInf( const Tensor<TensorDataType::FP32, CpuMemoryResource>& tensor ) const
        {
            const float* data = tensor.data();
            for (size_t i = 0; i < tensor.size(); ++i)
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
        bool allClose( const Tensor<TensorDataType::FP32, CpuMemoryResource>& tensor,
            float expected,
            float tolerance = 1e-5f ) const
        {
            const float* data = tensor.data();
            for (size_t i = 0; i < tensor.size(); ++i)
            {
                if (std::abs( data[i] - expected ) > tolerance)
                {
                    return false;
                }
            }

            return true;
        }

        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_ctx_;
        shape_t small_shape_;
        shape_t medium_shape_;
        shape_t large_shape_;

        float default_lr_;
        float default_beta1_;
        float default_beta2_;
        float default_epsilon_;
        float default_weight_decay_;
    };

    // ============================================================================
    // Construction and Configuration Tests
    // ============================================================================

    TEST_F( AdamWCpuTests, Construction_WithConfig )
    {
        auto config = AdamWConfig()
            .withLearningRate( 0.001f )
            .withBeta1( 0.9f )
            .withBeta2( 0.999f )
            .withEpsilon( 1e-8f )
            .withWeightDecay( 0.01f );

        auto optimizer = std::make_shared<AdamWOptimizer<DeviceType::Cpu, TensorDataType::FP32>>(
            exec_ctx_, config );
    }

    TEST_F( AdamWCpuTests, Construction_DefaultHyperparameters )
    {
        // Construct device-agnostic optimizer using AdamWConfig defaults
        AdamWConfig config; // default values as in AdamWConfig.ixx
        auto optimizer = std::make_shared<AdamWOptimizer<DeviceType::Cpu, TensorDataType::FP32>>(
            exec_ctx_, config );

        EXPECT_FLOAT_EQ( optimizer->getLearningRate(), default_lr_ );
        EXPECT_FLOAT_EQ( optimizer->getBeta1(), default_beta1_ );
        EXPECT_FLOAT_EQ( optimizer->getBeta2(), default_beta2_ );
        EXPECT_FLOAT_EQ( optimizer->getEpsilon(), default_epsilon_ );
        EXPECT_FLOAT_EQ( optimizer->getWeightDecay(), default_weight_decay_ );
        EXPECT_EQ( optimizer->getStepCount(), 0u );
    }

    TEST_F( AdamWCpuTests, Construction_CustomHyperparameters )
    {
        float lr = 0.002f;
        float beta1 = 0.85f;
        float beta2 = 0.995f;
        float eps = 1e-7f;
        float wd = 0.02f;

        AdamWConfig cfg;
        cfg.withLearningRate( lr )
            .withBeta1( beta1 )
            .withBeta2( beta2 )
            .withEpsilon( eps )
            .withWeightDecay( wd );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        EXPECT_FLOAT_EQ( optimizer->getLearningRate(), lr );
        EXPECT_FLOAT_EQ( optimizer->getBeta1(), beta1 );
        EXPECT_FLOAT_EQ( optimizer->getBeta2(), beta2 );
        EXPECT_FLOAT_EQ( optimizer->getEpsilon(), eps );
        EXPECT_FLOAT_EQ( optimizer->getWeightDecay(), wd );
    }

    TEST_F( AdamWCpuTests, Error_NullExecutionContext )
    {
        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> null_ctx;

        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ );

        EXPECT_THROW(
            (std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
                null_ctx, cfg )),
            std::invalid_argument
        );
    }

    TEST_F( AdamWCpuTests, Error_InvalidLearningRate )
    {
        AdamWConfig cfg_zero;
        cfg_zero.withLearningRate( 0.0f );

        EXPECT_THROW(
            (std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
                exec_ctx_, cfg_zero )),  // lr = 0
            std::invalid_argument
        );

        AdamWConfig cfg_negative;
        cfg_negative.withLearningRate( -0.001f );

        EXPECT_THROW(
            (std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
                exec_ctx_, cfg_negative )),  // lr < 0
            std::invalid_argument
        );
    }

    TEST_F( AdamWCpuTests, Error_InvalidBeta1 )
    {
        AdamWConfig cfg1;
        cfg1.withLearningRate( default_lr_ ).withBeta1( 0.0f );

        EXPECT_THROW(
            (std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
                exec_ctx_, cfg1 )),  // beta1 = 0
            std::invalid_argument
        );

        AdamWConfig cfg2;
        cfg2.withLearningRate( default_lr_ ).withBeta1( 1.0f );

        EXPECT_THROW(
            (std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
                exec_ctx_, cfg2 )),  // beta1 = 1
            std::invalid_argument
        );
    }

    TEST_F( AdamWCpuTests, Error_InvalidBeta2 )
    {
        AdamWConfig cfg1;
        cfg1.withLearningRate( default_lr_ ).withBeta1( default_beta1_ ).withBeta2( 0.0f );

        EXPECT_THROW(
            (std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
                exec_ctx_, cfg1 )),  // beta2 = 0
            std::invalid_argument
        );

        AdamWConfig cfg2;
        cfg2.withLearningRate( default_lr_ ).withBeta1( default_beta1_ ).withBeta2( 1.0f );

        EXPECT_THROW(
            (std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
                exec_ctx_, cfg2 )),  // beta2 = 1
            std::invalid_argument
        );
    }

    TEST_F( AdamWCpuTests, Error_InvalidEpsilon )
    {
        AdamWConfig cfg1;
        cfg1.withLearningRate( default_lr_ ).withBeta1( default_beta1_ ).withBeta2( default_beta2_ ).withEpsilon( 0.0f );

        EXPECT_THROW(
            (std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
                exec_ctx_, cfg1 )),
            std::invalid_argument
        );

        AdamWConfig cfg2;
        cfg2.withLearningRate( default_lr_ ).withBeta1( default_beta1_ ).withBeta2( default_beta2_ ).withEpsilon( -1e-8f );

        EXPECT_THROW(
            (std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
                exec_ctx_, cfg2 )),
            std::invalid_argument
        );
    }

    TEST_F( AdamWCpuTests, Error_InvalidWeightDecay )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ )
            .withBeta1( default_beta1_ )
            .withBeta2( default_beta2_ )
            .withEpsilon( default_epsilon_ )
            .withWeightDecay( -0.01f );

        EXPECT_THROW(
            (std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
                exec_ctx_, cfg )),
            std::invalid_argument
        );
    }

    // ============================================================================
    // Parameter Registration Tests
    // ============================================================================

    TEST_F( AdamWCpuTests, AddParameter_SingleParameter )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_ );
        auto grad = createGradient( small_shape_ );

        EXPECT_NO_THROW( optimizer->addParameter( param.get(), grad.get() ) );
        EXPECT_EQ( optimizer->getParameterCount(), 1u );
    }

    TEST_F( AdamWCpuTests, AddParameter_MultipleParameters )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        for (int i = 0; i < 5; ++i)
        {
            auto param = createParameter( small_shape_ );
            auto grad = createGradient( small_shape_ );

            optimizer->addParameter( param.get(), grad.get() );
        }

        EXPECT_EQ( optimizer->getParameterCount(), 5u );
    }

    TEST_F( AdamWCpuTests, AddParameter_DifferentShapes )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param1 = createParameter( small_shape_ );
        auto grad1 = createGradient( small_shape_ );
        optimizer->addParameter( param1.get(), grad1.get() );

        auto param2 = createParameter( medium_shape_ );
        auto grad2 = createGradient( medium_shape_ );
        optimizer->addParameter( param2.get(), grad2.get() );

        EXPECT_EQ( optimizer->getParameterCount(), 2u );
    }

    TEST_F( AdamWCpuTests, Error_AddParameter_NullParameter )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto grad = createGradient( small_shape_ );

        EXPECT_THROW(
            optimizer->addParameter( nullptr, grad.get() ),
            std::invalid_argument
        );
    }

    TEST_F( AdamWCpuTests, Error_AddParameter_NullGradient )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ )
            .withName("AdamW" );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_ );

        EXPECT_THROW(
            optimizer->addParameter( param.get(), nullptr ),
            std::invalid_argument
        );
    }

    TEST_F( AdamWCpuTests, Error_AddParameter_ShapeMismatch )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ )
            .withName( "AdamW" );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_ );
        auto grad = createGradient( medium_shape_ );  // Different shape

        EXPECT_THROW(
            optimizer->addParameter( param.get(), grad.get() ),
            std::invalid_argument
        );
    }

    // ============================================================================
    // Optimization Step Tests
    // ============================================================================

    TEST_F( AdamWCpuTests, Step_SingleParameter_SingleIteration )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ )
            .withName( "AdamW" );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.1f );

        optimizer->addParameter( param.get(), grad.get() );

        // Store initial values
        std::vector<float> initial_values( param->size() );
        std::copy_n( param->data(), param->size(), initial_values.begin() );

        EXPECT_NO_THROW( optimizer->step() );
        EXPECT_EQ( optimizer->getStepCount(), 1u );

        // Parameters should have changed
        const float* updated = param->data();
        bool params_changed = false;
        for (size_t i = 0; i < param->size(); ++i)
        {
            if (std::abs( updated[i] - initial_values[i] ) > 1e-6f)
            {
                params_changed = true;
                break;
            }
        }

        EXPECT_TRUE( params_changed ) << "Parameters should be updated after step()";

        // Parameters should have decreased (gradient descent with positive gradients)
        for (size_t i = 0; i < param->size(); ++i)
        {
            EXPECT_LT( updated[i], initial_values[i] )
                << "Parameters should decrease with positive gradients";
        }

        // No NaN or Inf
        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCpuTests, Step_MultipleIterations )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ )
            .withName( "AdamW" );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.1f );

        optimizer->addParameter( param.get(), grad.get() );

        // Run multiple optimization steps
        for (int iter = 1; iter <= 10; ++iter)
        {
            EXPECT_NO_THROW( optimizer->step() );
            EXPECT_EQ( optimizer->getStepCount(), static_cast<size_t>(iter) );
            EXPECT_FALSE( hasNaNorInf( *param ) );
        }

        // Parameters should have significantly decreased
        const float* updated = param->data();
        for (size_t i = 0; i < param->size(); ++i)
        {
            EXPECT_LT( updated[i], 1.0f ) << "Parameters should decrease after multiple iterations";
        }
    }

    TEST_F( AdamWCpuTests, Step_MultipleParameters )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ )
            .withName( "AdamW" );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, CpuMemoryResource>>> params;
        std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, CpuMemoryResource>>> grads;

        for (int i = 0; i < 3; ++i)
        {
            auto param = createParameter( small_shape_, 1.0f );
            auto grad = createGradient( small_shape_, 0.1f * (i + 1) );
            optimizer->addParameter( param.get(), grad.get() );
            params.push_back( param );
            grads.push_back( grad );
        }

        EXPECT_NO_THROW( optimizer->step() );

        // All parameters should have been updated
        for (const auto& param : params)
        {
            EXPECT_FALSE( hasNaNorInf( *param ) );

            const float* data = param->data();
            bool changed = false;
            for (size_t j = 0; j < param->size(); ++j)
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

    TEST_F( AdamWCpuTests, Step_LargeParameters )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( large_shape_, 1.0f );
        auto grad = createGradient( large_shape_, 0.01f );

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );
        EXPECT_FALSE( hasNaNorInf( *param ) );
        EXPECT_EQ( optimizer->getStepCount(), 1u );
    }

    TEST_F( AdamWCpuTests, Step_WithWeightDecay )
    {
        // High weight decay to see its effect
        AdamWConfig cfg;
        cfg.withLearningRate( 0.1f )
            .withBeta1( default_beta1_ )
            .withBeta2( default_beta2_ )
            .withEpsilon( default_epsilon_ )
            .withWeightDecay( 0.1f );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.0f );  // Zero gradient

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );

        // With zero gradient and weight decay, parameters should decrease
        const float* data = param->data();
        for (size_t i = 0; i < param->size(); ++i)
        {
            EXPECT_LT( data[i], 1.0f ) << "Weight decay should reduce parameters even with zero gradient";
        }
    }

    TEST_F( AdamWCpuTests, Error_StepWithoutParameters )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ )
            .withName( "AdamW" );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        EXPECT_THROW( optimizer->step(), std::runtime_error );
    }

    // ============================================================================
    // Gradient Zeroing Tests
    // ============================================================================

    TEST_F( AdamWCpuTests, ZeroGrad_SingleParameter )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ )
            .withName( "AdamW" );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_ );
        auto grad = createGradient( small_shape_, 0.5f );

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->zeroGrad() );

        EXPECT_TRUE( allClose( *grad, 0.0f ) ) << "Gradients should be zero after zeroGrad()";
    }

    TEST_F( AdamWCpuTests, ZeroGrad_MultipleParameters )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        std::vector<std::shared_ptr<Tensor<TensorDataType::FP32, CpuMemoryResource>>> grads;

        for (int i = 0; i < 5; ++i)
        {
            auto param = createParameter( small_shape_ );
            auto grad = createGradient( small_shape_, 0.5f );
            optimizer->addParameter( param.get(), grad.get() );
            grads.push_back( grad );
        }

        EXPECT_NO_THROW( optimizer->zeroGrad() );

        for (const auto& grad : grads)
        {
            EXPECT_TRUE( allClose( *grad, 0.0f ) ) << "All gradients should be zero";
        }
    }

    TEST_F( AdamWCpuTests, Error_ZeroGradWithoutParameters )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        EXPECT_THROW( optimizer->zeroGrad(), std::runtime_error );
    }

    // ============================================================================
    // Hyperparameter Update Tests
    // ============================================================================

    TEST_F( AdamWCpuTests, SetLearningRate )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        float new_lr = 0.002f;
        EXPECT_NO_THROW( optimizer->setLearningRate( new_lr ) );
        EXPECT_FLOAT_EQ( optimizer->getLearningRate(), new_lr );
    }

    TEST_F( AdamWCpuTests, Error_SetLearningRate_Invalid )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ )
            .withName( "AdamW" );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        EXPECT_THROW( optimizer->setLearningRate( 0.0f ), std::invalid_argument );
        EXPECT_THROW( optimizer->setLearningRate( -0.001f ), std::invalid_argument );
    }

    TEST_F( AdamWCpuTests, LearningRateSchedule )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( 0.1f )
            .withName( "AdamW" );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

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
            EXPECT_FALSE( hasNaNorInf( *param ) );
        }
    }

    // ============================================================================
    // Edge Cases and Numerical Stability Tests
    // ============================================================================

    TEST_F( AdamWCpuTests, EdgeCase_ZeroGradients )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ )
            .withBeta1( default_beta1_ )
            .withBeta2( default_beta2_ )
            .withEpsilon( default_epsilon_ )
            .withWeightDecay( 0.0f )
            .withName("AdamW" );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.0f );

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );

        // With zero gradients and zero weight decay, parameters should not change
        EXPECT_TRUE( allClose( *param, 1.0f, 1e-6f ) );
        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCpuTests, EdgeCase_LargeGradients )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ )
            .withName( "AdamW" );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 100.0f );  // Very large gradient

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );
        EXPECT_FALSE( hasNaNorInf( *param ) ) << "AdamW should handle large gradients without NaN/Inf";
    }

    TEST_F( AdamWCpuTests, EdgeCase_VerySmallGradients )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 1e-10f );  // Very small gradient

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );
        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCpuTests, EdgeCase_MixedSignGradients )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( default_lr_ );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_, 0.0f );
        auto grad = createGradient( small_shape_, 0.0f );

        // Set mixed sign gradients
        auto grad_data = grad->data();
        for (size_t i = 0; i < grad->size(); ++i)
        {
            grad_data[i] = (i % 2 == 0) ? 0.1f : -0.1f;
        }

        optimizer->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( optimizer->step() );
        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCpuTests, NumericalStability_ManyIterations )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( 0.01f );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.01f );

        optimizer->addParameter( param.get(), grad.get() );

        // Run many iterations to test numerical stability
        for (int iter = 0; iter < 1000; ++iter)
        {
            optimizer->step();

            if (iter % 100 == 0)
            {
                EXPECT_FALSE( hasNaNorInf( *param ) )
                    << "Numerical instability detected at iteration " << iter;
            }
        }

        EXPECT_FALSE( hasNaNorInf( *param ) );
        EXPECT_EQ( optimizer->getStepCount(), 1000u );
    }

    // ============================================================================
    // Bias Correction Tests
    // ============================================================================

    TEST_F( AdamWCpuTests, BiasCorrection_FirstSteps )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( 0.01f )
            .withBeta1( 0.9f )
            .withBeta2( 0.999f );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

        auto param = createParameter( small_shape_, 1.0f );
        auto grad = createGradient( small_shape_, 0.1f );

        optimizer->addParameter( param.get(), grad.get() );

        // First few steps should show bias correction effect
        std::vector<float> param_values_after_step;

        for (int step = 0; step < 5; ++step)
        {
            optimizer->step();
            param_values_after_step.push_back( param->data()[0] );
        }

        // Each step should produce different updates due to bias correction
        for (size_t i = 1; i < param_values_after_step.size(); ++i)
        {
            EXPECT_NE( param_values_after_step[i], param_values_after_step[i - 1] )
                << "Parameters should change at each step";
        }
    }

    // ============================================================================
    // Integration Tests
    // ============================================================================

    TEST_F( AdamWCpuTests, Integration_TrainingLoop )
    {
        AdamWConfig cfg;
        cfg.withLearningRate( 0.01f );

        auto optimizer = std::make_shared<CpuAdamWOptimizer<TensorDataType::FP32>>(
            exec_ctx_, cfg );

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
            auto grad1_data = grad1->data();
            auto grad2_data = grad2->data();

            for (size_t i = 0; i < grad1->size(); ++i)
            {
                grad1_data[i] = 0.1f * (1.0f - 0.1f * epoch);
            }
            for (size_t i = 0; i < grad2->size(); ++i)
            {
                grad2_data[i] = 0.05f * (1.0f - 0.1f * epoch);
            }

            // Optimization step
            optimizer->step();

            // Zero gradients
            optimizer->zeroGrad();

            // Verify state
            EXPECT_FALSE( hasNaNorInf( *param1 ) );
            EXPECT_FALSE( hasNaNorInf( *param2 ) );
            EXPECT_TRUE( allClose( *grad1, 0.0f ) );
            EXPECT_TRUE( allClose( *grad2, 0.0f ) );
        }

        EXPECT_EQ( optimizer->getStepCount(), 10u );
    }
}