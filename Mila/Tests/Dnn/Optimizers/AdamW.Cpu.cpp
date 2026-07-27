/**
 * @file AdamW.Cpu.cpp
 * @brief Tests for AdamWOptimizer<DeviceType::Cpu, FP32> against the current API.
 *
 * AdamW is part of the MNIST/Bard training spine. These tests exercise the
 * public device-agnostic wrapper (the same surface Network::createOptimizer
 * hands a model) rather than the internal CpuAdamWOptimizer:
 *   AdamWOptimizer( IExecutionContext*, const AdamWConfig& )
 *   addParameter( ITensor*, ITensor* ) ; step() ; get/setLearningRate ;
 *   getStepCount / getBeta1 / getBeta2 / getEpsilon / getWeightDecay /
 *   setWeightDecay / getParameterCount.
 *
 * Revival notes (translated from the pre-methodology file):
 * - Namespace moved to Mila::Tests::Dnn::Optimizers (path-mirroring rule).
 * - Execution context is created via createExecutionContext( Device::Cpu() )
 *   (a unique_ptr<IExecutionContext>) and passed as a raw pointer, matching the
 *   wrapper ctor and Network::createOptimizer.
 * - zeroGrad() is GONE from the Optimizer interface: gradient zeroing is now
 *   owned by the model (Network::zeroGradients()), so the old ZeroGrad_* tests
 *   are dropped here and belong to the Network test instead.
 * - AdamWConfig has no withName(); the old .withName("AdamW") calls are dropped.
 *
 * Net-new vs the old suite: a closed-loop convergence test (section "Convergence")
 * that recomputes the gradient of a known objective each step and asserts the
 * parameter converges to the minimizer -- proving the update direction and bias
 * correction are correct, not merely that step() runs.
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

import Mila;

namespace Mila::Tests::Dnn::Optimizers
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Optimizers;

    namespace
    {
        using AdamWCpu = AdamWOptimizer<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // Default hyperparameters as declared in AdamWConfig.ixx; the construction
        // test pins these so a default drift is caught here.
        constexpr float kDefaultLearningRate = 0.001f;
        constexpr float kDefaultBeta1 = 0.9f;
        constexpr float kDefaultBeta2 = 0.999f;
        constexpr float kDefaultEpsilon = 1e-8f;
        constexpr float kDefaultWeightDecay = 0.01f;
    }

    class AdamWCpuTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            exec_context_ = createExecutionContext( Device::Cpu() );

            small_shape_ = { 2, 3 };
            medium_shape_ = { 10, 20 };
            large_shape_ = { 100, 200 };
        }

        // A config carrying only a valid learning rate; other fields keep defaults.
        static AdamWConfig config( float learning_rate = kDefaultLearningRate )
        {
            return AdamWConfig().withLearningRate( learning_rate );
        }

        std::shared_ptr<AdamWCpu> optimizer( const AdamWConfig& cfg )
        {
            return std::make_shared<AdamWCpu>( exec_context_.get(), cfg );
        }

        std::shared_ptr<TensorFp32> makeParameter( const shape_t& shape, float init_value = 1.0f )
        {
            auto param = std::make_shared<TensorFp32>( Device::Cpu(), shape );

            float* data = param->data();
            for ( size_t i = 0; i < param->size(); ++i )
            {
                data[ i ] = init_value;
            }

            param->setName( "test_param" );

            return param;
        }

        std::shared_ptr<TensorFp32> makeGradient( const shape_t& shape, float grad_value = 0.1f )
        {
            auto grad = std::make_shared<TensorFp32>( Device::Cpu(), shape );

            float* data = grad->data();
            for ( size_t i = 0; i < grad->size(); ++i )
            {
                data[ i ] = grad_value;
            }

            grad->setName( "test_grad" );

            return grad;
        }

        static bool hasNaNorInf( const TensorFp32& tensor )
        {
            const float* data = tensor.data();
            for ( dim_t i = 0; i < tensor.size(); ++i )
            {
                if ( std::isnan( data[ i ] ) || std::isinf( data[ i ] ) )
                {
                    return true;
                }
            }

            return false;
        }

        static bool allClose( const TensorFp32& tensor, float expected, float tolerance = 1e-5f )
        {
            const float* data = tensor.data();
            for ( dim_t i = 0; i < tensor.size(); ++i )
            {
                if ( std::abs( data[ i ] - expected ) > tolerance )
                {
                    return false;
                }
            }

            return true;
        }

        std::unique_ptr<IExecutionContext> exec_context_;
        shape_t small_shape_;
        shape_t medium_shape_;
        shape_t large_shape_;
    };

    // ====================================================================
    // A. Construction & Configuration
    // ====================================================================

    TEST_F( AdamWCpuTests, Construct_WithConfigSucceeds )
    {
        EXPECT_NO_THROW( optimizer( config() ) );
    }

    TEST_F( AdamWCpuTests, Construct_DefaultHyperparameters )
    {
        auto opt = optimizer( AdamWConfig() );

        EXPECT_FLOAT_EQ( opt->getLearningRate(), kDefaultLearningRate );
        EXPECT_FLOAT_EQ( opt->getBeta1(), kDefaultBeta1 );
        EXPECT_FLOAT_EQ( opt->getBeta2(), kDefaultBeta2 );
        EXPECT_FLOAT_EQ( opt->getEpsilon(), kDefaultEpsilon );
        EXPECT_FLOAT_EQ( opt->getWeightDecay(), kDefaultWeightDecay );
        EXPECT_EQ( opt->getStepCount(), 0u );
    }

    TEST_F( AdamWCpuTests, Construct_CustomHyperparameters )
    {
        auto cfg = AdamWConfig()
            .withLearningRate( 0.002f )
            .withBeta1( 0.85f )
            .withBeta2( 0.995f )
            .withEpsilon( 1e-7f )
            .withWeightDecay( 0.02f );

        auto opt = optimizer( cfg );

        EXPECT_FLOAT_EQ( opt->getLearningRate(), 0.002f );
        EXPECT_FLOAT_EQ( opt->getBeta1(), 0.85f );
        EXPECT_FLOAT_EQ( opt->getBeta2(), 0.995f );
        EXPECT_FLOAT_EQ( opt->getEpsilon(), 1e-7f );
        EXPECT_FLOAT_EQ( opt->getWeightDecay(), 0.02f );
    }

    // ====================================================================
    // A. Validation (each documented @throws is one negative test)
    // ====================================================================

    TEST_F( AdamWCpuTests, Construct_ThrowsOnNullExecutionContext )
    {
        EXPECT_THROW( ( AdamWCpu( nullptr, config() ) ), std::invalid_argument );
    }

    TEST_F( AdamWCpuTests, Construct_ThrowsOnNonPositiveLearningRate )
    {
        EXPECT_THROW( optimizer( AdamWConfig().withLearningRate( 0.0f ) ), std::invalid_argument );
        EXPECT_THROW( optimizer( AdamWConfig().withLearningRate( -0.001f ) ), std::invalid_argument );
    }

    TEST_F( AdamWCpuTests, Construct_ThrowsOnBeta1OutOfRange )
    {
        EXPECT_THROW( optimizer( config().withBeta1( 0.0f ) ), std::invalid_argument );
        EXPECT_THROW( optimizer( config().withBeta1( 1.0f ) ), std::invalid_argument );
    }

    TEST_F( AdamWCpuTests, Construct_ThrowsOnBeta2OutOfRange )
    {
        EXPECT_THROW( optimizer( config().withBeta2( 0.0f ) ), std::invalid_argument );
        EXPECT_THROW( optimizer( config().withBeta2( 1.0f ) ), std::invalid_argument );
    }

    TEST_F( AdamWCpuTests, Construct_ThrowsOnNonPositiveEpsilon )
    {
        EXPECT_THROW( optimizer( config().withEpsilon( 0.0f ) ), std::invalid_argument );
        EXPECT_THROW( optimizer( config().withEpsilon( -1e-8f ) ), std::invalid_argument );
    }

    TEST_F( AdamWCpuTests, Construct_ThrowsOnNegativeWeightDecay )
    {
        EXPECT_THROW( optimizer( config().withWeightDecay( -0.01f ) ), std::invalid_argument );
    }

    // ====================================================================
    // G. Parameter Registration
    // ====================================================================

    TEST_F( AdamWCpuTests, AddParameter_SingleParameter )
    {
        auto opt = optimizer( config() );
        auto param = makeParameter( small_shape_ );
        auto grad = makeGradient( small_shape_ );

        EXPECT_NO_THROW( opt->addParameter( param.get(), grad.get() ) );
        EXPECT_EQ( opt->getParameterCount(), 1u );
    }

    TEST_F( AdamWCpuTests, AddParameter_MultipleParameters )
    {
        auto opt = optimizer( config() );

        std::vector<std::shared_ptr<TensorFp32>> params;
        std::vector<std::shared_ptr<TensorFp32>> grads;

        for ( int i = 0; i < 5; ++i )
        {
            auto param = makeParameter( small_shape_ );
            auto grad = makeGradient( small_shape_ );
            opt->addParameter( param.get(), grad.get() );
            params.push_back( param );
            grads.push_back( grad );
        }

        EXPECT_EQ( opt->getParameterCount(), 5u );
    }

    TEST_F( AdamWCpuTests, AddParameter_DifferentShapes )
    {
        auto opt = optimizer( config() );

        auto param1 = makeParameter( small_shape_ );
        auto grad1 = makeGradient( small_shape_ );
        opt->addParameter( param1.get(), grad1.get() );

        auto param2 = makeParameter( medium_shape_ );
        auto grad2 = makeGradient( medium_shape_ );
        opt->addParameter( param2.get(), grad2.get() );

        EXPECT_EQ( opt->getParameterCount(), 2u );
    }

    TEST_F( AdamWCpuTests, AddParameter_ThrowsOnNullParameter )
    {
        auto opt = optimizer( config() );
        auto grad = makeGradient( small_shape_ );

        EXPECT_THROW( opt->addParameter( nullptr, grad.get() ), std::invalid_argument );
    }

    TEST_F( AdamWCpuTests, AddParameter_ThrowsOnNullGradient )
    {
        auto opt = optimizer( config() );
        auto param = makeParameter( small_shape_ );

        EXPECT_THROW( opt->addParameter( param.get(), nullptr ), std::invalid_argument );
    }

    TEST_F( AdamWCpuTests, AddParameter_ThrowsOnShapeMismatch )
    {
        auto opt = optimizer( config() );
        auto param = makeParameter( small_shape_ );
        auto grad = makeGradient( medium_shape_ );

        EXPECT_THROW( opt->addParameter( param.get(), grad.get() ), std::invalid_argument );
    }

    // ====================================================================
    // E. Optimization Step
    // ====================================================================

    TEST_F( AdamWCpuTests, Step_ThrowsWithoutParameters )
    {
        auto opt = optimizer( config() );

        EXPECT_THROW( opt->step(), std::runtime_error );
    }

    TEST_F( AdamWCpuTests, Step_SingleParameterDecreasesWithPositiveGradient )
    {
        auto opt = optimizer( config() );
        auto param = makeParameter( small_shape_, 1.0f );
        auto grad = makeGradient( small_shape_, 0.1f );

        opt->addParameter( param.get(), grad.get() );

        std::vector<float> initial( param->size() );
        std::copy_n( param->data(), param->size(), initial.begin() );

        EXPECT_NO_THROW( opt->step() );
        EXPECT_EQ( opt->getStepCount(), 1u );

        const float* updated = param->data();
        for ( size_t i = 0; i < param->size(); ++i )
        {
            EXPECT_LT( updated[ i ], initial[ i ] )
                << "positive gradient should decrease the parameter at index " << i;
        }

        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCpuTests, Step_MultipleIterationsAdvanceStepCount )
    {
        auto opt = optimizer( config() );
        auto param = makeParameter( small_shape_, 1.0f );
        auto grad = makeGradient( small_shape_, 0.1f );

        opt->addParameter( param.get(), grad.get() );

        for ( int iter = 1; iter <= 10; ++iter )
        {
            EXPECT_NO_THROW( opt->step() );
            EXPECT_EQ( opt->getStepCount(), static_cast<size_t>( iter ) );
            EXPECT_FALSE( hasNaNorInf( *param ) );
        }

        const float* updated = param->data();
        for ( size_t i = 0; i < param->size(); ++i )
        {
            EXPECT_LT( updated[ i ], 1.0f ) << "parameter should keep decreasing across iterations";
        }
    }

    TEST_F( AdamWCpuTests, Step_MultipleParametersAllUpdated )
    {
        auto opt = optimizer( config() );

        std::vector<std::shared_ptr<TensorFp32>> params;

        for ( int i = 0; i < 3; ++i )
        {
            auto param = makeParameter( small_shape_, 1.0f );
            auto grad = makeGradient( small_shape_, 0.1f * static_cast<float>( i + 1 ) );
            opt->addParameter( param.get(), grad.get() );
            params.push_back( param );
        }

        EXPECT_NO_THROW( opt->step() );

        for ( const auto& param : params )
        {
            EXPECT_FALSE( hasNaNorInf( *param ) );
            EXPECT_FALSE( allClose( *param, 1.0f, 1e-6f ) ) << "every registered parameter should change";
        }
    }

    TEST_F( AdamWCpuTests, Step_LargeParameterStaysFinite )
    {
        auto opt = optimizer( config() );
        auto param = makeParameter( large_shape_, 1.0f );
        auto grad = makeGradient( large_shape_, 0.01f );

        opt->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( opt->step() );
        EXPECT_FALSE( hasNaNorInf( *param ) );
        EXPECT_EQ( opt->getStepCount(), 1u );
    }

    TEST_F( AdamWCpuTests, Step_WeightDecayShrinksParameterWithZeroGradient )
    {
        auto opt = optimizer( config( 0.1f ).withWeightDecay( 0.1f ) );
        auto param = makeParameter( small_shape_, 1.0f );
        auto grad = makeGradient( small_shape_, 0.0f );

        opt->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( opt->step() );

        const float* data = param->data();
        for ( size_t i = 0; i < param->size(); ++i )
        {
            EXPECT_LT( data[ i ], 1.0f ) << "decoupled weight decay should shrink the parameter even at zero gradient";
        }
    }

    // ====================================================================
    // D. Hyperparameter updates
    // ====================================================================

    TEST_F( AdamWCpuTests, SetLearningRate_Updates )
    {
        auto opt = optimizer( config() );

        EXPECT_NO_THROW( opt->setLearningRate( 0.002f ) );
        EXPECT_FLOAT_EQ( opt->getLearningRate(), 0.002f );
    }

    TEST_F( AdamWCpuTests, SetLearningRate_ThrowsOnNonPositive )
    {
        auto opt = optimizer( config() );

        EXPECT_THROW( opt->setLearningRate( 0.0f ), std::invalid_argument );
        EXPECT_THROW( opt->setLearningRate( -0.001f ), std::invalid_argument );
    }

    TEST_F( AdamWCpuTests, SetLearningRate_ScheduleAppliesAcrossSteps )
    {
        auto opt = optimizer( config( 0.1f ) );
        auto param = makeParameter( small_shape_, 1.0f );
        auto grad = makeGradient( small_shape_, 0.1f );
        opt->addParameter( param.get(), grad.get() );

        for ( int epoch = 0; epoch < 5; ++epoch )
        {
            const float lr = 0.1f * std::pow( 0.9f, static_cast<float>( epoch ) );
            opt->setLearningRate( lr );
            EXPECT_FLOAT_EQ( opt->getLearningRate(), lr );

            opt->step();
            EXPECT_FALSE( hasNaNorInf( *param ) );
        }
    }

    // ====================================================================
    // E. Numerical stability / edge cases
    // ====================================================================

    TEST_F( AdamWCpuTests, EdgeCase_ZeroGradientZeroDecayLeavesParameterUnchanged )
    {
        auto opt = optimizer( config().withWeightDecay( 0.0f ) );
        auto param = makeParameter( small_shape_, 1.0f );
        auto grad = makeGradient( small_shape_, 0.0f );

        opt->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( opt->step() );

        EXPECT_TRUE( allClose( *param, 1.0f, 1e-6f ) );
        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCpuTests, EdgeCase_LargeGradientStaysFinite )
    {
        auto opt = optimizer( config() );
        auto param = makeParameter( small_shape_, 1.0f );
        auto grad = makeGradient( small_shape_, 100.0f );

        opt->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( opt->step() );
        EXPECT_FALSE( hasNaNorInf( *param ) ) << "AdamW normalization should tame large gradients";
    }

    TEST_F( AdamWCpuTests, EdgeCase_VerySmallGradientStaysFinite )
    {
        auto opt = optimizer( config() );
        auto param = makeParameter( small_shape_, 1.0f );
        auto grad = makeGradient( small_shape_, 1e-10f );

        opt->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( opt->step() );
        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCpuTests, EdgeCase_MixedSignGradientStaysFinite )
    {
        auto opt = optimizer( config() );
        auto param = makeParameter( small_shape_, 0.0f );
        auto grad = makeGradient( small_shape_, 0.0f );

        float* grad_data = grad->data();
        for ( size_t i = 0; i < grad->size(); ++i )
        {
            grad_data[ i ] = ( i % 2 == 0 ) ? 0.1f : -0.1f;
        }

        opt->addParameter( param.get(), grad.get() );

        EXPECT_NO_THROW( opt->step() );
        EXPECT_FALSE( hasNaNorInf( *param ) );
    }

    TEST_F( AdamWCpuTests, NumericalStability_ManyIterations )
    {
        auto opt = optimizer( config( 0.01f ) );
        auto param = makeParameter( small_shape_, 1.0f );
        auto grad = makeGradient( small_shape_, 0.01f );

        opt->addParameter( param.get(), grad.get() );

        for ( int iter = 0; iter < 1000; ++iter )
        {
            opt->step();

            if ( iter % 100 == 0 )
            {
                EXPECT_FALSE( hasNaNorInf( *param ) )
                    << "numerical instability detected at iteration " << iter;
            }
        }

        EXPECT_FALSE( hasNaNorInf( *param ) );
        EXPECT_EQ( opt->getStepCount(), 1000u );
    }

    TEST_F( AdamWCpuTests, BiasCorrection_EarlyStepsDiffer )
    {
        auto opt = optimizer( config( 0.01f ) );
        auto param = makeParameter( small_shape_, 1.0f );
        auto grad = makeGradient( small_shape_, 0.1f );

        opt->addParameter( param.get(), grad.get() );

        std::vector<float> after_step;
        for ( int step = 0; step < 5; ++step )
        {
            opt->step();
            after_step.push_back( param->data()[ 0 ] );
        }

        for ( size_t i = 1; i < after_step.size(); ++i )
        {
            EXPECT_NE( after_step[ i ], after_step[ i - 1 ] )
                << "bias correction should produce a distinct update at each early step";
        }
    }

    // ====================================================================
    // Convergence (net-new closed-loop test)
    // ====================================================================
    //
    // The old suite drove step() with FIXED gradients (open loop) and only
    // asserted "parameters moved / stayed finite". This recomputes the gradient
    // of a known convex objective each step and asserts the parameter converges
    // to the minimizer -- a direct check that the AdamW update direction and
    // bias correction are correct, not merely that step() runs. Weight decay is
    // disabled so the unique fixed point is the target (decoupled decay would
    // otherwise pull the optimum toward zero).

    TEST_F( AdamWCpuTests, Convergence_MinimizesQuadraticToTarget )
    {
        // L(x) = sum_i (x_i - t_i)^2 ; dL/dx_i = 2 (x_i - t_i).
        const shape_t shape{ 4 };
        const std::vector<float> target{ 0.5f, -1.0f, 2.0f, -0.25f };

        auto opt = optimizer( config( 0.05f ).withWeightDecay( 0.0f ) );
        auto param = makeParameter( shape, 0.0f );
        auto grad = makeGradient( shape, 0.0f );

        opt->addParameter( param.get(), grad.get() );

        auto loss = [&]() {
            const float* x = param->data();
            float sum = 0.0f;
            for ( size_t i = 0; i < shape[ 0 ]; ++i )
            {
                const float d = x[ i ] - target[ i ];
                sum += d * d;
            }

            return sum;
        };

        const float initial_loss = loss();

        for ( int iter = 0; iter < 600; ++iter )
        {
            const float* x = param->data();
            float* g = grad->data();
            for ( size_t i = 0; i < shape[ 0 ]; ++i )
            {
                g[ i ] = 2.0f * ( x[ i ] - target[ i ] );
            }

            opt->step();
        }

        const float final_loss = loss();

        EXPECT_LT( final_loss, initial_loss ) << "loss must strictly decrease";
        EXPECT_FALSE( hasNaNorInf( *param ) );

        const float* x = param->data();
        for ( size_t i = 0; i < shape[ 0 ]; ++i )
        {
            EXPECT_NEAR( x[ i ], target[ i ], 1e-2f )
                << "parameter " << i << " should converge to its target";
        }
    }

    // ====================================================================
    // Integration (training-loop shape: forward gradient -> step, repeat)
    // ====================================================================

    TEST_F( AdamWCpuTests, Integration_TrainingLoopRemainsStable )
    {
        auto opt = optimizer( config( 0.01f ) );

        auto param1 = makeParameter( small_shape_, 1.0f );
        auto grad1 = makeGradient( small_shape_, 0.1f );
        opt->addParameter( param1.get(), grad1.get() );

        auto param2 = makeParameter( medium_shape_, 0.5f );
        auto grad2 = makeGradient( medium_shape_, 0.05f );
        opt->addParameter( param2.get(), grad2.get() );

        for ( int epoch = 0; epoch < 10; ++epoch )
        {
            float* g1 = grad1->data();
            float* g2 = grad2->data();
            for ( size_t i = 0; i < grad1->size(); ++i )
            {
                g1[ i ] = 0.1f * ( 1.0f - 0.1f * static_cast<float>( epoch ) );
            }
            for ( size_t i = 0; i < grad2->size(); ++i )
            {
                g2[ i ] = 0.05f * ( 1.0f - 0.1f * static_cast<float>( epoch ) );
            }

            opt->step();

            EXPECT_FALSE( hasNaNorInf( *param1 ) );
            EXPECT_FALSE( hasNaNorInf( *param2 ) );
        }

        EXPECT_EQ( opt->getStepCount(), 10u );
    }
}
