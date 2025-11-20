#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    template<TensorDataType TPrecision>
    struct LinearCpuTestData
    {
        shape_t input_shape;
        shape_t output_shape;
        LinearConfig config;
        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_context;
        std::shared_ptr<Linear<DeviceType::Cpu, TPrecision>> module;
        int64_t input_features;
        int64_t output_features;
        bool has_bias;

        LinearCpuTestData() : config( 1, 1 ), input_features( 0 ), output_features( 0 ), has_bias( true )
        {
        }

        static LinearCpuTestData Create(
            const std::string& name,
            const shape_t& input_shape,
            int64_t input_features,
            int64_t output_features,
            bool has_bias = true )
        {
            LinearCpuTestData data;
            data.input_shape = input_shape;
            data.input_features = input_features;
            data.output_features = output_features;
            data.has_bias = has_bias;

            data.output_shape = input_shape;
            data.output_shape.back() = output_features;

            data.config = LinearConfig( input_features, output_features );
            data.config.withName( name )
                .withBias( has_bias );

            data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            data.module = std::make_shared<Linear<DeviceType::Cpu, TPrecision>>( data.exec_context, data.config );

            return data;
        }

        static LinearCpuTestData CreateWithContext(
            const std::string& name,
            const shape_t& input_shape,
            int64_t input_features,
            int64_t output_features,
            std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
            bool has_bias = true )
        {
            LinearCpuTestData data;
            data.input_shape = input_shape;
            data.input_features = input_features;
            data.output_features = output_features;
            data.has_bias = has_bias;

            data.output_shape = input_shape;
            data.output_shape.back() = output_features;

            data.config = LinearConfig( input_features, output_features );
            data.config.withName( name )
                .withBias( has_bias );

            data.exec_context = context;
            data.module = std::make_shared<Linear<DeviceType::Cpu, TPrecision>>( data.exec_context, data.config );

            return data;
        }
    };

    class LinearCpuTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            small_shape_ = { 2, 3, 16 };
            medium_shape_ = { 4, 128, 512 };
            large_shape_ = { 8, 256, 1024 };

            input_features_ = 16;
            output_features_ = 32;
        }

        LinearCpuTestData<TensorDataType::FP32>& SmallFp32Data()
        {
            if (!small_fp32_.module)
            {
                small_fp32_ = LinearCpuTestData<TensorDataType::FP32>::Create(
                    "small_linear_cpu", small_shape_, input_features_, output_features_ );
            }
            return small_fp32_;
        }

        LinearCpuTestData<TensorDataType::FP32>& MediumFp32Data()
        {
            if (!medium_fp32_.module)
            {
                medium_fp32_ = LinearCpuTestData<TensorDataType::FP32>::Create(
                    "medium_linear_cpu", medium_shape_, 512, 256 );
            }
            return medium_fp32_;
        }

        LinearCpuTestData<TensorDataType::FP32>& LargeFp32Data()
        {
            if (!large_fp32_.module)
            {
                large_fp32_ = LinearCpuTestData<TensorDataType::FP32>::Create(
                    "large_linear_cpu", large_shape_, 1024, 768 );
            }
            return large_fp32_;
        }

        LinearCpuTestData<TensorDataType::FP32>& NoBiasFp32Data()
        {
            if (!no_bias_fp32_.module)
            {
                no_bias_fp32_ = LinearCpuTestData<TensorDataType::FP32>::Create(
                    "no_bias_linear_cpu", small_shape_, input_features_, output_features_, false );
            }
            return no_bias_fp32_;
        }

        shape_t small_shape_;
        shape_t medium_shape_;
        shape_t large_shape_;
        int64_t input_features_;
        int64_t output_features_;

        LinearCpuTestData<TensorDataType::FP32> small_fp32_;
        LinearCpuTestData<TensorDataType::FP32> medium_fp32_;
        LinearCpuTestData<TensorDataType::FP32> large_fp32_;
        LinearCpuTestData<TensorDataType::FP32> no_bias_fp32_;
    };

    template<TensorDataType TPrecision>
    void TestGetName( const LinearCpuTestData<TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.module->getName(), expected_name );
    }

    template<TensorDataType TPrecision>
    void TestDeviceType( const LinearCpuTestData<TPrecision>& data )
    {
        EXPECT_EQ( data.module->getDeviceType(), DeviceType::Cpu );
        ASSERT_NE( data.exec_context, nullptr );

        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
    }

    template<TensorDataType TPrecision>
    void TestIsBuilt( const LinearCpuTestData<TPrecision>& data, bool expected_built )
    {
        EXPECT_EQ( data.module->isBuilt(), expected_built );
    }

    template<TensorDataType TPrecision>
    void TestBuild( LinearCpuTestData<TPrecision>& data )
    {
        EXPECT_NO_THROW( data.module->build( data.input_shape ) );
        EXPECT_TRUE( data.module->isBuilt() );

        data.module->build( data.input_shape );
        EXPECT_TRUE( data.module->isBuilt() );
    }

    template<TensorDataType TPrecision>
    void TestParameterCount( const LinearCpuTestData<TPrecision>& data )
    {
        size_t expected_count = data.input_features * data.output_features;
        if (data.has_bias)
        {
            expected_count += data.output_features;
        }
        EXPECT_EQ( data.module->parameterCount(), expected_count );
    }

    template<TensorDataType TPrecision>
    void TestGetWeight( const LinearCpuTestData<TPrecision>& data )
    {
        auto weight = data.module->getWeight();
        ASSERT_NE( weight, nullptr );
        EXPECT_EQ( weight->shape()[0], data.output_features );
        EXPECT_EQ( weight->shape()[1], data.input_features );
    }

    template<TensorDataType TPrecision>
    void TestGetBias( const LinearCpuTestData<TPrecision>& data )
    {
        auto bias = data.module->getBias();

        if (data.has_bias)
        {
            ASSERT_NE( bias, nullptr );
            EXPECT_EQ( bias->shape()[0], data.output_features );
        }
        else
        {
            EXPECT_EQ( bias, nullptr );
        }
    }

    template<TensorDataType TPrecision>
    void TestHasBias( const LinearCpuTestData<TPrecision>& data )
    {
        EXPECT_EQ( data.module->hasBias(), data.has_bias );
    }

    template<TensorDataType TPrecision>
    void TestToString( const LinearCpuTestData<TPrecision>& data )
    {
        std::string output = data.module->toString();

        EXPECT_NE( output.find( "Linear" ), std::string::npos );
        EXPECT_NE( output.find( data.config.getName() ), std::string::npos );
        EXPECT_NE( output.find( "Input features:" ), std::string::npos );
        EXPECT_NE( output.find( "Output features:" ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
    }

    template<TensorDataType TPrecision>
    void TestForward( LinearCpuTestData<TPrecision>& data )
    {
        using TensorType = CpuTensor<TPrecision>;

        data.module->build( data.input_shape );

        TensorType input( "CPU", data.input_shape );
        TensorType output( "CPU", data.output_shape );

        random( input, -1.0f, 1.0f );

        EXPECT_NO_THROW( data.module->forward( input, output ) );
        EXPECT_EQ( output.size(),
            data.output_shape[0] * data.output_shape[1] * data.output_shape[2] );
        EXPECT_EQ( output.shape(), data.output_shape );
    }

    template<TensorDataType TPrecision>
    void TestGetParameters( const LinearCpuTestData<TPrecision>& data )
    {
        auto params = data.module->getParameters();

        if (data.has_bias)
        {
            EXPECT_EQ( params.size(), 2 );
            EXPECT_NE( params[0], nullptr );
            EXPECT_NE( params[1], nullptr );
        }
        else
        {
            EXPECT_EQ( params.size(), 1 );
            EXPECT_NE( params[0], nullptr );
        }
    }

    template<TensorDataType TPrecision>
    void TestGetWeightGrad( LinearCpuTestData<TPrecision>& data )
    {
        data.module->setTraining( true );
        data.module->build( data.input_shape );

        auto weight_grad = data.module->getWeightGrad();

        ASSERT_NE( weight_grad, nullptr ) << "Weight gradients should be allocated in training mode";
        EXPECT_EQ( weight_grad->shape()[0], data.output_features );
        EXPECT_EQ( weight_grad->shape()[1], data.input_features );
    }

    template<TensorDataType TPrecision>
    void TestGetBiasGrad( LinearCpuTestData<TPrecision>& data )
    {
        data.module->setTraining( true );
        data.module->build( data.input_shape );

        auto bias_grad = data.module->getBiasGrad();

        if (data.has_bias)
        {
            ASSERT_NE( bias_grad, nullptr ) << "Bias gradients should be allocated in training mode";
            EXPECT_EQ( bias_grad->shape()[0], data.output_features );
        }
        else
        {
            EXPECT_EQ( bias_grad, nullptr ) << "No bias gradient when bias is disabled";
        }
    }

    template<TensorDataType TPrecision>
    void TestBackward( LinearCpuTestData<TPrecision>& data )
    {
        using TensorType = CpuTensor<TPrecision>;

        data.module->setTraining( true );
        data.module->build( data.input_shape );

        TensorType input( "CPU", data.input_shape );
        TensorType output( "CPU", data.output_shape );
        TensorType output_grad( "CPU", data.output_shape );
        TensorType input_grad( "CPU", data.input_shape );

        random( input, -1.0f, 1.0f );
        random( output_grad, -0.1f, 0.1f );
        zeros( input_grad );

        data.module->forward( input, output );

        EXPECT_NO_THROW(
            data.module->backward( input, output_grad, input_grad )
        ) << "Backward pass should succeed for CPU Linear operation in training mode";

        EXPECT_EQ( input_grad.shape(), data.input_shape );

        bool has_nonzero_grad = false;
        for (size_t i = 0; i < input_grad.size(); ++i)
        {
            if (std::abs( input_grad.data()[i] ) > 1e-6f)
            {
                has_nonzero_grad = true;
                break;
            }
        }
        EXPECT_TRUE( has_nonzero_grad ) << "Input gradients should contain non-zero values";
    }

    // ====================================================================
    // Existing Tests
    // ====================================================================

    TEST_F( LinearCpuTests, GetName )
    {
        TestGetName( SmallFp32Data(), "small_linear_cpu" );
    }

    TEST_F( LinearCpuTests, DeviceType )
    {
        TestDeviceType( SmallFp32Data() );
    }

    TEST_F( LinearCpuTests, IsBuilt_BeforeBuild )
    {
        TestIsBuilt( SmallFp32Data(), false );
    }

    TEST_F( LinearCpuTests, IsBuilt_AfterBuild )
    {
        auto data = SmallFp32Data();

        EXPECT_FALSE( data.module->isBuilt() );

        data.module->build( data.input_shape );

        EXPECT_TRUE( data.module->isBuilt() );
    }

    TEST_F( LinearCpuTests, Build )
    {
        auto data = SmallFp32Data();
        TestBuild( data );
    }

    TEST_F( LinearCpuTests, ParameterCount_WithBias )
    {
        TestParameterCount( SmallFp32Data() );
    }

    TEST_F( LinearCpuTests, ParameterCount_WithoutBias )
    {
        TestParameterCount( NoBiasFp32Data() );
    }

    TEST_F( LinearCpuTests, GetWeight )
    {
        TestGetWeight( SmallFp32Data() );
    }

    TEST_F( LinearCpuTests, GetBias_WithBias )
    {
        TestGetBias( SmallFp32Data() );
    }

    TEST_F( LinearCpuTests, GetBias_WithoutBias )
    {
        TestGetBias( NoBiasFp32Data() );
    }

    TEST_F( LinearCpuTests, HasBias_True )
    {
        TestHasBias( SmallFp32Data() );
    }

    TEST_F( LinearCpuTests, HasBias_False )
    {
        TestHasBias( NoBiasFp32Data() );
    }

    TEST_F( LinearCpuTests, GetParameters_WithBias )
    {
        TestGetParameters( SmallFp32Data() );
    }

    TEST_F( LinearCpuTests, GetParameters_WithoutBias )
    {
        TestGetParameters( NoBiasFp32Data() );
    }

    TEST_F( LinearCpuTests, ToString )
    {
        auto data = SmallFp32Data();
        TestToString( data );
    }

    TEST_F( LinearCpuTests, Forward_SmallShape )
    {
        auto data = SmallFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCpuTests, Forward_MediumShape )
    {
        auto data = MediumFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCpuTests, Forward_LargeShape )
    {
        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCpuTests, Forward_WithoutBias )
    {
        auto data = NoBiasFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCpuTests, WithContext_Construction )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        auto data = LinearCpuTestData<TensorDataType::FP32>::CreateWithContext(
            "context_linear_cpu", small_shape_, input_features_, output_features_, ctx );

        EXPECT_EQ( data.module->getName(), "context_linear_cpu" );
        EXPECT_EQ( data.exec_context, ctx );
    }

    TEST_F( LinearCpuTests, EdgeCase_MinimalShape )
    {
        shape_t shape = { 1, 1, 1 };

        auto data = LinearCpuTestData<TensorDataType::FP32>::Create(
            "minimal_cpu", shape, 1, 1 );

        TestForward( data );
    }

    TEST_F( LinearCpuTests, EdgeCase_LargeFeatures )
    {
        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCpuTests, EdgeCase_BatchSize1 )
    {
        shape_t shape = { 1, 8, 16 };

        auto data = LinearCpuTestData<TensorDataType::FP32>::Create(
            "batch1_cpu", shape, 16, 32 );

        TestForward( data );
    }

    TEST_F( LinearCpuTests, Error_NullExecutionContext )
    {
        LinearConfig config( 16, 32 );
        config.withName( "test_cpu" );

        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> null_ctx;

        EXPECT_THROW(
            (std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>( null_ctx, config )),
            std::invalid_argument
        );
    }

    TEST_F( LinearCpuTests, Error_InvalidConfig_ZeroInputFeatures )
    {
        LinearConfig invalid_config( 0, 32 );
        invalid_config.withName( "invalid_cpu" );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        EXPECT_THROW(
            (std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>( ctx, invalid_config )),
            std::invalid_argument
        );
    }

    TEST_F( LinearCpuTests, Error_InvalidConfig_ZeroOutputFeatures )
    {
        LinearConfig invalid_config( 16, 0 );
        invalid_config.withName( "invalid_cpu" );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        EXPECT_THROW(
            (std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>( ctx, invalid_config )),
            std::invalid_argument
        );
    }

    TEST_F( LinearCpuTests, Error_ForwardBeforeBuild )
    {
        auto data = LinearCpuTestData<TensorDataType::FP32>::Create(
            "unbuild_cpu", small_shape_, input_features_, output_features_ );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.input_shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.output_shape );

        EXPECT_THROW(
            data.module->forward( input, output ),
            std::runtime_error
        );
    }

    TEST_F( LinearCpuTests, Error_ShapeMismatch )
    {
        auto data = SmallFp32Data();
        data.module->build( data.input_shape );

        shape_t wrong_shape = { 2, 3, 64 };

        CpuTensor<TensorDataType::FP32> input( "CPU", wrong_shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", { 2, 3, 32 } );

        EXPECT_THROW(
            data.module->forward( input, output ),
            std::invalid_argument
        );
    }

    TEST_F( LinearCpuTests, Synchronize )
    {
        auto data = SmallFp32Data();

        EXPECT_NO_THROW( data.module->synchronize() );
    }

    TEST_F( LinearCpuTests, SetTrainingMode )
    {
        auto data = SmallFp32Data();

        EXPECT_FALSE( data.module->isTraining() );

        data.module->setTraining( true );
        EXPECT_TRUE( data.module->isTraining() );

        data.module->setTraining( false );
        EXPECT_FALSE( data.module->isTraining() );
    }

    TEST_F( LinearCpuTests, MultipleForwardCalls )
    {
        auto data = MediumFp32Data();
        data.module->build( data.input_shape );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.input_shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.output_shape );

        for (int iter = 0; iter < 10; ++iter)
        {
            random( input, -1.0f, 1.0f );

            EXPECT_NO_THROW( data.module->forward( input, output ) );
        }
    }

    // ====================================================================
    // Backward Pass Tests
    // ====================================================================

    TEST_F( LinearCpuTests, GetWeightGrad_BeforeBackward )
    {
        auto data = SmallFp32Data();
        TestGetWeightGrad( data );
    }

    TEST_F( LinearCpuTests, GetBiasGrad_BeforeBackward_WithBias )
    {
        auto data = SmallFp32Data();
        TestGetBiasGrad( data );
    }

    TEST_F( LinearCpuTests, GetBiasGrad_BeforeBackward_WithoutBias )
    {
        auto data = NoBiasFp32Data();
        TestGetBiasGrad( data );
    }

    TEST_F( LinearCpuTests, Backward_SmallShape )
    {
        auto data = SmallFp32Data();
        TestBackward( data );
    }

    TEST_F( LinearCpuTests, Backward_MediumShape )
    {
        auto data = MediumFp32Data();
        TestBackward( data );
    }

    TEST_F( LinearCpuTests, Backward_LargeShape )
    {
        auto data = LargeFp32Data();
        TestBackward( data );
    }

    TEST_F( LinearCpuTests, Backward_WithoutBias )
    {
        auto data = NoBiasFp32Data();
        TestBackward( data );
    }

    TEST_F( LinearCpuTests, Error_BackwardBeforeBuild )
    {
        auto data = LinearCpuTestData<TensorDataType::FP32>::Create(
            "unbuild_backward_cpu", small_shape_, input_features_, output_features_ );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.input_shape );
        CpuTensor<TensorDataType::FP32> output_grad( "CPU", data.output_shape );
        CpuTensor<TensorDataType::FP32> input_grad( "CPU", data.input_shape );

        EXPECT_THROW(
            data.module->backward( input, output_grad, input_grad ),
            std::runtime_error
        );
    }

    TEST_F( LinearCpuTests, Backward_MultipleIterations )
    {
        auto data = SmallFp32Data();

        data.module->setTraining( true );
        data.module->build( data.input_shape );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.input_shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.output_shape );
        CpuTensor<TensorDataType::FP32> output_grad( "CPU", data.output_shape );
        CpuTensor<TensorDataType::FP32> input_grad( "CPU", data.input_shape );

        for (int iter = 0; iter < 5; ++iter)
        {
            random( input, -1.0f, 1.0f );
            random( output_grad, -0.1f, 0.1f );
            zeros( input_grad );

            data.module->forward( input, output );

            EXPECT_NO_THROW(
                data.module->backward( input, output_grad, input_grad )
            ) << "Backward iteration " << iter << " failed";
        }
    }

    TEST_F( LinearCpuTests, Backward_EdgeCase_MinimalShape )
    {
        shape_t shape = { 1, 1, 1 };

        auto data = LinearCpuTestData<TensorDataType::FP32>::Create(
            "minimal_backward_cpu", shape, 1, 1 );

        TestBackward( data );
    }

    TEST_F( LinearCpuTests, Backward_EdgeCase_BatchSize1 )
    {
        shape_t shape = { 1, 8, 16 };

        auto data = LinearCpuTestData<TensorDataType::FP32>::Create(
            "batch1_backward_cpu", shape, 16, 32 );

        TestBackward( data );
    }

    TEST_F( LinearCpuTests, Training_InferenceToTrainingToInference )
    {
        auto data = SmallFp32Data();

        EXPECT_FALSE( data.module->isTraining() );
        data.module->build( data.input_shape );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.input_shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.output_shape );
        CpuTensor<TensorDataType::FP32> output_grad( "CPU", data.output_shape );
        CpuTensor<TensorDataType::FP32> input_grad( "CPU", data.input_shape );

        random( input, -1.0f, 1.0f );
        random( output_grad, -0.1f, 0.1f );

        EXPECT_NO_THROW( data.module->forward( input, output ) );

        EXPECT_THROW(
            data.module->backward( input, output_grad, input_grad ),
            std::runtime_error
        ) << "Backward should fail in inference mode";

        data.module->setTraining( true );
        EXPECT_TRUE( data.module->isTraining() );

        auto weight_grad = data.module->getWeightGrad();
        ASSERT_NE( weight_grad, nullptr ) << "Gradients should be initialized when switching to training";

        EXPECT_NO_THROW( data.module->forward( input, output ) );

        zeros( input_grad );
        EXPECT_NO_THROW(
            data.module->backward( input, output_grad, input_grad )
        ) << "Backward should work after switching to training mode";

        data.module->setTraining( false );
        EXPECT_FALSE( data.module->isTraining() );

        EXPECT_NO_THROW( data.module->forward( input, output ) );

        EXPECT_THROW(
            data.module->backward( input, output_grad, input_grad ),
            std::runtime_error
        ) << "Backward should fail after switching back to inference mode";
    }

    TEST_F( LinearCpuTests, Training_EnableBeforeBuild )
    {
        auto data = SmallFp32Data();

        data.module->setTraining( true );
        EXPECT_TRUE( data.module->isTraining() );

        data.module->build( data.input_shape );

        auto weight_grad = data.module->getWeightGrad();
        ASSERT_NE( weight_grad, nullptr ) << "Weight gradients should be allocated when training enabled before build";

        if (data.has_bias)
        {
            auto bias_grad = data.module->getBiasGrad();
            ASSERT_NE( bias_grad, nullptr ) << "Bias gradients should be allocated when training enabled before build";
        }

        CpuTensor<TensorDataType::FP32> input( "CPU", data.input_shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.output_shape );
        CpuTensor<TensorDataType::FP32> output_grad( "CPU", data.output_shape );
        CpuTensor<TensorDataType::FP32> input_grad( "CPU", data.input_shape );

        random( input, -1.0f, 1.0f );
        random( output_grad, -0.1f, 0.1f );
        zeros( input_grad );

        EXPECT_NO_THROW( data.module->forward( input, output ) );
        EXPECT_NO_THROW( data.module->backward( input, output_grad, input_grad ) );
    }

    TEST_F( LinearCpuTests, Error_BackwardInInferenceMode )
    {
        auto data = SmallFp32Data();

        data.module->build( data.input_shape );
        EXPECT_FALSE( data.module->isTraining() );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.input_shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.output_shape );
        CpuTensor<TensorDataType::FP32> output_grad( "CPU", data.output_shape );
        CpuTensor<TensorDataType::FP32> input_grad( "CPU", data.input_shape );

        random( input, -1.0f, 1.0f );

        data.module->forward( input, output );

        EXPECT_THROW(
            data.module->backward( input, output_grad, input_grad ),
            std::runtime_error
        ) << "Backward should throw when module is not in training mode";
    }
}