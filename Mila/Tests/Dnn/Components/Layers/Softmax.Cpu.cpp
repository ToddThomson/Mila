#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cstdint>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    template<TensorDataType TPrecision>
    struct SoftmaxCpuTestData
    {
        shape_t shape;
        SoftmaxConfig config;
        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_context;
        std::shared_ptr<Softmax<DeviceType::Cpu, TPrecision>> module;
        int64_t axis;
        bool is_training;

        static SoftmaxCpuTestData Create(
            const std::string& name,
            const shape_t& shape,
            int64_t axis = -1,
            bool is_training = false )
        {
            SoftmaxCpuTestData data;
            data.shape = shape;
            data.axis = axis;
            data.is_training = is_training;

            data.config = SoftmaxConfig();
            data.config.withName( name )
                .withAxis( axis );

            data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            data.module = std::make_shared<Softmax<DeviceType::Cpu, TPrecision>>( data.exec_context, data.config );

			data.module->setTraining( is_training );

            return data;
        }

        static SoftmaxCpuTestData CreateWithContext(
            const std::string& name,
            const shape_t& shape,
            std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
            int64_t axis = -1,
            bool is_training = false )
        {
            SoftmaxCpuTestData data;
            data.shape = shape;
            data.axis = axis;
            data.is_training = is_training;

            data.config = SoftmaxConfig();
            data.config.withName( name )
                .withAxis( axis );

            data.exec_context = context;
            data.module = std::make_shared<Softmax<DeviceType::Cpu, TPrecision>>( data.exec_context, data.config );

			data.module->setTraining( is_training );

            return data;
        }
    };

    class SoftmaxCpuTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            small_shape_ = { 2, 3, 4 };
            medium_shape_ = { 4, 128, 1024 };
            large_shape_ = { 8, 256, 2048 };
            axis_ = -1;
        }

        shape_t small_shape_;
        shape_t medium_shape_;
        shape_t large_shape_;
        int64_t axis_;
    };

    template<TensorDataType TPrecision>
    void ValidateNormalization( const CpuTensor<TPrecision>& output, int64_t axis )
    {
        const auto& shape = output.shape();
        const int64_t ndim = static_cast<int64_t>(shape.size());

        int64_t normalized_axis = axis;
        if ( normalized_axis < 0 )
            normalized_axis = ndim + normalized_axis;

        int64_t outer_size = 1;
        for ( int64_t i = 0; i < normalized_axis; ++i )
            outer_size *= shape[i];

        int64_t dim_size = shape[normalized_axis];

        int64_t inner_size = 1;
        for ( int64_t i = normalized_axis + 1; i < ndim; ++i )
            inner_size *= shape[i];

        auto output_ptr = output.data();

        for ( int64_t outer = 0; outer < outer_size; ++outer )
        {
            for ( int64_t inner = 0; inner < inner_size; ++inner )
            {
                float sum = 0.0f;

                for ( int64_t i = 0; i < dim_size; ++i )
                {
                    size_t idx = (outer * dim_size * inner_size) + (i * inner_size) + inner;
                    sum += static_cast<float>( output_ptr[idx] );
                }

                EXPECT_NEAR( sum, 1.0f, 1e-4f ) << "Sum check failed at outer=" << outer << ", inner=" << inner;
            }
        }
    }

    TEST_F( SoftmaxCpuTests, GetName )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "small_softmax_cpu", small_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );
        EXPECT_EQ( data.module->getName(), "small_softmax_cpu" );
    }

    TEST_F( SoftmaxCpuTests, DeviceType )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "small_softmax_cpu", small_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );
        EXPECT_EQ( data.module->getDeviceType(), DeviceType::Cpu );

        ASSERT_NE( data.exec_context, nullptr );
        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( SoftmaxCpuTests, TrainingMode_Default )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "small_softmax_cpu", small_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );
        EXPECT_FALSE( data.module->isTraining() );
    }

    TEST_F( SoftmaxCpuTests, TrainingMode_Enabled )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "training_softmax_cpu", medium_shape_, axis_, true );

        ASSERT_NE( data.module, nullptr );
        EXPECT_TRUE( data.module->isTraining() );
    }

    TEST_F( SoftmaxCpuTests, IsBuilt_BeforeBuild )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "small_softmax_cpu", small_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );
        EXPECT_FALSE( data.module->isBuilt() );
    }

    TEST_F( SoftmaxCpuTests, IsBuilt_AfterBuild )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "small_softmax_cpu", small_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );
        EXPECT_NO_THROW( data.module->build( data.shape ) );
        EXPECT_TRUE( data.module->isBuilt() );
    }

    TEST_F( SoftmaxCpuTests, Build )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "small_softmax_cpu", small_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );
        EXPECT_NO_THROW( data.module->build( data.shape ) );
        EXPECT_TRUE( data.module->isBuilt() );
    }

    TEST_F( SoftmaxCpuTests, ParameterCount )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "small_softmax_cpu", small_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );
        EXPECT_EQ( data.module->parameterCount(), 0u );
    }

    TEST_F( SoftmaxCpuTests, ToString )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "small_softmax_cpu", small_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );

        std::string output = data.module->toString();

        EXPECT_NE( output.find( "Softmax" ), std::string::npos );
        EXPECT_NE( output.find( data.config.getName() ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
        EXPECT_NE( output.find( "Axis:" ), std::string::npos );
    }

    TEST_F( SoftmaxCpuTests, Forward_SmallShape )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "small_softmax_cpu", small_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.module->build( data.shape ) );

        TensorType input( "CPU", data.shape );
        TensorType output( "CPU", data.shape );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( data.module->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    TEST_F( SoftmaxCpuTests, Forward_MediumShape )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "medium_softmax_cpu", medium_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.module->build( data.shape ) );

        TensorType input( "CPU", data.shape );
        TensorType output( "CPU", data.shape );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( data.module->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    TEST_F( SoftmaxCpuTests, Forward_LargeShape )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "large_softmax_cpu", large_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.module->build( data.shape ) );

        TensorType input( "CPU", data.shape );
        TensorType output( "CPU", data.shape );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( data.module->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    TEST_F( SoftmaxCpuTests, Forward_Normalization )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "medium_softmax_cpu", medium_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );

        data.module->build( data.shape );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.shape );

        random( input, -5.0f, 5.0f );

        data.module->forward( input, output );

        ValidateNormalization( output, data.axis );
    }

    TEST_F( SoftmaxCpuTests, WithContext_Construction )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::CreateWithContext(
            "context_softmax_cpu", medium_shape_, ctx );

        ASSERT_NE( data.module, nullptr );
        EXPECT_EQ( data.module->getName(), "context_softmax_cpu" );
        EXPECT_EQ( data.exec_context, ctx );
    }

    TEST_F( SoftmaxCpuTests, EdgeCase_MinimalShape )
    {
        shape_t shape = { 1, 1, 8 };

        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "minimal_cpu", shape );

        ASSERT_NE( data.module, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.module->build( data.shape ) );

        TensorType input( "CPU", data.shape );
        TensorType output( "CPU", data.shape );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( data.module->forward( input, output ) );
    }

    TEST_F( SoftmaxCpuTests, EdgeCase_LargeVocab )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "large_softmax_cpu", large_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.module->build( data.shape ) );

        TensorType input( "CPU", data.shape );
        TensorType output( "CPU", data.shape );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( data.module->forward( input, output ) );
    }

    TEST_F( SoftmaxCpuTests, DifferentAxes_Axis0 )
    {
        shape_t test_shape = { 2, 3, 4 };

        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "axis0_cpu", test_shape, 0 );

        ASSERT_NE( data.module, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.module->build( data.shape ) );

        TensorType input( "CPU", data.shape );
        TensorType output( "CPU", data.shape );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( data.module->forward( input, output ) );
    }

    TEST_F( SoftmaxCpuTests, DifferentAxes_Axis1 )
    {
        shape_t test_shape = { 2, 3, 4 };

        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "axis1_cpu", test_shape, 1 );

        ASSERT_NE( data.module, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.module->build( data.shape ) );

        TensorType input( "CPU", data.shape );
        TensorType output( "CPU", data.shape );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( data.module->forward( input, output ) );
    }

    TEST_F( SoftmaxCpuTests, DifferentAxes_Axis2 )
    {
        shape_t test_shape = { 2, 3, 4 };

        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "axis2_cpu", test_shape, 2 );

        ASSERT_NE( data.module, nullptr );

        using TensorType = CpuTensor<TensorDataType::FP32>;

        EXPECT_NO_THROW( data.module->build( data.shape ) );

        TensorType input( "CPU", data.shape );
        TensorType output( "CPU", data.shape );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( data.module->forward( input, output ) );
    }

    TEST_F( SoftmaxCpuTests, Error_NullExecutionContext )
    {
        SoftmaxConfig config;
        config.withName( "test_cpu" ).withAxis( -1 );

        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> null_ctx;

        EXPECT_THROW(
            (std::make_shared<Softmax<DeviceType::Cpu, TensorDataType::FP32>>( null_ctx, config )),
            std::invalid_argument
        );
    }

    TEST_F( SoftmaxCpuTests, Error_ForwardBeforeBuild )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "unbuild_cpu", medium_shape_ );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.shape );

        EXPECT_THROW(
            data.module->forward( input, output ),
            std::runtime_error
        );
    }

    TEST_F( SoftmaxCpuTests, Synchronize )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "small_softmax_cpu", small_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );

        EXPECT_NO_THROW( data.module->synchronize() );
    }

    TEST_F( SoftmaxCpuTests, SetTrainingMode )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "small_softmax_cpu", small_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );

        EXPECT_FALSE( data.module->isTraining() );

        data.module->setTraining( true );
        EXPECT_TRUE( data.module->isTraining() );

        data.module->setTraining( false );
        EXPECT_FALSE( data.module->isTraining() );
    }

    TEST_F( SoftmaxCpuTests, MultipleForwardCalls )
    {
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create( "medium_softmax_cpu", medium_shape_, axis_ );

        ASSERT_NE( data.module, nullptr );

        EXPECT_NO_THROW( data.module->build( data.shape ) );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.shape );

        for ( int iter = 0; iter < 10; ++iter )
        {
            random( input, -5.0f, 5.0f );

            EXPECT_NO_THROW( data.module->forward( input, output ) );
        }
    }
}