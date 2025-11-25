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

        SoftmaxCpuTestData<TensorDataType::FP32>& SmallFp32Data()
        {
            if (!small_fp32_.module)
            {
                small_fp32_ = SoftmaxCpuTestData<TensorDataType::FP32>::Create(
                    "small_softmax_cpu", small_shape_, axis_ );
            }
            return small_fp32_;
        }

        SoftmaxCpuTestData<TensorDataType::FP32>& MediumFp32Data()
        {
            if (!medium_fp32_.module)
            {
                medium_fp32_ = SoftmaxCpuTestData<TensorDataType::FP32>::Create(
                    "medium_softmax_cpu", medium_shape_, axis_ );
            }
            return medium_fp32_;
        }

        SoftmaxCpuTestData<TensorDataType::FP32>& LargeFp32Data()
        {
            if (!large_fp32_.module)
            {
                large_fp32_ = SoftmaxCpuTestData<TensorDataType::FP32>::Create(
                    "large_softmax_cpu", large_shape_, axis_ );
            }
            return large_fp32_;
        }

        SoftmaxCpuTestData<TensorDataType::FP32>& TrainingFp32Data()
        {
            if (!training_fp32_.module)
            {
                training_fp32_ = SoftmaxCpuTestData<TensorDataType::FP32>::Create(
                    "training_softmax_cpu", medium_shape_, axis_, true );
            }
            return training_fp32_;
        }

        shape_t small_shape_;
        shape_t medium_shape_;
        shape_t large_shape_;
        int64_t axis_;

        SoftmaxCpuTestData<TensorDataType::FP32> small_fp32_;
        SoftmaxCpuTestData<TensorDataType::FP32> medium_fp32_;
        SoftmaxCpuTestData<TensorDataType::FP32> large_fp32_;
        SoftmaxCpuTestData<TensorDataType::FP32> training_fp32_;
    };

    template<TensorDataType TPrecision>
    void TestGetName( const SoftmaxCpuTestData<TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.module->getName(), expected_name );
    }

    template<TensorDataType TPrecision>
    void TestDeviceType( const SoftmaxCpuTestData<TPrecision>& data )
    {
        EXPECT_EQ( data.module->getDeviceType(), DeviceType::Cpu );
        ASSERT_NE( data.exec_context, nullptr );

        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
    }

    template<TensorDataType TPrecision>
    void TestTrainingMode( const SoftmaxCpuTestData<TPrecision>& data, bool expected_mode )
    {
        EXPECT_EQ( data.module->isTraining(), expected_mode );
    }

    template<TensorDataType TPrecision>
    void TestIsBuilt( const SoftmaxCpuTestData<TPrecision>& data, bool expected_built )
    {
        EXPECT_EQ( data.module->isBuilt(), expected_built );
    }

    template<TensorDataType TPrecision>
    void TestBuild( SoftmaxCpuTestData<TPrecision>& data )
    {
        EXPECT_NO_THROW( data.module->build( data.shape ) );
        EXPECT_TRUE( data.module->isBuilt() );

        data.module->build( data.shape );
        EXPECT_TRUE( data.module->isBuilt() );
    }

    template<TensorDataType TPrecision>
    void TestParameterCount( const SoftmaxCpuTestData<TPrecision>& data, size_t expected_count )
    {
        EXPECT_EQ( data.module->parameterCount(), expected_count );
    }

    template<TensorDataType TPrecision>
    void TestToString( const SoftmaxCpuTestData<TPrecision>& data )
    {
        std::string output = data.module->toString();

        EXPECT_NE( output.find( "Softmax" ), std::string::npos );
        EXPECT_NE( output.find( data.config.getName() ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
        EXPECT_NE( output.find( "Axis:" ), std::string::npos );
    }

    template<TensorDataType TPrecision>
    void TestForward( SoftmaxCpuTestData<TPrecision>& data )
    {
        using TensorType = CpuTensor<TPrecision>;

        data.module->build( data.shape );

        TensorType input( "CPU", data.shape );
        TensorType output( "CPU", data.shape );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( data.module->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    template<TensorDataType TPrecision>
    void ValidateNormalization( const CpuTensor<TPrecision>& output, int64_t axis )
    {
        const auto& shape = output.shape();
        const int64_t ndim = static_cast<int64_t>(shape.size());

        int64_t normalized_axis = axis;
        if (normalized_axis < 0)
            normalized_axis = ndim + normalized_axis;

        int64_t outer_size = 1;
        for (int64_t i = 0; i < normalized_axis; ++i)
            outer_size *= shape[i];

        int64_t dim_size = shape[normalized_axis];

        int64_t inner_size = 1;
        for (int64_t i = normalized_axis + 1; i < ndim; ++i)
            inner_size *= shape[i];

        auto output_ptr = output.data();

        for (int64_t outer = 0; outer < outer_size; ++outer)
        {
            for (int64_t inner = 0; inner < inner_size; ++inner)
            {
                float sum = 0.0f;

                for (int64_t i = 0; i < dim_size; ++i)
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
        TestGetName( SmallFp32Data(), "small_softmax_cpu" );
    }

    TEST_F( SoftmaxCpuTests, DeviceType )
    {
        TestDeviceType( SmallFp32Data() );
    }

    TEST_F( SoftmaxCpuTests, TrainingMode_Default )
    {
        TestTrainingMode( SmallFp32Data(), false );
    }

    TEST_F( SoftmaxCpuTests, TrainingMode_Enabled )
    {
        TestTrainingMode( TrainingFp32Data(), true );
    }

    TEST_F( SoftmaxCpuTests, IsBuilt_BeforeBuild )
    {
        TestIsBuilt( SmallFp32Data(), false );
    }

    TEST_F( SoftmaxCpuTests, IsBuilt_AfterBuild )
    {
        auto data = SmallFp32Data();

        EXPECT_FALSE( data.module->isBuilt() );

        data.module->build( data.shape );

        EXPECT_TRUE( data.module->isBuilt() );
    }

    TEST_F( SoftmaxCpuTests, Build )
    {
        auto data = SmallFp32Data();
        TestBuild( data );
    }

    TEST_F( SoftmaxCpuTests, ParameterCount )
    {
        TestParameterCount( SmallFp32Data(), 0 );
    }

    TEST_F( SoftmaxCpuTests, ToString )
    {
        auto data = SmallFp32Data();
        TestToString( data );
    }

    TEST_F( SoftmaxCpuTests, Forward_SmallShape )
    {
        auto data = SmallFp32Data();
        TestForward( data );
    }

    TEST_F( SoftmaxCpuTests, Forward_MediumShape )
    {
        auto data = MediumFp32Data();
        TestForward( data );
    }

    TEST_F( SoftmaxCpuTests, Forward_LargeShape )
    {
        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( SoftmaxCpuTests, Forward_Normalization )
    {
        auto data = MediumFp32Data();
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

        EXPECT_EQ( data.module->getName(), "context_softmax_cpu" );
        EXPECT_EQ( data.exec_context, ctx );
    }

    TEST_F( SoftmaxCpuTests, EdgeCase_MinimalShape )
    {
        shape_t shape = { 1, 1, 8 };

        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create(
            "minimal_cpu", shape );

        TestForward( data );
    }

    TEST_F( SoftmaxCpuTests, EdgeCase_LargeVocab )
    {
        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( SoftmaxCpuTests, DifferentAxes_Axis0 )
    {
        shape_t test_shape = { 2, 3, 4 };

        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create(
            "axis0_cpu", test_shape, 0 );

        TestForward( data );
    }

    TEST_F( SoftmaxCpuTests, DifferentAxes_Axis1 )
    {
        shape_t test_shape = { 2, 3, 4 };

        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create(
            "axis1_cpu", test_shape, 1 );

        TestForward( data );
    }

    TEST_F( SoftmaxCpuTests, DifferentAxes_Axis2 )
    {
        shape_t test_shape = { 2, 3, 4 };

        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create(
            "axis2_cpu", test_shape, 2 );

        TestForward( data );
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
        auto data = SoftmaxCpuTestData<TensorDataType::FP32>::Create(
            "unbuild_cpu", medium_shape_ );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.shape );

        EXPECT_THROW(
            data.module->forward( input, output ),
            std::runtime_error
        );
    }

    TEST_F( SoftmaxCpuTests, Synchronize )
    {
        auto data = SmallFp32Data();

        EXPECT_NO_THROW( data.module->synchronize() );
    }

    TEST_F( SoftmaxCpuTests, SetTrainingMode )
    {
        auto data = SmallFp32Data();

        EXPECT_FALSE( data.module->isTraining() );

        data.module->setTraining( true );
        EXPECT_TRUE( data.module->isTraining() );

        data.module->setTraining( false );
        EXPECT_FALSE( data.module->isTraining() );
    }

    TEST_F( SoftmaxCpuTests, MultipleForwardCalls )
    {
        auto data = MediumFp32Data();
        data.module->build( data.shape );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.shape );

        for (int iter = 0; iter < 10; ++iter)
        {
            random( input, -5.0f, 5.0f );

            EXPECT_NO_THROW( data.module->forward( input, output ) );
        }
    }
}