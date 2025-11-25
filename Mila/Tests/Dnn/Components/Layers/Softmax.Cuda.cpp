#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <cuda_runtime.h>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    template<TensorDataType TPrecision>
    struct SoftmaxCudaTestData
    {
        shape_t shape;
        SoftmaxConfig config;
        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> exec_context;
        std::shared_ptr<Softmax<DeviceType::Cuda, TPrecision>> module;
        int64_t axis;
        bool is_training;

        static SoftmaxCudaTestData Create(
            const std::string& name,
            const shape_t& shape,
            int64_t axis = -1,
            bool is_training = false )
        {
            SoftmaxCudaTestData data;
            data.shape = shape;
            data.axis = axis;
            data.is_training = is_training;

            data.config = SoftmaxConfig();
            data.config.withName( name )
                .withAxis( axis );

            data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            data.module = std::make_shared<Softmax<DeviceType::Cuda, TPrecision>>( data.exec_context, data.config );

            return data;
        }

        static SoftmaxCudaTestData CreateWithContext(
            const std::string& name,
            const shape_t& shape,
            std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
            int64_t axis = -1,
            bool is_training = false )
        {
            SoftmaxCudaTestData data;
            data.shape = shape;
            data.axis = axis;
            data.is_training = is_training;

            data.config = SoftmaxConfig();
            data.config.withName( name )
                .withAxis( axis );

            data.exec_context = context;
            data.module = std::make_shared<Softmax<DeviceType::Cuda, TPrecision>>( data.exec_context, data.config );

            return data;
        }
    };

    class SoftmaxCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = 0;
            cudaError_t error = cudaGetDeviceCount( &device_count );
            cuda_available_ = (error == cudaSuccess && device_count > 0);

            if (!cuda_available_)
            {
                return;
            }

            small_shape_ = { 2, 3, 4 };
            medium_shape_ = { 64, 128, 1024 };
            large_shape_ = { 128, 256, 2048 };
            axis_ = -1;
        }

        SoftmaxCudaTestData<TensorDataType::FP32>& SmallFp32Data()
        {
            if (!small_fp32_.module)
            {
                small_fp32_ = SoftmaxCudaTestData<TensorDataType::FP32>::Create(
                    "small_softmax_cuda", small_shape_, axis_ );
            }
            return small_fp32_;
        }

        SoftmaxCudaTestData<TensorDataType::FP32>& MediumFp32Data()
        {
            if (!medium_fp32_.module)
            {
                medium_fp32_ = SoftmaxCudaTestData<TensorDataType::FP32>::Create(
                    "medium_softmax_cuda", medium_shape_, axis_ );
            }
            return medium_fp32_;
        }

        SoftmaxCudaTestData<TensorDataType::FP32>& LargeFp32Data()
        {
            if (!large_fp32_.module)
            {
                large_fp32_ = SoftmaxCudaTestData<TensorDataType::FP32>::Create(
                    "large_softmax_cuda", large_shape_, axis_ );
            }
            return large_fp32_;
        }

        SoftmaxCudaTestData<TensorDataType::FP32>& TrainingFp32Data()
        {
            if (!training_fp32_.module)
            {
                training_fp32_ = SoftmaxCudaTestData<TensorDataType::FP32>::Create(
                    "training_softmax_cuda", medium_shape_, axis_, true );
            }
            return training_fp32_;
        }

        bool cuda_available_{ false };

        shape_t small_shape_;
        shape_t medium_shape_;
        shape_t large_shape_;
        int64_t axis_;

        SoftmaxCudaTestData<TensorDataType::FP32> small_fp32_;
        SoftmaxCudaTestData<TensorDataType::FP32> medium_fp32_;
        SoftmaxCudaTestData<TensorDataType::FP32> large_fp32_;
        SoftmaxCudaTestData<TensorDataType::FP32> training_fp32_;
    };

    template<TensorDataType TPrecision>
    void TestGetName( const SoftmaxCudaTestData<TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.module->getName(), expected_name );
    }

    template<TensorDataType TPrecision>
    void TestDeviceType( const SoftmaxCudaTestData<TPrecision>& data )
    {
        EXPECT_EQ( data.module->getDeviceType(), DeviceType::Cuda );
        ASSERT_NE( data.exec_context, nullptr );

        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
    }

    template<TensorDataType TPrecision>
    void TestTrainingMode( const SoftmaxCudaTestData<TPrecision>& data, bool expected_mode )
    {
        EXPECT_EQ( data.module->isTraining(), expected_mode );
    }

    template<TensorDataType TPrecision>
    void TestIsBuilt( const SoftmaxCudaTestData<TPrecision>& data, bool expected_built )
    {
        EXPECT_EQ( data.module->isBuilt(), expected_built );
    }

    template<TensorDataType TPrecision>
    void TestBuild( SoftmaxCudaTestData<TPrecision>& data )
    {
        EXPECT_NO_THROW( data.module->build( data.shape ) );
        EXPECT_TRUE( data.module->isBuilt() );

        data.module->build( data.shape );
        EXPECT_TRUE( data.module->isBuilt() );
    }

    template<TensorDataType TPrecision>
    void TestParameterCount( const SoftmaxCudaTestData<TPrecision>& data, size_t expected_count )
    {
        EXPECT_EQ( data.module->parameterCount(), expected_count );
    }

    template<TensorDataType TPrecision>
    void TestToString( const SoftmaxCudaTestData<TPrecision>& data )
    {
        std::string output = data.module->toString();

        EXPECT_NE( output.find( "Softmax" ), std::string::npos );
        EXPECT_NE( output.find( data.config.getName() ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
        EXPECT_NE( output.find( "Axis:" ), std::string::npos );
    }

    template<TensorDataType TPrecision>
    void TestForward( SoftmaxCudaTestData<TPrecision>& data )
    {
        using DeviceTensorType = CudaTensor<TPrecision>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        data.module->build( data.shape );

        HostTensorType host_input( "CPU", data.shape );
        random( host_input, -5.0f, 5.0f );

        DeviceTensorType device_input( "CUDA:0", data.shape );
        DeviceTensorType device_output( "CUDA:0", data.shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.module->forward( device_input, device_output ) );
        EXPECT_EQ( device_output.size(), device_input.size() );
        EXPECT_EQ( device_output.shape(), device_input.shape() );

        HostTensorType host_output = toHost<TensorDataType::FP32>( device_output );
        EXPECT_EQ( host_output.size(), host_input.size() );
    }

    template<TensorDataType TPrecision>
    void ValidateNormalization( const CpuTensor<TensorDataType::FP32>& output, int64_t axis )
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

    TEST_F( SoftmaxCudaTests, GetName )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetName( SmallFp32Data(), "small_softmax_cuda" );
    }

    TEST_F( SoftmaxCudaTests, DeviceType )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestDeviceType( SmallFp32Data() );
    }

    TEST_F( SoftmaxCudaTests, TrainingMode_Default )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestTrainingMode( SmallFp32Data(), false );
    }

    TEST_F( SoftmaxCudaTests, TrainingMode_Enabled )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestTrainingMode( TrainingFp32Data(), true );
    }

    TEST_F( SoftmaxCudaTests, IsBuilt_BeforeBuild )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestIsBuilt( SmallFp32Data(), false );
    }

    TEST_F( SoftmaxCudaTests, IsBuilt_AfterBuild )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_FALSE( data.module->isBuilt() );

        data.module->build( data.shape );

        EXPECT_TRUE( data.module->isBuilt() );
    }

    TEST_F( SoftmaxCudaTests, Build )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestBuild( data );
    }

    TEST_F( SoftmaxCudaTests, ParameterCount )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestParameterCount( SmallFp32Data(), 0 );
    }

    TEST_F( SoftmaxCudaTests, ToString )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestToString( data );
    }

    TEST_F( SoftmaxCudaTests, Forward_SmallShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestForward( data );
    }

    TEST_F( SoftmaxCudaTests, Forward_MediumShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        TestForward( data );
    }

    TEST_F( SoftmaxCudaTests, Forward_LargeShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( SoftmaxCudaTests, Forward_Normalization )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        data.module->build( data.shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.shape );
        random( host_input, -5.0f, 5.0f );

        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", data.shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.shape );

        copy( host_input, device_input );

        data.module->forward( device_input, device_output );

        CpuTensor<TensorDataType::FP32> host_output = toHost<TensorDataType::FP32>( device_output );

        ValidateNormalization<TensorDataType::FP32>( host_output, data.axis );
    }

    TEST_F( SoftmaxCudaTests, WithContext_Construction )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        auto data = SoftmaxCudaTestData<TensorDataType::FP32>::CreateWithContext(
            "context_softmax_cuda", medium_shape_, ctx );

        EXPECT_EQ( data.module->getName(), "context_softmax_cuda" );
        EXPECT_EQ( data.exec_context, ctx );
    }

    TEST_F( SoftmaxCudaTests, EdgeCase_MinimalShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 1, 1, 8 };

        auto data = SoftmaxCudaTestData<TensorDataType::FP32>::Create(
            "minimal_cuda", shape );

        TestForward( data );
    }

    TEST_F( SoftmaxCudaTests, EdgeCase_LargeVocab )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( SoftmaxCudaTests, DifferentAxes_Axis0 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t test_shape = { 2, 3, 4 };

        auto data = SoftmaxCudaTestData<TensorDataType::FP32>::Create(
            "axis0_cuda", test_shape, 0 );

        TestForward( data );
    }

    TEST_F( SoftmaxCudaTests, DifferentAxes_Axis1 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t test_shape = { 2, 3, 4 };

        auto data = SoftmaxCudaTestData<TensorDataType::FP32>::Create(
            "axis1_cuda", test_shape, 1 );

        TestForward( data );
    }

    TEST_F( SoftmaxCudaTests, DifferentAxes_Axis2 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t test_shape = { 2, 3, 4 };

        auto data = SoftmaxCudaTestData<TensorDataType::FP32>::Create(
            "axis2_cuda", test_shape, 2 );

        TestForward( data );
    }

    TEST_F( SoftmaxCudaTests, Error_NullExecutionContext )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        SoftmaxConfig config;
        config.withName( "test_cuda" ).withAxis( -1 );

        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> null_ctx;

        EXPECT_THROW(
            (std::make_shared<Softmax<DeviceType::Cuda, TensorDataType::FP32>>( null_ctx, config )),
            std::invalid_argument
        );
    }

    TEST_F( SoftmaxCudaTests, Error_ForwardBeforeBuild )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SoftmaxCudaTestData<TensorDataType::FP32>::Create(
            "unbuild_cuda", medium_shape_ );

        CudaTensor<TensorDataType::FP32> input( "CUDA:0", data.shape );
        CudaTensor<TensorDataType::FP32> output( "CUDA:0", data.shape );

        EXPECT_THROW(
            data.module->forward( input, output ),
            std::runtime_error
        );
    }

    TEST_F( SoftmaxCudaTests, Synchronize )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_NO_THROW( data.module->synchronize() );
    }

    TEST_F( SoftmaxCudaTests, SetTrainingMode )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_FALSE( data.module->isTraining() );

        data.module->setTraining( true );
        EXPECT_TRUE( data.module->isTraining() );

        data.module->setTraining( false );
        EXPECT_FALSE( data.module->isTraining() );
    }

    TEST_F( SoftmaxCudaTests, MultipleForwardCalls )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        data.module->build( data.shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.shape );
        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", data.shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.shape );

        for (int iter = 0; iter < 10; ++iter)
        {
            random( host_input, -5.0f, 5.0f );
            copy( host_input, device_input );

            EXPECT_NO_THROW( data.module->forward( device_input, device_output ) );
        }
    }

    TEST_F( SoftmaxCudaTests, CpuCuda_OutputEquivalence )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t test_shape = { 2, 4, 8 };

        SoftmaxConfig cpu_config;
        cpu_config.withName( "cpu_equiv" ).withAxis( -1 );

        auto cpu_exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cpu_module = std::make_shared<Softmax<DeviceType::Cpu, TensorDataType::FP32>>( cpu_exec_context, cpu_config );

        auto cuda_data = SoftmaxCudaTestData<TensorDataType::FP32>::Create(
            "cuda_equiv", test_shape );

        cpu_module->build( test_shape );
        cuda_data.module->build( test_shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", test_shape );
        random( host_input, -2.0f, 2.0f );

        CpuTensor<TensorDataType::FP32> cpu_output( "CPU", test_shape );
        cpu_module->forward( host_input, cpu_output );

        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", test_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", test_shape );
        copy( host_input, device_input );
        cuda_data.module->forward( device_input, device_output );

        CpuTensor<TensorDataType::FP32> cuda_output_host = toHost<TensorDataType::FP32>( device_output );

        const float epsilon = 1e-4f;
        bool all_close = true;

        for (size_t i = 0; i < cpu_output.size(); ++i)
        {
            float cpu_val = cpu_output.data()[i];
            float cuda_val = cuda_output_host.data()[i];
            float diff = std::abs( cpu_val - cuda_val );

            if (diff > epsilon)
            {
                all_close = false;
                break;
            }
        }

        EXPECT_TRUE( all_close ) << "CPU and CUDA implementations produced different results";
    }
}