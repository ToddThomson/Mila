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
    struct LinearCudaTestData
    {
        shape_t input_shape;
        shape_t output_shape;
        LinearConfig config;
        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> exec_context;
        std::shared_ptr<Linear<DeviceType::Cuda, TPrecision>> module;
        int64_t input_features;
        int64_t output_features;
        bool has_bias;

        LinearCudaTestData() : config( 1, 1 ), input_features( 0 ), output_features( 0 ), has_bias( true )
        {
        }

        static LinearCudaTestData Create(
            const std::string& name,
            const shape_t& input_shape,
            int64_t input_features,
            int64_t output_features,
            bool has_bias = true )
        {
            LinearCudaTestData data;
            data.input_shape = input_shape;
            data.input_features = input_features;
            data.output_features = output_features;
            data.has_bias = has_bias;

            data.output_shape = input_shape;
            data.output_shape.back() = output_features;

            data.config = LinearConfig( input_features, output_features );
            data.config.withName( name )
                .withBias( has_bias );

            data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            data.module = std::make_shared<Linear<DeviceType::Cuda, TPrecision>>( data.exec_context, data.config );

            return data;
        }

        static LinearCudaTestData CreateWithContext(
            const std::string& name,
            const shape_t& input_shape,
            int64_t input_features,
            int64_t output_features,
            std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
            bool has_bias = true )
        {
            LinearCudaTestData data;
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
            data.module = std::make_shared<Linear<DeviceType::Cuda, TPrecision>>( data.exec_context, data.config );

            return data;
        }
    };

    class LinearCudaTests : public ::testing::Test
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

            small_shape_ = { 2, 3, 16 };
            medium_shape_ = { 64, 128, 512 };
            large_shape_ = { 128, 256, 1024 };

            input_features_ = 16;
            output_features_ = 32;
        }

        LinearCudaTestData<TensorDataType::FP32>& SmallFp32Data()
        {
            if (!small_fp32_.module)
            {
                small_fp32_ = LinearCudaTestData<TensorDataType::FP32>::Create(
                    "small_linear_cuda", small_shape_, input_features_, output_features_ );
            }
            return small_fp32_;
        }

        LinearCudaTestData<TensorDataType::FP32>& MediumFp32Data()
        {
            if (!medium_fp32_.module)
            {
                medium_fp32_ = LinearCudaTestData<TensorDataType::FP32>::Create(
                    "medium_linear_cuda", medium_shape_, 512, 256 );
            }
            return medium_fp32_;
        }

        LinearCudaTestData<TensorDataType::FP32>& LargeFp32Data()
        {
            if (!large_fp32_.module)
            {
                large_fp32_ = LinearCudaTestData<TensorDataType::FP32>::Create(
                    "large_linear_cuda", large_shape_, 1024, 768 );
            }
            return large_fp32_;
        }

        LinearCudaTestData<TensorDataType::FP32>& NoBiasFp32Data()
        {
            if (!no_bias_fp32_.module)
            {
                no_bias_fp32_ = LinearCudaTestData<TensorDataType::FP32>::Create(
                    "no_bias_linear_cuda", small_shape_, input_features_, output_features_, false );
            }
            return no_bias_fp32_;
        }

        LinearCudaTestData<TensorDataType::FP16>& SmallFp16Data()
        {
            if (!small_fp16_.module)
            {
                small_fp16_ = LinearCudaTestData<TensorDataType::FP16>::Create(
                    "small_linear_cuda_fp16", small_shape_, input_features_, output_features_ );
            }
            return small_fp16_;
        }

        LinearCudaTestData<TensorDataType::BF16>& SmallBf16Data()
        {
            if (!small_bf16_.module)
            {
                small_bf16_ = LinearCudaTestData<TensorDataType::BF16>::Create(
                    "small_linear_cuda_bf16", small_shape_, input_features_, output_features_ );
            }
            return small_bf16_;
        }

        bool cuda_available_{ false };

        shape_t small_shape_;
        shape_t medium_shape_;
        shape_t large_shape_;
        int64_t input_features_;
        int64_t output_features_;

        LinearCudaTestData<TensorDataType::FP32> small_fp32_;
        LinearCudaTestData<TensorDataType::FP32> medium_fp32_;
        LinearCudaTestData<TensorDataType::FP32> large_fp32_;
        LinearCudaTestData<TensorDataType::FP32> no_bias_fp32_;
        LinearCudaTestData<TensorDataType::FP16> small_fp16_;
        LinearCudaTestData<TensorDataType::BF16> small_bf16_;
    };

    template<TensorDataType TPrecision>
    void TestGetName( const LinearCudaTestData<TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.module->getName(), expected_name );
    }

    template<TensorDataType TPrecision>
    void TestDeviceType( const LinearCudaTestData<TPrecision>& data )
    {
        EXPECT_EQ( data.module->getDeviceType(), DeviceType::Cuda );
        ASSERT_NE( data.exec_context, nullptr );

        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
    }

    template<TensorDataType TPrecision>
    void TestIsBuilt( const LinearCudaTestData<TPrecision>& data, bool expected_built )
    {
        EXPECT_EQ( data.module->isBuilt(), expected_built );
    }

    template<TensorDataType TPrecision>
    void TestBuild( LinearCudaTestData<TPrecision>& data )
    {
        EXPECT_NO_THROW( data.module->build( data.input_shape ) );
        EXPECT_TRUE( data.module->isBuilt() );

        data.module->build( data.input_shape );
        EXPECT_TRUE( data.module->isBuilt() );
    }

    template<TensorDataType TPrecision>
    void TestParameterCount( const LinearCudaTestData<TPrecision>& data )
    {
        size_t expected_count = data.input_features * data.output_features;
        if (data.has_bias)
        {
            expected_count += data.output_features;
        }
        EXPECT_EQ( data.module->parameterCount(), expected_count );
    }

    template<TensorDataType TPrecision>
    void TestGetWeight( const LinearCudaTestData<TPrecision>& data )
    {
        auto weight = data.module->getWeight();
        ASSERT_NE( weight, nullptr );
        EXPECT_EQ( weight->shape()[0], data.output_features );
        EXPECT_EQ( weight->shape()[1], data.input_features );
    }

    template<TensorDataType TPrecision>
    void TestGetBias( const LinearCudaTestData<TPrecision>& data )
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
    void TestHasBias( const LinearCudaTestData<TPrecision>& data )
    {
        EXPECT_EQ( data.module->hasBias(), data.has_bias );
    }

    template<TensorDataType TPrecision>
    void TestToString( const LinearCudaTestData<TPrecision>& data )
    {
        std::string output = data.module->toString();

        EXPECT_NE( output.find( "Linear" ), std::string::npos );
        EXPECT_NE( output.find( data.config.getName() ), std::string::npos );
        EXPECT_NE( output.find( "Input features:" ), std::string::npos );
        EXPECT_NE( output.find( "Output features:" ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
    }

    template<TensorDataType TPrecision>
    void TestForward( LinearCudaTestData<TPrecision>& data )
    {
        using DeviceTensorType = CudaTensor<TPrecision>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        data.module->build( data.input_shape );

        HostTensorType host_input( "CPU", data.input_shape );
        random( host_input, -1.0f, 1.0f );

        DeviceTensorType device_input( "CUDA:0", data.input_shape );
        DeviceTensorType device_output( "CUDA:0", data.output_shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.module->forward( device_input, device_output ) );
        EXPECT_EQ( device_output.size(),
            data.output_shape[0] * data.output_shape[1] * data.output_shape[2] );
        EXPECT_EQ( device_output.shape(), data.output_shape );

        HostTensorType host_output = toHost<TensorDataType::FP32>( device_output );
        EXPECT_EQ( host_output.size(), device_output.size() );
    }

    template<TensorDataType TPrecision>
    void TestGetParameters( const LinearCudaTestData<TPrecision>& data )
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
    void TestGetWeightGrad( LinearCudaTestData<TPrecision>& data )
    {
        // Enable training mode BEFORE build to initialize gradients
        data.module->setTraining( true );
        data.module->build( data.input_shape );

        auto weight_grad = data.module->getWeightGrad();

        // In training mode, gradients should be allocated
        ASSERT_NE( weight_grad, nullptr ) << "Weight gradients should be allocated in training mode";
        EXPECT_EQ( weight_grad->shape()[0], data.output_features );
        EXPECT_EQ( weight_grad->shape()[1], data.input_features );
    }

    template<TensorDataType TPrecision>
    void TestGetBiasGrad( LinearCudaTestData<TPrecision>& data )
    {
        // Enable training mode BEFORE build to initialize gradients
        data.module->setTraining( true );
        data.module->build( data.input_shape );

        auto bias_grad = data.module->getBiasGrad();

        if (data.has_bias)
        {
            // In training mode with bias, bias gradient should be allocated
            ASSERT_NE( bias_grad, nullptr ) << "Bias gradients should be allocated in training mode";
            EXPECT_EQ( bias_grad->shape()[0], data.output_features );
        }
        else
        {
            EXPECT_EQ( bias_grad, nullptr ) << "No bias gradient when bias is disabled";
        }
    }

    template<TensorDataType TPrecision>
    void TestBackward( LinearCudaTestData<TPrecision>& data )
    {
        using DeviceTensorType = CudaTensor<TPrecision>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        // CRITICAL: Enable training mode BEFORE build
        data.module->setTraining( true );
        data.module->build( data.input_shape );

        HostTensorType host_input( "CPU", data.input_shape );
        HostTensorType host_output_grad( "CPU", data.output_shape );
        random( host_input, -1.0f, 1.0f );
        random( host_output_grad, -0.1f, 0.1f );

        DeviceTensorType device_input( "CUDA:0", data.input_shape );
        DeviceTensorType device_output( "CUDA:0", data.output_shape );
        DeviceTensorType device_output_grad( "CUDA:0", data.output_shape );
        DeviceTensorType device_input_grad( "CUDA:0", data.input_shape );

        copy( host_input, device_input );
        copy( host_output_grad, device_output_grad );
        zeros( device_input_grad );

        data.module->forward( device_input, device_output );

        EXPECT_NO_THROW(
            data.module->backward( device_input, device_output_grad, device_input_grad )
        ) << "Backward pass should succeed for CUDA Linear operation in training mode";

        EXPECT_EQ( device_input_grad.shape(), data.input_shape );

        HostTensorType host_input_grad = toHost<TensorDataType::FP32>( device_input_grad );
        EXPECT_EQ( host_input_grad.size(), device_input_grad.size() );

        // Verify gradients were computed (non-zero)
        bool has_nonzero_grad = false;
        for (size_t i = 0; i < host_input_grad.size(); ++i)
        {
            if (std::abs( host_input_grad.data()[i] ) > 1e-6f)
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

    TEST_F( LinearCudaTests, GetName )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetName( SmallFp32Data(), "small_linear_cuda" );
    }

    TEST_F( LinearCudaTests, DeviceType )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestDeviceType( SmallFp32Data() );
    }

    TEST_F( LinearCudaTests, IsBuilt_BeforeBuild )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestIsBuilt( SmallFp32Data(), false );
    }

    TEST_F( LinearCudaTests, IsBuilt_AfterBuild )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_FALSE( data.module->isBuilt() );

        data.module->build( data.input_shape );

        EXPECT_TRUE( data.module->isBuilt() );
    }

    TEST_F( LinearCudaTests, Build )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestBuild( data );
    }

    TEST_F( LinearCudaTests, ParameterCount_WithBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestParameterCount( SmallFp32Data() );
    }

    TEST_F( LinearCudaTests, ParameterCount_WithoutBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestParameterCount( NoBiasFp32Data() );
    }

    TEST_F( LinearCudaTests, GetWeight )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetWeight( SmallFp32Data() );
    }

    TEST_F( LinearCudaTests, GetBias_WithBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetBias( SmallFp32Data() );
    }

    TEST_F( LinearCudaTests, GetBias_WithoutBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetBias( NoBiasFp32Data() );
    }

    TEST_F( LinearCudaTests, HasBias_True )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestHasBias( SmallFp32Data() );
    }

    TEST_F( LinearCudaTests, HasBias_False )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestHasBias( NoBiasFp32Data() );
    }

    TEST_F( LinearCudaTests, GetParameters_WithBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetParameters( SmallFp32Data() );
    }

    TEST_F( LinearCudaTests, GetParameters_WithoutBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetParameters( NoBiasFp32Data() );
    }

    TEST_F( LinearCudaTests, ToString )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestToString( data );
    }

    TEST_F( LinearCudaTests, Forward_SmallShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCudaTests, Forward_MediumShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCudaTests, Forward_LargeShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCudaTests, Forward_WithoutBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = NoBiasFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCudaTests, Forward_FP16 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp16Data();
        TestForward( data );
    }

    TEST_F( LinearCudaTests, Forward_BF16 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallBf16Data();
        TestForward( data );
    }

    TEST_F( LinearCudaTests, WithContext_Construction )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        auto data = LinearCudaTestData<TensorDataType::FP32>::CreateWithContext(
            "context_linear_cuda", small_shape_, input_features_, output_features_, ctx );

        EXPECT_EQ( data.module->getName(), "context_linear_cuda" );
        EXPECT_EQ( data.exec_context, ctx );
    }

    TEST_F( LinearCudaTests, EdgeCase_MinimalShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 1, 1, 1 };

        auto data = LinearCudaTestData<TensorDataType::FP32>::Create(
            "minimal_cuda", shape, 1, 1 );

        TestForward( data );
    }

    TEST_F( LinearCudaTests, EdgeCase_LargeFeatures )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCudaTests, EdgeCase_BatchSize1 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 1, 8, 16 };

        auto data = LinearCudaTestData<TensorDataType::FP32>::Create(
            "batch1_cuda", shape, 16, 32 );

        TestForward( data );
    }

    TEST_F( LinearCudaTests, Error_NullExecutionContext )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        LinearConfig config( 16, 32 );
        config.withName( "test_cuda" );

        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> null_ctx;

        EXPECT_THROW(
            (std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>( null_ctx, config )),
            std::invalid_argument
        );
    }

    TEST_F( LinearCudaTests, Error_ForwardBeforeBuild )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LinearCudaTestData<TensorDataType::FP32>::Create(
            "unbuild_cuda", small_shape_, input_features_, output_features_ );

        CudaTensor<TensorDataType::FP32> input( "CUDA:0", data.input_shape );
        CudaTensor<TensorDataType::FP32> output( "CUDA:0", data.output_shape );

        EXPECT_THROW(
            data.module->forward( input, output ),
            std::runtime_error
        );
    }

    TEST_F( LinearCudaTests, Error_ShapeMismatch )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.module->build( data.input_shape );

        shape_t wrong_shape = { 2, 3, 64 };

        CudaTensor<TensorDataType::FP32> input( "CUDA:0", wrong_shape );
        CudaTensor<TensorDataType::FP32> output( "CUDA:0", { 2, 3, 32 } );

        EXPECT_THROW(
            data.module->forward( input, output ),
            std::invalid_argument
        );
    }

    TEST_F( LinearCudaTests, Synchronize )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_NO_THROW( data.module->synchronize() );
    }

    TEST_F( LinearCudaTests, SetTrainingMode )
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

    TEST_F( LinearCudaTests, MultipleForwardCalls )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        data.module->build( data.input_shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.input_shape );
        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", data.input_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.output_shape );

        for (int iter = 0; iter < 10; ++iter)
        {
            random( host_input, -1.0f, 1.0f );
            copy( host_input, device_input );

            EXPECT_NO_THROW( data.module->forward( device_input, device_output ) );
        }
    }

    TEST_F( LinearCudaTests, CpuCuda_OutputEquivalence )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t test_shape = { 2, 4, 8 };

        LinearConfig cpu_config( 8, 16 );
        cpu_config.withName( "cpu_equiv" ).withBias( true );

        auto cpu_exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cpu_module = std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>( cpu_exec_context, cpu_config );

        auto cuda_data = LinearCudaTestData<TensorDataType::FP32>::Create(
            "cuda_equiv", test_shape, 8, 16, true );

        cpu_module->build( test_shape );
        cuda_data.module->build( test_shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", test_shape );
        random( host_input, -1.0f, 1.0f );

        auto cpu_weight = cpu_module->getWeight();
        auto cuda_weight = cuda_data.module->getWeight();

        CpuTensor<TensorDataType::FP32> init_weight( "CPU", cpu_weight->shape() );
        fill( init_weight, 0.1f );

        copy( init_weight, *cpu_weight );
        copy( init_weight, *cuda_weight );

        auto cpu_bias = cpu_module->getBias();
        auto cuda_bias = cuda_data.module->getBias();

        if (cpu_bias && cuda_bias)
        {
            CpuTensor<TensorDataType::FP32> init_bias( "CPU", cpu_bias->shape() );
            zeros( init_bias );

            copy( init_bias, *cpu_bias );
            copy( init_bias, *cuda_bias );
        }

        shape_t output_shape = test_shape;
        output_shape.back() = 16;

        CpuTensor<TensorDataType::FP32> cpu_output( "CPU", output_shape );
        cpu_module->forward( host_input, cpu_output );

        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", test_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", output_shape );
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

    // ====================================================================
// Backward Pass Tests
// ====================================================================

    TEST_F( LinearCudaTests, GetWeightGrad_BeforeBackward )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestGetWeightGrad( data );
    }

    TEST_F( LinearCudaTests, GetBiasGrad_BeforeBackward_WithBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestGetBiasGrad( data );
    }

    TEST_F( LinearCudaTests, GetBiasGrad_BeforeBackward_WithoutBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = NoBiasFp32Data();
        TestGetBiasGrad( data );
    }

    TEST_F( LinearCudaTests, Backward_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestBackward( data );
    }

    TEST_F( LinearCudaTests, Backward_FP16 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp16Data();
        TestBackward( data );
    }

    TEST_F( LinearCudaTests, Backward_BF16 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallBf16Data();

        // BF16 forward is not yet implemented, so backward will fail
        EXPECT_THROW(
            TestBackward( data ),
            std::logic_error
        ) << "BF16 forward not yet implemented";
    }

    TEST_F( LinearCudaTests, Backward_WithoutBias_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = NoBiasFp32Data();
        TestBackward( data );
    }

    TEST_F( LinearCudaTests, Backward_MediumShape_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        TestBackward( data );
    }

    TEST_F( LinearCudaTests, Backward_LargeShape_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LargeFp32Data();
        TestBackward( data );
    }

    TEST_F( LinearCudaTests, Error_BackwardBeforeBuild_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LinearCudaTestData<TensorDataType::FP32>::Create(
            "unbuild_backward_cuda", small_shape_, input_features_, output_features_ );

        CudaTensor<TensorDataType::FP32> input( "CUDA:0", data.input_shape );
        CudaTensor<TensorDataType::FP32> output_grad( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> input_grad( "CUDA:0", data.input_shape );

        EXPECT_THROW(
            data.module->backward( input, output_grad, input_grad ),
            std::runtime_error
        );
    }

    TEST_F( LinearCudaTests, Backward_EdgeCase_MinimalShape_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 1, 1, 1 };

        auto data = LinearCudaTestData<TensorDataType::FP32>::Create(
            "minimal_backward_cuda", shape, 1, 1 );

        TestBackward( data );
    }

    TEST_F( LinearCudaTests, Backward_EdgeCase_BatchSize1_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 1, 8, 16 };

        auto data = LinearCudaTestData<TensorDataType::FP32>::Create(
            "batch1_backward_cuda", shape, 16, 32 );

        TestBackward( data );
    }

    TEST_F( LinearCudaTests, Backward_MultipleIterations_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        // Enable training mode BEFORE build
        data.module->setTraining( true );
        data.module->build( data.input_shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.input_shape );
        CpuTensor<TensorDataType::FP32> host_output_grad( "CPU", data.output_shape );

        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", data.input_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_output_grad( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_input_grad( "CUDA:0", data.input_shape );

        for (int iter = 0; iter < 5; ++iter)
        {
            random( host_input, -1.0f, 1.0f );
            random( host_output_grad, -0.1f, 0.1f );

            copy( host_input, device_input );
            copy( host_output_grad, device_output_grad );
            zeros( device_input_grad );

            data.module->forward( device_input, device_output );

            EXPECT_NO_THROW(
                data.module->backward( device_input, device_output_grad, device_input_grad )
            ) << "Backward iteration " << iter << " failed";
        }
    }

    TEST_F( LinearCudaTests, Training_InferenceToTrainingToInference_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        // Build in inference mode (default)
        EXPECT_FALSE( data.module->isTraining() );
        data.module->build( data.input_shape );

        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", data.input_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_output_grad( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_input_grad( "CUDA:0", data.input_shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.input_shape );
        random( host_input, -1.0f, 1.0f );
        copy( host_input, device_input );

        // Forward in inference mode should work
        EXPECT_NO_THROW( data.module->forward( device_input, device_output ) );

        // Backward should fail in inference mode
        EXPECT_THROW(
            data.module->backward( device_input, device_output_grad, device_input_grad ),
            std::runtime_error
        ) << "Backward should fail in inference mode";

        // Switch to training mode AFTER build
        data.module->setTraining( true );
        EXPECT_TRUE( data.module->isTraining() );

        // Verify gradients were initialized by setTraining(true)
        auto weight_grad = data.module->getWeightGrad();
        ASSERT_NE( weight_grad, nullptr ) << "Gradients should be initialized when switching to training";

        // Forward and backward should now work
        EXPECT_NO_THROW( data.module->forward( device_input, device_output ) );

        CpuTensor<TensorDataType::FP32> host_output_grad( "CPU", data.output_shape );
        random( host_output_grad, -0.1f, 0.1f );
        copy( host_output_grad, device_output_grad );
        zeros( device_input_grad );

        EXPECT_NO_THROW(
            data.module->backward( device_input, device_output_grad, device_input_grad )
        ) << "Backward should work after switching to training mode";

        // Switch back to inference
        data.module->setTraining( false );
        EXPECT_FALSE( data.module->isTraining() );

        // Forward should still work
        EXPECT_NO_THROW( data.module->forward( device_input, device_output ) );

        // Backward should fail again
        EXPECT_THROW(
            data.module->backward( device_input, device_output_grad, device_input_grad ),
            std::runtime_error
        ) << "Backward should fail after switching back to inference mode";
    }

    TEST_F( LinearCudaTests, Training_EnableBeforeBuild_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        // Enable training BEFORE build
        data.module->setTraining( true );
        EXPECT_TRUE( data.module->isTraining() );

        data.module->build( data.input_shape );

        // Verify gradients were allocated during build
        auto weight_grad = data.module->getWeightGrad();
        ASSERT_NE( weight_grad, nullptr ) << "Weight gradients should be allocated when training enabled before build";

        if (data.has_bias)
        {
            auto bias_grad = data.module->getBiasGrad();
            ASSERT_NE( bias_grad, nullptr ) << "Bias gradients should be allocated when training enabled before build";
        }

        // Run a training iteration
        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", data.input_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_output_grad( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_input_grad( "CUDA:0", data.input_shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.input_shape );
        CpuTensor<TensorDataType::FP32> host_output_grad( "CPU", data.output_shape );
        random( host_input, -1.0f, 1.0f );
        random( host_output_grad, -0.1f, 0.1f );

        copy( host_input, device_input );
        copy( host_output_grad, device_output_grad );
        zeros( device_input_grad );

        EXPECT_NO_THROW( data.module->forward( device_input, device_output ) );
        EXPECT_NO_THROW( data.module->backward( device_input, device_output_grad, device_input_grad ) );
    }

    TEST_F( LinearCudaTests, Error_BackwardInInferenceMode_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        // Build in inference mode (default)
        data.module->build( data.input_shape );
        EXPECT_FALSE( data.module->isTraining() );

        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", data.input_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_output_grad( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_input_grad( "CUDA:0", data.input_shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.input_shape );
        random( host_input, -1.0f, 1.0f );
        copy( host_input, device_input );

        data.module->forward( device_input, device_output );

        // Backward should throw when not in training mode
        EXPECT_THROW(
            data.module->backward( device_input, device_output_grad, device_input_grad ),
            std::runtime_error
        ) << "Backward should throw when module is not in training mode";
    }
}