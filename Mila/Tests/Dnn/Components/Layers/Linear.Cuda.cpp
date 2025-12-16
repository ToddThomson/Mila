#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Dnn::Components::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    /**
     * @brief Test data structure for CUDA Linear component tests.
     *
     * Updated to use the new Linear constructor signature that accepts an explicit
     * component name. Config objects no longer carry a name; name is provided at
     * component construction time.
     */
    template<TensorDataType TPrecision>
    struct LinearCudaTestData
    {
        shape_t input_shape;
        shape_t output_shape;
        LinearConfig config;
        std::shared_ptr<Linear<DeviceType::Cuda, TPrecision>> component;
        int64_t input_features;
        int64_t output_features;
        bool has_bias;

        LinearCudaTestData() : config( 1, 1 ), input_features( 0 ), output_features( 0 ), has_bias( true )
        {}

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
            data.config.withBias( has_bias );

            // Construct using name + DeviceId (component will create/own its context).
            data.component = std::make_shared<Linear<DeviceType::Cuda, TPrecision>>(
                name,
                data.config,
                Device::Cuda( 0 )
            );

            return data;
        }

        static LinearCudaTestData CreateWithContext(
            const std::string& name,
            const shape_t& input_shape,
            int64_t input_features,
            int64_t output_features,
            std::unique_ptr<IExecutionContext> context,
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
            data.config.withBias( has_bias );

            // Use DeviceId from provided context to construct component.
            DeviceId ctx_id = context->getDeviceId();

            data.component = std::make_shared<Linear<DeviceType::Cuda, TPrecision>>(
                name,
                data.config,
                ctx_id
            );

            return data;
        }
    };

    class LinearCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = getDeviceCount( DeviceType::Cuda );
            cuda_available_ = (device_count > 0);

            if ( !cuda_available_ )
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
            if ( !small_fp32_.component )
            {
                small_fp32_ = LinearCudaTestData<TensorDataType::FP32>::Create(
                    "small_linear_cuda", small_shape_, input_features_, output_features_ );
            }
            return small_fp32_;
        }

        LinearCudaTestData<TensorDataType::FP32>& MediumFp32Data()
        {
            if ( !medium_fp32_.component )
            {
                medium_fp32_ = LinearCudaTestData<TensorDataType::FP32>::Create(
                    "medium_linear_cuda", medium_shape_, 512, 256 );
            }
            return medium_fp32_;
        }

        LinearCudaTestData<TensorDataType::FP32>& LargeFp32Data()
        {
            if ( !large_fp32_.component )
            {
                large_fp32_ = LinearCudaTestData<TensorDataType::FP32>::Create(
                    "large_linear_cuda", large_shape_, 1024, 768 );
            }
            return large_fp32_;
        }

        LinearCudaTestData<TensorDataType::FP32>& NoBiasFp32Data()
        {
            if ( !no_bias_fp32_.component )
            {
                no_bias_fp32_ = LinearCudaTestData<TensorDataType::FP32>::Create(
                    "no_bias_linear_cuda", small_shape_, input_features_, output_features_, false );
            }
            return no_bias_fp32_;
        }

        LinearCudaTestData<TensorDataType::FP16>& SmallFp16Data()
        {
            if ( !small_fp16_.component )
            {
                small_fp16_ = LinearCudaTestData<TensorDataType::FP16>::Create(
                    "small_linear_cuda_fp16", small_shape_, input_features_, output_features_ );
            }
            return small_fp16_;
        }

        LinearCudaTestData<TensorDataType::BF16>& SmallBf16Data()
        {
            if ( !small_bf16_.component )
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
        EXPECT_EQ( data.component->getName(), expected_name );
    }

    template<TensorDataType TPrecision>
    void TestDeviceType( const LinearCudaTestData<TPrecision>& data )
    {
        EXPECT_EQ( data.component->getDeviceType(), DeviceType::Cuda );

        auto device = data.component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cuda );
    }

    template<TensorDataType TPrecision>
    void TestIsBuilt( const LinearCudaTestData<TPrecision>& data, bool expected_built )
    {
        EXPECT_EQ( data.component->isBuilt(), expected_built );
    }

    template<TensorDataType TPrecision>
    void TestBuild( LinearCudaTestData<TPrecision>& data )
    {
        EXPECT_NO_THROW( data.component->build( data.input_shape ) );
        EXPECT_TRUE( data.component->isBuilt() );
    }

    template<TensorDataType TPrecision>
    void TestParameterCount( const LinearCudaTestData<TPrecision>& data )
    {
        size_t expected_count = data.input_features * data.output_features;
        if ( data.has_bias )
        {
            expected_count += data.output_features;
        }
        EXPECT_EQ( data.component->parameterCount(), expected_count );
    }

    template<TensorDataType TPrecision>
    void TestGetWeight( const LinearCudaTestData<TPrecision>& data )
    {
        auto weight = data.component->getWeight();
        ASSERT_NE( weight, nullptr );
        EXPECT_EQ( weight->shape()[ 0 ], data.output_features );
        EXPECT_EQ( weight->shape()[ 1 ], data.input_features );
    }

    template<TensorDataType TPrecision>
    void TestGetBias( const LinearCudaTestData<TPrecision>& data )
    {
        auto bias = data.component->getBias();

        if ( data.has_bias )
        {
            ASSERT_NE( bias, nullptr );
            EXPECT_EQ( bias->shape()[ 0 ], data.output_features );
        }
        else
        {
            EXPECT_EQ( bias, nullptr );
        }
    }

    template<TensorDataType TPrecision>
    void TestHasBias( const LinearCudaTestData<TPrecision>& data )
    {
        EXPECT_EQ( data.component->hasBias(), data.has_bias );
    }

    template<TensorDataType TPrecision>
    void TestToString( const LinearCudaTestData<TPrecision>& data )
    {
        std::string output = data.component->toString();

        EXPECT_NE( output.find( "Linear" ), std::string::npos );
        EXPECT_NE( output.find( data.component->getName() ), std::string::npos );
        EXPECT_NE( output.find( "Input features:" ), std::string::npos );
        EXPECT_NE( output.find( "Output features:" ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
    }

    template<TensorDataType TPrecision>
    void TestForward( LinearCudaTestData<TPrecision>& data )
    {
        using DeviceTensorType = CudaTensor<TPrecision>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        data.component->build( data.input_shape );

        HostTensorType host_input( Device::Cpu(), data.input_shape );
        random( host_input, -1.0f, 1.0f );

        DeviceTensorType device_input( Device::Cuda( 0 ), data.input_shape );
        DeviceTensorType device_output( Device::Cuda( 0 ), data.output_shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.component->forward( device_input, device_output ) );
        EXPECT_EQ( device_output.size(),
            data.output_shape[ 0 ] * data.output_shape[ 1 ] * data.output_shape[ 2 ] );
        EXPECT_EQ( device_output.shape(), data.output_shape );

        HostTensorType host_output = toHost<TensorDataType::FP32>( device_output );
        EXPECT_EQ( host_output.size(), device_output.size() );
    }

    template<TensorDataType TPrecision>
    void TestGetParameters( const LinearCudaTestData<TPrecision>& data )
    {
        auto params = data.component->getParameters();

        if ( data.has_bias )
        {
            EXPECT_EQ( params.size(), 2 );
            EXPECT_NE( params[ 0 ], nullptr );
            EXPECT_NE( params[ 1 ], nullptr );
        }
        else
        {
            EXPECT_EQ( params.size(), 1 );
            EXPECT_NE( params[ 0 ], nullptr );
        }
    }

    template<TensorDataType TPrecision>
    void TestGetWeightGrad( LinearCudaTestData<TPrecision>& data )
    {
        data.component->setTraining( true );
        data.component->build( data.input_shape );

        auto weight_grad = data.component->getWeightGrad();

        ASSERT_NE( weight_grad, nullptr ) << "Weight gradients should be allocated in training mode";
        EXPECT_EQ( weight_grad->shape()[ 0 ], data.output_features );
        EXPECT_EQ( weight_grad->shape()[ 1 ], data.input_features );
    }

    template<TensorDataType TPrecision>
    void TestGetBiasGrad( LinearCudaTestData<TPrecision>& data )
    {
        data.component->setTraining( true );
        data.component->build( data.input_shape );

        auto bias_grad = data.component->getBiasGrad();

        if ( data.has_bias )
        {
            ASSERT_NE( bias_grad, nullptr ) << "Bias gradients should be allocated in training mode";
            EXPECT_EQ( bias_grad->shape()[ 0 ], data.output_features );
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

        data.component->setTraining( true );
        data.component->build( data.input_shape );

        HostTensorType host_input( Device::Cpu(), data.input_shape );
        HostTensorType host_output_grad( Device::Cpu(), data.output_shape );

        random( host_input, -1.0f, 1.0f );
        random( host_output_grad, -0.1f, 0.1f );

        DeviceTensorType device_input( Device::Cuda( 0 ), data.input_shape );
        DeviceTensorType device_output( Device::Cuda( 0 ), data.output_shape );
        DeviceTensorType device_output_grad( Device::Cuda( 0 ), data.output_shape );
        DeviceTensorType device_input_grad( Device::Cuda( 0 ), data.input_shape );

        copy( host_input, device_input );
        copy( host_output_grad, device_output_grad );
        zeros( device_input_grad );

        data.component->forward( device_input, device_output );

        EXPECT_NO_THROW(
            data.component->backward( device_input, device_output_grad, device_input_grad )
        ) << "Backward pass should succeed for CUDA Linear operation in training mode";

        EXPECT_EQ( device_input_grad.shape(), data.input_shape );

        HostTensorType host_input_grad = toHost<TensorDataType::FP32>( device_input_grad );
        EXPECT_EQ( host_input_grad.size(), device_input_grad.size() );

        bool has_nonzero_grad = false;
        for ( size_t i = 0; i < host_input_grad.size(); ++i )
        {
            if ( std::abs( host_input_grad.data()[ i ] ) > 1e-6f )
            {
                has_nonzero_grad = true;
                break;
            }
        }
        EXPECT_TRUE( has_nonzero_grad ) << "Input gradients should contain non-zero values";
    }

    // ====================================================================
    // Construction Tests
    // ====================================================================

    TEST_F( LinearCudaTests, Construction_WithDeviceId )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        LinearConfig config( 16, 32 );
        config.withBias( true );

        EXPECT_NO_THROW(
            (std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(
                "construct_with_dev",
                config,
                Device::Cuda( 0 )
            ))
        );
    }

    TEST_F( LinearCudaTests, Construction_NoDeviceId_GetDeviceIdThrows )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        LinearConfig config( 16, 32 );
        config.withBias( true );

        // Shared-mode construction (name provided, no DeviceId) is allowed;
        // getDeviceId() should throw until an execution context is set.
        auto comp = std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(
            "no_dev_ctor",
            config
        );

        EXPECT_THROW(
            comp->getDeviceId(),
            std::runtime_error
        );
    }

    TEST_F( LinearCudaTests, Construction_DeviceTypeMismatch_Throws )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        LinearConfig config( 16, 32 );
        config.withBias( true );

        EXPECT_THROW(
            (std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(
                "mismatch_ctor",
                config,
                Device::Cpu()
            )),
            std::invalid_argument
        );
    }

    // ====================================================================
    // Basic Property Tests
    // ====================================================================

    TEST_F( LinearCudaTests, GetName )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetName( SmallFp32Data(), "small_linear_cuda" );
    }

    TEST_F( LinearCudaTests, DeviceType )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestDeviceType( SmallFp32Data() );
    }

    TEST_F( LinearCudaTests, IsBuilt_BeforeBuild )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestIsBuilt( SmallFp32Data(), false );
    }

    TEST_F( LinearCudaTests, IsBuilt_AfterBuild )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto& data = SmallFp32Data();

        EXPECT_FALSE( data.component->isBuilt() );

        data.component->build( data.input_shape );

        EXPECT_TRUE( data.component->isBuilt() );
    }

    TEST_F( LinearCudaTests, Build )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto& data = SmallFp32Data();
        TestBuild( data );
    }

    TEST_F( LinearCudaTests, ParameterCount_WithBias )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestParameterCount( SmallFp32Data() );
    }

    TEST_F( LinearCudaTests, ParameterCount_WithoutBias )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestParameterCount( NoBiasFp32Data() );
    }

    TEST_F( LinearCudaTests, GetWeight )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetWeight( SmallFp32Data() );
    }

    TEST_F( LinearCudaTests, GetBias_WithBias )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetBias( SmallFp32Data() );
    }

    TEST_F( LinearCudaTests, GetBias_WithoutBias )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetBias( NoBiasFp32Data() );
    }

    TEST_F( LinearCudaTests, HasBias_True )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestHasBias( SmallFp32Data() );
    }

    TEST_F( LinearCudaTests, HasBias_False )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestHasBias( NoBiasFp32Data() );
    }

    TEST_F( LinearCudaTests, GetParameters_WithBias )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetParameters( SmallFp32Data() );
    }

    TEST_F( LinearCudaTests, GetParameters_WithoutBias )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetParameters( NoBiasFp32Data() );
    }

    TEST_F( LinearCudaTests, ToString )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        const auto& data = SmallFp32Data();
        TestToString( data );
    }

    // ====================================================================
    // Remaining tests unchanged (forward/backward/synchronization/training)
    // ====================================================================

    TEST_F( LinearCudaTests, Forward_SmallShape )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto& data = SmallFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCudaTests, Forward_MediumShape )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto& data = MediumFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCudaTests, Forward_LargeShape )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto& data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCudaTests, Forward_WithoutBias )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto& data = NoBiasFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCudaTests, WithContext_Construction )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto ctx = createExecutionContext( Device::Cuda( 0 ) );
        auto ctx_id = ctx->getDeviceId();

        auto data = LinearCudaTestData<TensorDataType::FP32>::CreateWithContext(
            "context_linear_cuda", small_shape_, input_features_, output_features_, std::move( ctx ) );

        EXPECT_EQ( data.component->getName(), "context_linear_cuda" );
        EXPECT_EQ( data.component->getDeviceId(), ctx_id );
    }

    TEST_F( LinearCudaTests, EdgeCase_MinimalShape )
    {
        if ( !cuda_available_ )
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
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto& data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCudaTests, EdgeCase_BatchSize1 )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 1, 8, 16 };

        auto data = LinearCudaTestData<TensorDataType::FP32>::Create(
            "batch1_cuda", shape, 16, 32 );

        TestForward( data );
    }

    TEST_F( LinearCudaTests, Error_ForwardBeforeBuild )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LinearCudaTestData<TensorDataType::FP32>::Create(
            "unbuild_cuda", small_shape_, input_features_, output_features_ );

        CudaTensor<TensorDataType::FP32> input( Device::Cuda( 0 ), data.input_shape );
        CudaTensor<TensorDataType::FP32> output( Device::Cuda( 0 ), data.output_shape );

        EXPECT_THROW(
            data.component->forward( input, output ),
            std::runtime_error
        );
    }

    TEST_F( LinearCudaTests, Error_ShapeMismatch )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto& data = SmallFp32Data();
        data.component->build( data.input_shape );

        shape_t wrong_shape = { 2, 3, 64 };

        CudaTensor<TensorDataType::FP32> input( Device::Cuda( 0 ), wrong_shape );
        CudaTensor<TensorDataType::FP32> output( Device::Cuda( 0 ), { 2, 3, 32 } );

        EXPECT_THROW(
            data.component->forward( input, output ),
            std::invalid_argument
        );
    }

    TEST_F( LinearCudaTests, Synchronize )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto& data = SmallFp32Data();

        EXPECT_NO_THROW( data.component->synchronize() );
    }

    TEST_F( LinearCudaTests, SetTrainingMode )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto& data = SmallFp32Data();

        EXPECT_FALSE( data.component->isTraining() );

        data.component->setTraining( true );
        EXPECT_TRUE( data.component->isTraining() );

        data.component->setTraining( false );
        EXPECT_FALSE( data.component->isTraining() );
    }

    TEST_F( LinearCudaTests, MultipleForwardCalls )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto& data = MediumFp32Data();
        data.component->build( data.input_shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), data.input_shape );
        CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), data.input_shape );
        CudaTensor<TensorDataType::FP32> device_output( Device::Cuda( 0 ), data.output_shape );

        for ( int iter = 0; iter < 10; ++iter )
        {
            random( host_input, -1.0f, 1.0f );
            copy( host_input, device_input );

            EXPECT_NO_THROW( data.component->forward( device_input, device_output ) );
        }
    }

    // Backward tests unchanged (omitted here for brevity)...
}