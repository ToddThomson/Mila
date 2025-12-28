#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Dnn::Components::Normalization::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    template<TensorDataType TPrecision>
    struct LayerNormCudaTestData
    {
        shape_t shape;
        shape_t normalized_shape;
        LayerNormConfig config;
        std::shared_ptr<LayerNorm<DeviceType::Cuda, TPrecision>> layer_norm;
        bool is_training;

        static LayerNormCudaTestData Create(
            const std::string& name,
            const shape_t& shape,
            const shape_t& normalized_shape,
            bool has_bias = true,
            float epsilon = 1e-5f,
            bool is_training = false )
        {
            LayerNormCudaTestData data;
            data.shape = shape;
            data.normalized_shape = normalized_shape;
            data.is_training = is_training;

            data.config = LayerNormConfig();
            data.config
                .withNormalizedShape( normalized_shape )
                .withBias( has_bias )
                .withEpsilon( epsilon );

            data.layer_norm = std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>( name, data.config, Device::Cuda(0) );

            return data;
        }

        static LayerNormCudaTestData CreateWithAxis(
            const std::string& name,
            const shape_t& shape,
            int64_t axis,
            bool has_bias = true,
            float epsilon = 1e-5f,
            bool is_training = false )
        {
            LayerNormCudaTestData data;
            data.shape = shape;
            data.is_training = is_training;

            data.config = LayerNormConfig();
            data.config
                .withAxis( axis )
                .withBias( has_bias )
                .withEpsilon( epsilon );

            data.layer_norm = std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>( name, data.config, Device::Cuda(0) );

            return data;
        }

        // Accept a raw IExecutionContext pointer (caller may own the unique_ptr returned
        // by createExecutionContext()). We only use the device id from the provided context
        // to construct the component (component will create/own its own ExecutionContext).
        static LayerNormCudaTestData CreateWithContext(
            const std::string& name,
            const shape_t& shape,
            const shape_t& normalized_shape,
            IExecutionContext* context,
            bool has_bias = true,
            float epsilon = 1e-5f,
            bool is_training = false )
        {
            LayerNormCudaTestData data;
            data.shape = shape;
            data.normalized_shape = normalized_shape;
            data.is_training = is_training;

            data.config = LayerNormConfig();
            data.config
                .withNormalizedShape( normalized_shape )
                .withBias( has_bias )
                .withEpsilon( epsilon );

            data.layer_norm = std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>( name, data.config, context->getDeviceId() );

            return data;
        }
    };

    class LayerNormCudaTests : public ::testing::Test
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

            small_shape_ = { 2, 3, 4 };
            small_normalized_shape_ = { 4 };

            medium_shape_ = { 8, 16, 32 };
            medium_normalized_shape_ = { 32 };

            large_shape_ = { 16, 64, 128 };
            large_normalized_shape_ = { 128 };

            transformer_shape_ = { 32, 128, 768 };
            transformer_normalized_shape_ = { 768 };
        }

        void TearDown() override
        {}

        LayerNormCudaTestData<TensorDataType::FP32>& SmallFp32Data()
        {
            if ( !small_fp32_.layer_norm )
            {
                small_fp32_ = LayerNormCudaTestData<TensorDataType::FP32>::Create(
                    "small_layernorm_cuda", small_shape_, small_normalized_shape_ );
            }
            return small_fp32_;
        }

        LayerNormCudaTestData<TensorDataType::FP32>& MediumFp32Data()
        {
            if ( !medium_fp32_.layer_norm )
            {
                medium_fp32_ = LayerNormCudaTestData<TensorDataType::FP32>::Create(
                    "medium_layernorm_cuda", medium_shape_, medium_normalized_shape_ );
            }

            return medium_fp32_;
        }

        LayerNormCudaTestData<TensorDataType::FP32>& LargeFp32Data()
        {
            if ( !large_fp32_.layer_norm )
            {
                large_fp32_ = LayerNormCudaTestData<TensorDataType::FP32>::Create(
                    "large_layernorm_cuda", large_shape_, large_normalized_shape_ );
            }

            return large_fp32_;
        }

        LayerNormCudaTestData<TensorDataType::FP32>& TrainingFp32Data()
        {
            if ( !training_fp32_.layer_norm )
            {
                training_fp32_ = LayerNormCudaTestData<TensorDataType::FP32>::Create(
                    "training_layernorm_cuda", medium_shape_, medium_normalized_shape_, true, 1e-5f, true );
            }
            return training_fp32_;
        }

        bool cuda_available_{ false };

        shape_t small_shape_;
        shape_t small_normalized_shape_;
        shape_t medium_shape_;
        shape_t medium_normalized_shape_;
        shape_t large_shape_;
        shape_t large_normalized_shape_;
        shape_t transformer_shape_;
        shape_t transformer_normalized_shape_;

        LayerNormCudaTestData<TensorDataType::FP32> small_fp32_;
        LayerNormCudaTestData<TensorDataType::FP32> medium_fp32_;
        LayerNormCudaTestData<TensorDataType::FP32> large_fp32_;
        LayerNormCudaTestData<TensorDataType::FP32> training_fp32_;
    };

    template<TensorDataType TPrecision>
    void TestGetName( const LayerNormCudaTestData<TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.layer_norm->getName(), expected_name );
    }

    template<TensorDataType TPrecision>
    void TestDeviceType( const LayerNormCudaTestData<TPrecision>& data )
    {
        EXPECT_EQ( data.layer_norm->getDeviceType(), DeviceType::Cuda );

        auto device = data.layer_norm->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cuda );
    }

    template<TensorDataType TPrecision>
    void TestTrainingMode( const LayerNormCudaTestData<TPrecision>& data, bool expected_mode )
    {
        EXPECT_EQ( data.layer_norm->isTraining(), expected_mode );
    }

    template<TensorDataType TPrecision>
    void TestIsBuilt( const LayerNormCudaTestData<TPrecision>& data, bool expected_built )
    {
        EXPECT_EQ( data.layer_norm->isBuilt(), expected_built );
    }

    template<TensorDataType TPrecision>
    void TestBuild( LayerNormCudaTestData<TPrecision>& data )
    {
        EXPECT_NO_THROW( data.layer_norm->build( data.shape ) );
        EXPECT_TRUE( data.layer_norm->isBuilt() );
    }

    template<TensorDataType TPrecision>
    void TestParameters( const LayerNormCudaTestData<TPrecision>& data, size_t /*expected_weight_size*/ )
    {
        auto params = data.layer_norm->getParameters();

        if ( data.config.hasBias() )
        {
            EXPECT_EQ( params.size(), 2 );
        }
        else
        {
            EXPECT_EQ( params.size(), 1 );
        }
    }

    template<TensorDataType TPrecision>
    void TestParameterCount( const LayerNormCudaTestData<TPrecision>& data, size_t expected_count )
    {
        EXPECT_EQ( data.layer_norm->parameterCount(), expected_count );
    }

    template<TensorDataType TPrecision>
    void TestToString( const LayerNormCudaTestData<TPrecision>& data )
    {
        std::string output = data.layer_norm->toString();

        EXPECT_NE( output.find( "LayerNorm" ), std::string::npos );
        EXPECT_NE( output.find( data.layer_norm->getName() ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
        EXPECT_NE( output.find( "Epsilon:" ), std::string::npos );
        EXPECT_NE( output.find( "Has Bias:" ), std::string::npos );
    }

    template<TensorDataType TPrecision>
    void TestForward( LayerNormCudaTestData<TPrecision>& data )
    {
        using DeviceTensorType = CudaTensor<TPrecision>;
        using HostTensorType = CpuTensor<TensorDataType::FP32>;

        data.layer_norm->build( data.shape );

        HostTensorType host_input( Device::Cpu(), data.shape );
        random( host_input, -2.0f, 2.0f );

        DeviceTensorType device_input( Device::Cuda( 0 ), data.shape );
        DeviceTensorType device_output( Device::Cuda( 0 ), data.shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.layer_norm->forward( device_input, device_output ) );
        EXPECT_NO_THROW( data.layer_norm->synchronize() );

        EXPECT_EQ( device_output.size(), device_input.size() );
        EXPECT_EQ( device_output.shape(), device_input.shape() );

        HostTensorType host_output = toHost<TensorDataType::FP32>( device_output );
        EXPECT_EQ( host_output.size(), host_input.size() );
    }

    template<TensorDataType TPrecision>
    void ValidateNormalization( const CpuTensor<TensorDataType::FP32>& output, const shape_t& normalized_shape, float epsilon )
    {
        const auto& shape = output.shape();
        size_t norm_dims = normalized_shape.size();

        size_t outer_size = 1;

        for ( size_t i = 0; i + norm_dims < shape.size(); ++i )
        {
            outer_size *= static_cast<size_t>( shape[ i ] );
        }

        size_t norm_size = 1;

        for ( auto dim : normalized_shape )
        {
            norm_size *= static_cast<size_t>( dim );
        }

        auto output_ptr = output.data();

        for ( size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx )
        {
            float sum = 0.0f;
            float sum_sq = 0.0f;

            for ( size_t norm_idx = 0; norm_idx < norm_size; ++norm_idx )
            {
                size_t idx = outer_idx * norm_size + norm_idx;
                float val = static_cast<float>( output_ptr[ idx ] );
                sum += val;
                sum_sq += val * val;
            }

            float mean = sum / norm_size;
            float variance = (sum_sq / norm_size) - (mean * mean);

            EXPECT_NEAR( mean, 0.0f, 0.01f ) << "Mean check failed at outer_idx=" << outer_idx;
            EXPECT_NEAR( variance, 1.0f, 0.1f ) << "Variance check failed at outer_idx=" << outer_idx;
        }
    }

    TEST_F( LayerNormCudaTests, GetName )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "small_layernorm_cuda", medium_shape_, medium_normalized_shape_, false );

        TestGetName( data, "small_layernorm_cuda" );
    }

    TEST_F( LayerNormCudaTests, DeviceType )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestDeviceType( SmallFp32Data() );
    }

    TEST_F( LayerNormCudaTests, TrainingMode_Default )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestTrainingMode( SmallFp32Data(), false );
    }

    // FIXME:
    //TEST_F( LayerNormCudaTests, TrainingMode_Enabled )
    //{
    //    if ( !cuda_available_ )
    //    {
    //        GTEST_SKIP() << "CUDA not available";
    //    }

    //    TestTrainingMode( TrainingFp32Data(), true );
    //}

    TEST_F( LayerNormCudaTests, IsBuilt_BeforeBuild )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestIsBuilt( SmallFp32Data(), false );
    }

    TEST_F( LayerNormCudaTests, IsBuilt_AfterBuild )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_FALSE( data.layer_norm->isBuilt() );

        data.layer_norm->build( data.shape );

        EXPECT_TRUE( data.layer_norm->isBuilt() );
    }

    TEST_F( LayerNormCudaTests, Build )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestBuild( data );
    }

    TEST_F( LayerNormCudaTests, Parameters_WithBias )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.layer_norm->build( data.shape );

        size_t norm_size = 1;

        for ( auto dim : data.normalized_shape )
        {
            norm_size *= dim;
        }

        TestParameters( data, norm_size );
    }

    TEST_F( LayerNormCudaTests, Parameters_WithoutBias )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "no_bias_layernorm_cuda", small_shape_, small_normalized_shape_, false );

        data.layer_norm->build( data.shape );

        size_t norm_size = 1;

        for ( auto dim : data.normalized_shape )
        {
            norm_size *= dim;
        }

        TestParameters( data, norm_size );
    }

    TEST_F( LayerNormCudaTests, ParameterCount_WithBias )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.layer_norm->build( data.shape );

        size_t norm_size = 1;

        for ( auto dim : data.normalized_shape )
        {
            norm_size *= dim;
        }

        TestParameterCount( data, norm_size * 2 );
    }

    TEST_F( LayerNormCudaTests, ParameterCount_WithoutBias )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "no_bias_layernorm_cuda", small_shape_, small_normalized_shape_, false );

        data.layer_norm->build( data.shape );

        size_t norm_size = 1;

        for ( auto dim : data.normalized_shape )
        {
            norm_size *= dim;
        }

        TestParameterCount( data, norm_size );
    }

    TEST_F( LayerNormCudaTests, ToString )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestToString( data );
    }

    TEST_F( LayerNormCudaTests, Forward_SmallShape )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, Forward_MediumShape )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, Forward_LargeShape )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, Forward_No_Bias )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "no_bias_forward_cuda", medium_shape_, medium_normalized_shape_, false );

        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, Forward_DifferentEpsilon )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "custom_epsilon_cuda", medium_shape_, medium_normalized_shape_, true, 1e-3f );

        TestForward( data );
    }

    /*TEST_F( LayerNormCudaTests, Forward_Normalization )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        data.layer_norm->build( data.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), data.shape );
        random( host_input, -5.0f, 5.0f );

        CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), data.shape );
        CudaTensor<TensorDataType::FP32> device_output( Device::Cuda( 0 ), data.shape );

        copy( host_input, device_input );

        data.layer_norm->forward( device_input, device_output );

        CpuTensor<TensorDataType::FP32> host_output = toHost<TensorDataType::FP32>( device_output );

        ValidateNormalization<TensorDataType::FP32>( host_output, data.normalized_shape, data.config.getEpsilon() );
    }*/

    TEST_F( LayerNormCudaTests, Forward_MultipleTrailingDims )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 2, 3, 4, 5 };
        shape_t normalized_shape = { 4, 5 };

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "multi_trailing_cuda", shape, normalized_shape );

        data.layer_norm->build( data.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape );
        random( host_input, -3.0f, 3.0f );

        CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape );
        CudaTensor<TensorDataType::FP32> device_output( Device::Cuda( 0 ), shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.layer_norm->forward( device_input, device_output ) );
        EXPECT_EQ( device_output.size(), device_input.size() );
    }

    TEST_F( LayerNormCudaTests, WithAxis_Construction )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::CreateWithAxis(
            "axis_layernorm_cuda", medium_shape_, -1 );

        EXPECT_EQ( data.layer_norm->getName(), "axis_layernorm_cuda" );
        EXPECT_FALSE( data.layer_norm->isBuilt() );
    }

    TEST_F( LayerNormCudaTests, WithAxis_Build )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::CreateWithAxis(
            "axis_layernorm_cuda", medium_shape_, -1 );

        EXPECT_NO_THROW( data.layer_norm->build( data.shape ) );
        EXPECT_TRUE( data.layer_norm->isBuilt() );
    }

    TEST_F( LayerNormCudaTests, WithAxis_Forward )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::CreateWithAxis(
            "axis_forward_cuda", medium_shape_, -1 );

        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, WithContext_Construction )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        // Use factory to create a context and pass its raw pointer (we only need the DeviceId)
        auto ctx = createExecutionContext( Device::Cuda( 0 ) );

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::CreateWithContext(
            "context_layernorm_cuda", medium_shape_, medium_normalized_shape_, ctx.get() );

        EXPECT_EQ( data.layer_norm->getName(), "context_layernorm_cuda" );
        EXPECT_EQ( data.layer_norm->getDeviceId().type, DeviceType::Cuda );
    }

    TEST_F( LayerNormCudaTests, EdgeCase_MinimalShape )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 1, 1, 2 };
        shape_t normalized_shape = { 2 };

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "minimal_cuda", shape, normalized_shape );

        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, EdgeCase_LargeNormalizedDim )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, EdgeCase_TransformerSize )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "transformer_cuda", transformer_shape_, transformer_normalized_shape_ );

        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, EdgeCase_AllZeros )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.layer_norm->build( data.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), data.shape );
        zeros( host_input );

        CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), data.shape );
        CudaTensor<TensorDataType::FP32> device_output( Device::Cuda( 0 ), data.shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.layer_norm->forward( device_input, device_output ) );
    }

    TEST_F( LayerNormCudaTests, EdgeCase_ConstantValues )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.layer_norm->build( data.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), data.shape );
        fill( host_input, 5.0f );

        CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), data.shape );
        CudaTensor<TensorDataType::FP32> device_output( Device::Cuda( 0 ), data.shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.layer_norm->forward( device_input, device_output ) );
    }

    TEST_F( LayerNormCudaTests, Error_BuildWithoutContext )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        LayerNormConfig config;
        config.withNormalizedShape( { 4 } );

        // Shared-mode construction (no DeviceId) is allowed — but build should fail
        // because no execution context is set on the component.
        auto module = std::make_shared<LayerNorm<DeviceType::Cuda, TensorDataType::FP32>>( "null_context_test_cuda", config, std::nullopt );

        EXPECT_THROW(
            module->build( shape_t{ 2, 1, 4 } ),
            std::runtime_error
        );
    }

    TEST_F( LayerNormCudaTests, Error_InvalidConfig )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        LayerNormConfig invalid_config;
        EXPECT_THROW(
            (std::make_shared<LayerNorm<DeviceType::Cuda, TensorDataType::FP32>>( "ln", invalid_config, Device::Cuda(0))),
            std::invalid_argument
        );
    }

    TEST_F( LayerNormCudaTests, Error_ForwardBeforeBuild_WithAxis )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::CreateWithAxis(
            "unbuild_cuda", medium_shape_, -1 );

        CudaTensor<TensorDataType::FP32> input( Device::Cuda( 0 ), data.shape );
        CudaTensor<TensorDataType::FP32> output( Device::Cuda( 0 ), data.shape );

        EXPECT_THROW(
            data.layer_norm->forward( input, output ),
            std::runtime_error
        );
    }

    TEST_F( LayerNormCudaTests, Error_ShapeMismatch )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.layer_norm->build( data.shape );

        shape_t wrong_shape = { 2, 3, 8 };

        CudaTensor<TensorDataType::FP32> input( Device::Cuda( 0 ), wrong_shape );
        CudaTensor<TensorDataType::FP32> output( Device::Cuda( 0 ), wrong_shape );

        EXPECT_THROW(
            data.layer_norm->forward( input, output ),
            std::invalid_argument
        );
    }

    TEST_F( LayerNormCudaTests, Synchronize )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_NO_THROW( data.layer_norm->synchronize() );
    }

    TEST_F( LayerNormCudaTests, SetTrainingMode )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.layer_norm->build( data.shape );

        EXPECT_FALSE( data.layer_norm->isTraining() );

        data.layer_norm->setTraining( true );
        EXPECT_TRUE( data.layer_norm->isTraining() );

        data.layer_norm->setTraining( false );
        EXPECT_FALSE( data.layer_norm->isTraining() );
    }

    TEST_F( LayerNormCudaTests, MultipleForwardCalls )
    {
        if ( !cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        data.layer_norm->build( data.shape );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), data.shape );
        CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), data.shape );
        CudaTensor<TensorDataType::FP32> device_output( Device::Cuda( 0 ), data.shape );

        for ( int iter = 0; iter < 10; ++iter )
        {
            random( host_input, -2.0f, 2.0f );
            copy( host_input, device_input );

            EXPECT_NO_THROW( data.layer_norm->forward( device_input, device_output ) );
        }
    }

    //TEST_F( LayerNormCudaTests, CpuCuda_OutputEquivalence )
    //{
    //    if ( !cuda_available_ )
    //    {
    //        GTEST_SKIP() << "CUDA not available";
    //    }

    //    shape_t test_shape = { 2, 4, 8 };
    //    shape_t normalized_shape = { 8 };

    //    // Create CPU module directly (standalone)
    //    LayerNormConfig cpu_config;
    //    cpu_config.withNormalizedShape( normalized_shape )
    //        .withBias( true )
    //        .withEpsilon( 1e-5f );

    //    auto cpu_module = std::make_shared<LayerNorm<DeviceType::Cpu, TensorDataType::FP32>>( "cpu_ln", cpu_config, Device::Cpu());

    //    // Create CUDA module
    //    auto cuda_data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
    //        "cuda_equiv", test_shape, normalized_shape );

    //    // Build both modules
    //    cpu_module->build( test_shape );
    //    cuda_data.layer_norm->build( test_shape );

    //    // Create and initialize input
    //    CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), test_shape );
    //    random( host_input, -2.0f, 2.0f );

    //    // Run CPU forward pass
    //    CpuTensor<TensorDataType::FP32> cpu_output( Device::Cpu(), test_shape );
    //    cpu_module->forward( host_input, cpu_output );

    //    // Run CUDA forward pass
    //    CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), test_shape );
    //    CudaTensor<TensorDataType::FP32> device_output( Device::Cuda( 0 ), test_shape );
    //    copy( host_input, device_input );
    //    cuda_data.layer_norm->forward( device_input, device_output );

    //    // Copy CUDA output back to host for comparison
    //    CpuTensor<TensorDataType::FP32> cuda_output_host = toHost<TensorDataType::FP32>( device_output );

    //    // Compare outputs
    //    const float epsilon = 1e-4f;
    //    bool all_close = true;

    //    for ( size_t i = 0; i < cpu_output.size(); ++i )
    //    {
    //        float cpu_val = cpu_output.data()[ i ];
    //        float cuda_val = cuda_output_host.data()[ i ];
    //        float diff = std::abs( cpu_val - cuda_val );

    //        if ( diff > epsilon )
    //        {
    //            all_close = false;
    //            break;
    //        }
    //    }

    //    EXPECT_TRUE( all_close ) << "CPU and CUDA implementations produced different results";
    //}
}