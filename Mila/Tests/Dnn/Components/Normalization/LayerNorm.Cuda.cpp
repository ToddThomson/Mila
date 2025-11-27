#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cstdint>

import Mila;

namespace Modules::Normalization::Tests
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
        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> exec_context;
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
            data.config.withName( name )
                .withNormalizedShape( normalized_shape )
                .withBias( has_bias )
                .withEpsilon( epsilon );

            data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            data.layer_norm = std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>( data.exec_context, data.config );

			data.layer_norm->setTraining( is_training );

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
            data.config.withName( name )
                .withAxis( axis )
                .withBias( has_bias )
                .withEpsilon( epsilon );

            data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            data.layer_norm = std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>( data.exec_context, data.config );

			data.layer_norm->setTraining( is_training );

            return data;
        }

        static LayerNormCudaTestData CreateWithContext(
            const std::string& name,
            const shape_t& shape,
            const shape_t& normalized_shape,
            std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context,
            bool has_bias = true,
            float epsilon = 1e-5f,
            bool is_training = false )
        {
            LayerNormCudaTestData data;
            data.shape = shape;
            data.normalized_shape = normalized_shape;
            data.is_training = is_training;

            data.config = LayerNormConfig();
            data.config.withName( name )
                .withNormalizedShape( normalized_shape )
                .withBias( has_bias )
                .withEpsilon( epsilon );

            data.exec_context = context;
            data.layer_norm = std::make_shared<LayerNorm<DeviceType::Cuda, TPrecision>>( data.exec_context, data.config );

			data.layer_norm->setTraining( is_training );

            return data;
        }
    };

    class LayerNormCudaTests : public ::testing::Test
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
            small_normalized_shape_ = { 4 };

            medium_shape_ = { 8, 16, 32 };
            medium_normalized_shape_ = { 32 };

            large_shape_ = { 16, 64, 128 };
            large_normalized_shape_ = { 128 };

            transformer_shape_ = { 32, 128, 768 };
            transformer_normalized_shape_ = { 768 };
        }

        void TearDown() override
        {
			// Tensor RAII handles cleanup
        }

        LayerNormCudaTestData<TensorDataType::FP32>& SmallFp32Data()
        {
            if (!small_fp32_.layer_norm)
            {
                small_fp32_ = LayerNormCudaTestData<TensorDataType::FP32>::Create(
                    "small_layernorm_cuda", small_shape_, small_normalized_shape_ );
            }
            return small_fp32_;
        }

        LayerNormCudaTestData<TensorDataType::FP32>& MediumFp32Data()
        {
            if (!medium_fp32_.layer_norm)
            {
                medium_fp32_ = LayerNormCudaTestData<TensorDataType::FP32>::Create(
                    "medium_layernorm_cuda", medium_shape_, medium_normalized_shape_ );
            }
            
            return medium_fp32_;
        }

        LayerNormCudaTestData<TensorDataType::FP32>& LargeFp32Data()
        {
            if (!large_fp32_.layer_norm)
            {
                large_fp32_ = LayerNormCudaTestData<TensorDataType::FP32>::Create(
                    "large_layernorm_cuda", large_shape_, large_normalized_shape_ );
            }
            
            return large_fp32_;
        }

        LayerNormCudaTestData<TensorDataType::FP32>& TrainingFp32Data()
        {
            if (!training_fp32_.layer_norm)
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
        ASSERT_NE( data.exec_context, nullptr );

        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
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
    void TestParameters( const LayerNormCudaTestData<TPrecision>& data, size_t expected_weight_size )
    {
        /*auto weight = data.layer_norm->getWeight();
        ASSERT_NE( weight, nullptr );
        EXPECT_EQ( weight->size(), expected_weight_size );

        auto bias = data.layer_norm->getBias();

        if (data.config.hasBias())
        {
            ASSERT_NE( bias, nullptr );
            EXPECT_EQ( bias->size(), expected_weight_size );
        }
        else
        {
            EXPECT_EQ( bias, nullptr );
        }*/

        auto params = data.layer_norm->getParameters();

        if (data.config.hasBias())
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
        EXPECT_NE( output.find( data.config.getName() ), std::string::npos );
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

        HostTensorType host_input( "CPU", data.shape );
        random( host_input, -2.0f, 2.0f );

        DeviceTensorType device_input( "CUDA:0", data.shape );
        DeviceTensorType device_output( "CUDA:0", data.shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.layer_norm->forward( device_input, device_output ) );
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

        for (size_t i = 0; i + norm_dims < shape.size(); ++i)
        {
            outer_size *= static_cast<size_t>( shape[i] );
        }

        size_t norm_size = 1;

        for (auto dim : normalized_shape)
        {
            norm_size *= static_cast<size_t>( dim );
        }

        auto output_ptr = output.data();

        for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx)
        {
            float sum = 0.0f;
            float sum_sq = 0.0f;

            for (size_t norm_idx = 0; norm_idx < norm_size; ++norm_idx)
            {
                size_t idx = outer_idx * norm_size + norm_idx;
                float val = static_cast<float>( output_ptr[idx] );
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
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "small_layernorm_cuda", medium_shape_, medium_normalized_shape_, false );

        TestGetName( data, "small_layernorm_cuda" );
    }

    TEST_F( LayerNormCudaTests, DeviceType )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestDeviceType( SmallFp32Data() );
    }

    TEST_F( LayerNormCudaTests, TrainingMode_Default )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestTrainingMode( SmallFp32Data(), false );
    }

    TEST_F( LayerNormCudaTests, TrainingMode_Enabled )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestTrainingMode( TrainingFp32Data(), true );
    }

    TEST_F( LayerNormCudaTests, IsBuilt_BeforeBuild )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestIsBuilt( SmallFp32Data(), false );
    }

    TEST_F( LayerNormCudaTests, IsBuilt_AfterBuild )
    {
        if (!cuda_available_)
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
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestBuild( data );
    }

    TEST_F( LayerNormCudaTests, Parameters_WithBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.layer_norm->build( data.shape );

        size_t norm_size = 1;

        for (auto dim : data.normalized_shape)
        {
            norm_size *= dim;
        }

        TestParameters( data, norm_size );
    }

    TEST_F( LayerNormCudaTests, Parameters_WithoutBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "no_bias_layernorm_cuda", small_shape_, small_normalized_shape_, false );

        data.layer_norm->build( data.shape );

        size_t norm_size = 1;

        for (auto dim : data.normalized_shape)
        {
            norm_size *= dim;
        }

        TestParameters( data, norm_size );
    }

    TEST_F( LayerNormCudaTests, ParameterCount_WithBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.layer_norm->build( data.shape );

        size_t norm_size = 1;

        for (auto dim : data.normalized_shape)
        {
            norm_size *= dim;
        }

        TestParameterCount( data, norm_size * 2 );
    }

    TEST_F( LayerNormCudaTests, ParameterCount_WithoutBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "no_bias_layernorm_cuda", small_shape_, small_normalized_shape_, false );

        data.layer_norm->build( data.shape );

        size_t norm_size = 1;

        for (auto dim : data.normalized_shape)
        {
            norm_size *= dim;
        }

        TestParameterCount( data, norm_size );
    }

    TEST_F( LayerNormCudaTests, ToString )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestToString( data );
    }

    TEST_F( LayerNormCudaTests, Forward_SmallShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

		// 1. Use cached test data
        auto data = SmallFp32Data();


		// 2. Alternatively, create new test data
        /*auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "small_layernorm_cuda", small_shape_, small_normalized_shape_, false );*/

        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, Forward_MediumShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, Forward_LargeShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, Forward_WithoutBias )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "no_bias_forward_cuda", medium_shape_, medium_normalized_shape_, false );

        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, Forward_DifferentEpsilon )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "custom_epsilon_cuda", medium_shape_, medium_normalized_shape_, true, 1e-3f );

        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, Forward_Normalization )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        data.layer_norm->build( data.shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.shape );
        random( host_input, -5.0f, 5.0f );

        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", data.shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.shape );

        copy( host_input, device_input );

        //auto weight = data.layer_norm->getWeight();
        //auto bias = data.layer_norm->getBias();

        //CpuTensor<TensorDataType::FP32> host_weight( "CPU", weight->shape() );
        //CpuTensor<TensorDataType::FP32> host_bias( "CPU", bias->shape() );

        //ones( host_weight );
        //zeros( host_bias );

        //copy( host_weight, *weight );
        //copy( host_bias, *bias );

        data.layer_norm->forward( device_input, device_output );

        CpuTensor<TensorDataType::FP32> host_output = toHost<TensorDataType::FP32>( device_output );

        ValidateNormalization<TensorDataType::FP32>( host_output, data.normalized_shape, data.config.getEpsilon() );
    }

    TEST_F( LayerNormCudaTests, Forward_MultipleTrailingDims )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 2, 3, 4, 5 };
        shape_t normalized_shape = { 4, 5 };

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "multi_trailing_cuda", shape, normalized_shape );

        data.layer_norm->build( data.shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", shape );
        random( host_input, -3.0f, 3.0f );

        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.layer_norm->forward( device_input, device_output ) );
        EXPECT_EQ( device_output.size(), device_input.size() );
    }

    TEST_F( LayerNormCudaTests, WithAxis_Construction )
    {
        if (!cuda_available_)
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
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::CreateWithAxis(
            "axis_layernorm_cuda", medium_shape_, -1 );

        EXPECT_NO_THROW( data.layer_norm->build( data.shape ) );
        EXPECT_TRUE( data.layer_norm->isBuilt() );

        //auto weight = data.layer_norm->getWeight();

        //ASSERT_NE( weight, nullptr );
        //EXPECT_EQ( weight->size(), static_cast<size_t>(data.shape.back()) );
    }

    TEST_F( LayerNormCudaTests, WithAxis_Forward )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::CreateWithAxis(
            "axis_forward_cuda", medium_shape_, -1 );

        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, WithContext_Construction )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::CreateWithContext(
            "context_layernorm_cuda", medium_shape_, medium_normalized_shape_, ctx );

        EXPECT_EQ( data.layer_norm->getName(), "context_layernorm_cuda" );
        EXPECT_EQ( data.exec_context, ctx );
    }

    TEST_F( LayerNormCudaTests, EdgeCase_MinimalShape )
    {
        if (!cuda_available_)
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
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, EdgeCase_TransformerSize )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "transformer_cuda", transformer_shape_, transformer_normalized_shape_ );

        TestForward( data );
    }

    TEST_F( LayerNormCudaTests, EdgeCase_AllZeros )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.layer_norm->build( data.shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.shape );
        zeros( host_input );

        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", data.shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.layer_norm->forward( device_input, device_output ) );
    }

    TEST_F( LayerNormCudaTests, EdgeCase_ConstantValues )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.layer_norm->build( data.shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.shape );
        fill( host_input, 5.0f );

        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", data.shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.shape );

        copy( host_input, device_input );

        EXPECT_NO_THROW( data.layer_norm->forward( device_input, device_output ) );
    }

    TEST_F( LayerNormCudaTests, Error_NullExecutionContext )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        LayerNormConfig config;
        config.withName( "test_cuda" ).withNormalizedShape( { 4 } );

        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> null_ctx;

        EXPECT_THROW(
            (std::make_shared<LayerNorm<DeviceType::Cuda, TensorDataType::FP32>>( null_ctx, config )),
            std::invalid_argument
        );
    }

    TEST_F( LayerNormCudaTests, Error_InvalidConfig )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        LayerNormConfig invalid_config;
        invalid_config.withName( "invalid_cuda" );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        EXPECT_THROW(
            (std::make_shared<LayerNorm<DeviceType::Cuda, TensorDataType::FP32>>( ctx, invalid_config )),
            std::invalid_argument
        );
    }

    TEST_F( LayerNormCudaTests, Error_ForwardBeforeBuild_WithAxis )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LayerNormCudaTestData<TensorDataType::FP32>::CreateWithAxis(
            "unbuild_cuda", medium_shape_, -1 );

        CudaTensor<TensorDataType::FP32> input( "CUDA:0", data.shape );
        CudaTensor<TensorDataType::FP32> output( "CUDA:0", data.shape );

        EXPECT_THROW(
            data.layer_norm->forward( input, output ),
            std::runtime_error
        );
    }

    TEST_F( LayerNormCudaTests, Error_ShapeMismatch )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        data.layer_norm->build( data.shape );

        shape_t wrong_shape = { 2, 3, 8 };

        CudaTensor<TensorDataType::FP32> input( "CUDA:0", wrong_shape );
        CudaTensor<TensorDataType::FP32> output( "CUDA:0", wrong_shape );

        EXPECT_THROW(
            data.layer_norm->forward( input, output ),
            std::invalid_argument
        );
    }

    TEST_F( LayerNormCudaTests, Synchronize )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_NO_THROW( data.layer_norm->synchronize() );
    }

    TEST_F( LayerNormCudaTests, SetTrainingMode )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_FALSE( data.layer_norm->isTraining() );

        data.layer_norm->setTraining( true );
        EXPECT_TRUE( data.layer_norm->isTraining() );

        data.layer_norm->setTraining( false );
        EXPECT_FALSE( data.layer_norm->isTraining() );
    }

    TEST_F( LayerNormCudaTests, MultipleForwardCalls )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        data.layer_norm->build( data.shape );

        CpuTensor<TensorDataType::FP32> host_input( "CPU", data.shape );
        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", data.shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.shape );

        for (int iter = 0; iter < 10; ++iter)
        {
            random( host_input, -2.0f, 2.0f );
            copy( host_input, device_input );

            EXPECT_NO_THROW( data.layer_norm->forward( device_input, device_output ) );
        }
    }

    TEST_F( LayerNormCudaTests, CpuCuda_OutputEquivalence )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t test_shape = { 2, 4, 8 };
        shape_t normalized_shape = { 8 };

        // Create CPU module directly
        LayerNormConfig cpu_config;
        cpu_config.withName( "cpu_equiv" )
            .withNormalizedShape( normalized_shape )
            .withBias( true )
            .withEpsilon( 1e-5f );

        auto cpu_exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cpu_module = std::make_shared<LayerNorm<DeviceType::Cpu, TensorDataType::FP32>>( cpu_exec_context, cpu_config );

        // Create CUDA module
        auto cuda_data = LayerNormCudaTestData<TensorDataType::FP32>::Create(
            "cuda_equiv", test_shape, normalized_shape );

        // Build both modules
        cpu_module->build( test_shape );
        cuda_data.layer_norm->build( test_shape );

        // Create and initialize input
        CpuTensor<TensorDataType::FP32> host_input( "CPU", test_shape );
        random( host_input, -2.0f, 2.0f );

        // Initialize parameters with same values for both CPU and CUDA
        //CpuTensor<TensorDataType::FP32> cpu_weight( "CPU", normalized_shape );
        //CpuTensor<TensorDataType::FP32> cpu_bias( "CPU", normalized_shape );
        //ones( cpu_weight );
        //zeros( cpu_bias );

        //// Copy parameters to CPU module
        //copy( cpu_weight, *cpu_module->getWeight() );
        //copy( cpu_bias, *cpu_module->getBias() );

        //// Copy parameters to CUDA module
        //copy( cpu_weight, *cuda_data.layer_norm->getWeight() );
        //copy( cpu_bias, *cuda_data.layer_norm->getBias() );

        // Run CPU forward pass
        CpuTensor<TensorDataType::FP32> cpu_output( "CPU", test_shape );
        cpu_module->forward( host_input, cpu_output );

        // Run CUDA forward pass
        CudaTensor<TensorDataType::FP32> device_input( "CUDA:0", test_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", test_shape );
        copy( host_input, device_input );
        cuda_data.layer_norm->forward( device_input, device_output );

        // Copy CUDA output back to host for comparison
        CpuTensor<TensorDataType::FP32> cuda_output_host = toHost<TensorDataType::FP32>( device_output );

        // Compare outputs
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