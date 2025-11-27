#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <stdexcept>

import Mila;

namespace Modules::Normalization::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    template<TensorDataType TPrecision>
    struct LayerNormCpuTestData
    {
        shape_t shape;
        shape_t normalized_shape;
        LayerNormConfig config;
        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_context;
        std::shared_ptr<LayerNorm<DeviceType::Cpu, TPrecision>> module;
        bool is_training;

        static LayerNormCpuTestData Create(
            const std::string& name,
            const shape_t& shape,
            const shape_t& normalized_shape,
            bool has_bias = true,
            float epsilon = 1e-5f,
            bool is_training = false )
        {
            LayerNormCpuTestData data;
            data.shape = shape;
            data.normalized_shape = normalized_shape;
            data.is_training = is_training;

            data.config = LayerNormConfig();
            data.config.withName( name )
                .withNormalizedShape( normalized_shape )
                .withBias( has_bias )
                .withEpsilon( epsilon );

            data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            data.module = std::make_shared<LayerNorm<DeviceType::Cpu, TPrecision>>( data.exec_context, data.config );

            // Training is no longer a construction-time configuration.
            // If the test requested training mode, set it explicitly.
            if (data.is_training)
            {
                data.module->setTraining( true );
            }

            return data;
        }

        static LayerNormCpuTestData CreateWithAxis(
            const std::string& name,
            const shape_t& shape,
            int64_t axis,
            bool has_bias = true,
            float epsilon = 1e-5f,
            bool is_training = false )
        {
            LayerNormCpuTestData data;
            data.shape = shape;
            data.is_training = is_training;

            data.config = LayerNormConfig();
            data.config.withName( name )
                .withAxis( axis )
                .withBias( has_bias )
                .withEpsilon( epsilon );

            data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            data.module = std::make_shared<LayerNorm<DeviceType::Cpu, TPrecision>>( data.exec_context, data.config );

            // Training must be set explicitly via setTraining()
            if (data.is_training)
            {
                data.module->setTraining( true );
            }

            return data;
        }

        static LayerNormCpuTestData CreateWithContext(
            const std::string& name,
            const shape_t& shape,
            const shape_t& normalized_shape,
            std::shared_ptr<ExecutionContext<DeviceType::Cpu>> context,
            bool has_bias = true,
            float epsilon = 1e-5f,
            bool is_training = false )
        {
            LayerNormCpuTestData data;
            data.shape = shape;
            data.normalized_shape = normalized_shape;
            data.is_training = is_training;

            data.config = LayerNormConfig();
            data.config.withName( name )
                .withNormalizedShape( normalized_shape )
                .withBias( has_bias )
                .withEpsilon( epsilon );

            data.exec_context = context;
            data.module = std::make_shared<LayerNorm<DeviceType::Cpu, TPrecision>>( data.exec_context, data.config );

            // Respect requested training flag by calling setTraining()
            if (data.is_training)
            {
                data.module->setTraining( true );
            }

            return data;
        }
    };

    class LayerNormCpuTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            small_shape_ = { 2, 3, 4 };
            small_normalized_shape_ = { 4 };

            medium_shape_ = { 8, 16, 32 };
            medium_normalized_shape_ = { 32 };

            large_shape_ = { 16, 64, 128 };
            large_normalized_shape_ = { 128 };

            transformer_shape_ = { 32, 128, 768 };
            transformer_normalized_shape_ = { 768 };
        }

        LayerNormCpuTestData<TensorDataType::FP32>& SmallFp32Data()
        {
            if (!small_fp32_.module)
            {
                small_fp32_ = LayerNormCpuTestData<TensorDataType::FP32>::Create(
                    "small_layernorm", small_shape_, small_normalized_shape_ );
            }

            return small_fp32_;
        }

        LayerNormCpuTestData<TensorDataType::FP32>& MediumFp32Data()
        {
            if (!medium_fp32_.module)
            {
                medium_fp32_ = LayerNormCpuTestData<TensorDataType::FP32>::Create(
                    "medium_layernorm", medium_shape_, medium_normalized_shape_ );
            }
            return medium_fp32_;
        }

        LayerNormCpuTestData<TensorDataType::FP32>& LargeFp32Data()
        {
            if (!large_fp32_.module)
            {
                large_fp32_ = LayerNormCpuTestData<TensorDataType::FP32>::Create(
                    "large_layernorm", large_shape_, large_normalized_shape_ );
            }
            return large_fp32_;
        }

        LayerNormCpuTestData<TensorDataType::FP32>& TrainingFp32Data()
        {
            if (!training_fp32_.module)
            {
                training_fp32_ = LayerNormCpuTestData<TensorDataType::FP32>::Create(
                    "training_layernorm", medium_shape_, medium_normalized_shape_, true, 1e-5f, true );
            }
            return training_fp32_;
        }

        shape_t small_shape_;
        shape_t small_normalized_shape_;
        shape_t medium_shape_;
        shape_t medium_normalized_shape_;
        shape_t large_shape_;
        shape_t large_normalized_shape_;
        shape_t transformer_shape_;
        shape_t transformer_normalized_shape_;

        LayerNormCpuTestData<TensorDataType::FP32> small_fp32_;
        LayerNormCpuTestData<TensorDataType::FP32> medium_fp32_;
        LayerNormCpuTestData<TensorDataType::FP32> large_fp32_;
        LayerNormCpuTestData<TensorDataType::FP32> training_fp32_;
    };

    template<TensorDataType TPrecision>
    void TestGetName( const LayerNormCpuTestData<TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.module->getName(), expected_name );
    }

    template<TensorDataType TPrecision>
    void TestDeviceType( const LayerNormCpuTestData<TPrecision>& data )
    {
        EXPECT_EQ( data.module->getDeviceType(), DeviceType::Cpu );
        ASSERT_NE( data.exec_context, nullptr );

        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
    }

    template<TensorDataType TPrecision>
    void TestTrainingMode( const LayerNormCpuTestData<TPrecision>& data, bool expected_mode )
    {
        EXPECT_EQ( data.module->isTraining(), expected_mode );
    }

    template<TensorDataType TPrecision>
    void TestIsBuilt( const LayerNormCpuTestData<TPrecision>& data, bool expected_built )
    {
        EXPECT_EQ( data.module->isBuilt(), expected_built );
    }

    template<TensorDataType TPrecision>
    void TestBuild( LayerNormCpuTestData<TPrecision>& data )
    {
        EXPECT_NO_THROW( data.module->build( data.shape ) );
        EXPECT_TRUE( data.module->isBuilt() );

        EXPECT_THROW( data.module->build( data.shape ), std::logic_error );
    }

    template<TensorDataType TPrecision>
    void TestParameters( const LayerNormCpuTestData<TPrecision>& data, size_t expected_weight_size )
    {
        auto params = data.module->getParameters();

        ASSERT_FALSE( params.empty() );
        EXPECT_EQ( static_cast<size_t>(static_cast<CpuTensor<TPrecision>*>(params[0])->size()), expected_weight_size );

        if (data.config.hasBias())
        {
            ASSERT_EQ( params.size(), 2 );
            EXPECT_EQ( static_cast<size_t>(static_cast<CpuTensor<TPrecision>*>(params[1])->size()), expected_weight_size );
        }
        else
        {
            EXPECT_EQ( params.size(), 1 );
        }
    }

    template<TensorDataType TPrecision>
    void TestParameterCount( const LayerNormCpuTestData<TPrecision>& data, size_t expected_count )
    {
        EXPECT_EQ( data.module->parameterCount(), expected_count );
    }

    template<TensorDataType TPrecision>
    void TestToString( const LayerNormCpuTestData<TPrecision>& data )
    {
        std::string output = data.module->toString();

        EXPECT_NE( output.find( "LayerNorm" ), std::string::npos );
        EXPECT_NE( output.find( data.config.getName() ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
        EXPECT_NE( output.find( "Epsilon:" ), std::string::npos );
        EXPECT_NE( output.find( "Has Bias:" ), std::string::npos );
    }

    template<TensorDataType TPrecision>
    void TestForward( LayerNormCpuTestData<TPrecision>& data )
    {
        using TensorType = CpuTensor<TPrecision>;

        data.module->build( data.shape );

        TensorType input( "CPU", data.shape );
        TensorType output( "CPU", data.shape );

        random( input, -2.0f, 2.0f );

        EXPECT_NO_THROW( data.module->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    template<TensorDataType TPrecision>
    void ValidateNormalization( const CpuTensor<TPrecision>& output, const shape_t& normalized_shape, float epsilon )
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

    TEST_F( LayerNormCpuTests, GetName )
    {
        TestGetName( SmallFp32Data(), "small_layernorm" );
    }

    TEST_F( LayerNormCpuTests, DeviceType )
    {
        TestDeviceType( SmallFp32Data() );
    }

    TEST_F( LayerNormCpuTests, TrainingMode_Default )
    {
        TestTrainingMode( SmallFp32Data(), false );
    }

    TEST_F( LayerNormCpuTests, TrainingMode_Enabled )
    {
        TestTrainingMode( TrainingFp32Data(), true );
    }

    TEST_F( LayerNormCpuTests, IsBuilt_BeforeBuild )
    {
        // Module with normalized_shape (eager parameter creation)
        auto data = SmallFp32Data();
        EXPECT_FALSE( data.module->isBuilt() );

        // Parameters should exist (created in constructor)
        /*EXPECT_NE( data.module->getWeight(), nullptr );
        EXPECT_NE( data.module->getBias(), nullptr );*/
    }

    TEST_F( LayerNormCpuTests, IsBuilt_AfterBuild )
    {
        auto data = SmallFp32Data();

        // Before build
        EXPECT_FALSE( data.module->isBuilt() );

        // After build
        data.module->build( data.shape );
        EXPECT_TRUE( data.module->isBuilt() );
    }

    TEST_F( LayerNormCpuTests, IsBuilt_WithAxis_BeforeBuild )
    {
        // Module with axis (lazy parameter creation)
        auto data = LayerNormCpuTestData<TensorDataType::FP32>::CreateWithAxis(
            "axis_test", medium_shape_, -1 );

        EXPECT_FALSE( data.module->isBuilt() );

        // Parameters should NOT exist yet (lazy creation)
        auto params_before = data.module->getParameters();
        EXPECT_TRUE( params_before.empty() );
    }

    TEST_F( LayerNormCpuTests, IsBuilt_WithAxis_AfterBuild )
    {
        auto data = LayerNormCpuTestData<TensorDataType::FP32>::CreateWithAxis(
            "axis_test", medium_shape_, -1 );

        // Build creates parameters
        data.module->build( data.shape );

        EXPECT_TRUE( data.module->isBuilt() );
        auto params = data.module->getParameters();
        ASSERT_FALSE( params.empty() );
        EXPECT_EQ( static_cast<size_t>(static_cast<CpuTensor<TensorDataType::FP32>*>(params[0])->size()),
            static_cast<size_t>(data.shape.back()) );
    }

    TEST_F( LayerNormCpuTests, Build )
    {
        auto data = SmallFp32Data();
        TestBuild( data );
    }

    TEST_F( LayerNormCpuTests, Parameters_WithBias )
    {
        auto data = SmallFp32Data();
        data.module->build( data.shape );

        size_t norm_size = 1;

        for (auto dim : data.normalized_shape)
        {
            norm_size *= dim;
        }

        TestParameters( data, norm_size );
    }

    TEST_F( LayerNormCpuTests, Parameters_WithoutBias )
    {
        auto data = LayerNormCpuTestData<TensorDataType::FP32>::Create(
            "no_bias_layernorm", small_shape_, small_normalized_shape_, false );

        data.module->build( data.shape );

        size_t norm_size = 1;

        for (auto dim : data.normalized_shape)
        {
            norm_size *= dim;
        }

        TestParameters( data, norm_size );
    }

    TEST_F( LayerNormCpuTests, ParameterCount_WithBias )
    {
        auto data = SmallFp32Data();
        data.module->build( data.shape );

        size_t norm_size = 1;

        for (auto dim : data.normalized_shape)
        {
            norm_size *= dim;
        }

        TestParameterCount( data, norm_size * 2 );
    }

    TEST_F( LayerNormCpuTests, ParameterCount_WithoutBias )
    {
        auto data = LayerNormCpuTestData<TensorDataType::FP32>::Create(
            "no_bias_layernorm", small_shape_, small_normalized_shape_, false );

        data.module->build( data.shape );

        size_t norm_size = 1;

        for (auto dim : data.normalized_shape)
        {
            norm_size *= dim;
        }

        TestParameterCount( data, norm_size );
    }

    TEST_F( LayerNormCpuTests, ToString )
    {
        auto data = SmallFp32Data();
        TestToString( data );
    }

    TEST_F( LayerNormCpuTests, Forward_SmallShape )
    {
        auto data = SmallFp32Data();
        TestForward( data );
    }

    TEST_F( LayerNormCpuTests, Forward_MediumShape )
    {
        auto data = MediumFp32Data();
        TestForward( data );
    }

    TEST_F( LayerNormCpuTests, Forward_WithoutBias )
    {
        auto data = LayerNormCpuTestData<TensorDataType::FP32>::Create(
            "no_bias_forward", medium_shape_, medium_normalized_shape_, false );

        TestForward( data );
    }

    TEST_F( LayerNormCpuTests, Forward_DifferentEpsilon )
    {
        auto data = LayerNormCpuTestData<TensorDataType::FP32>::Create(
            "custom_epsilon", medium_shape_, medium_normalized_shape_, true, 1e-3f );

        TestForward( data );
    }

    TEST_F( LayerNormCpuTests, Forward_Normalization )
    {
        auto data = MediumFp32Data();
        data.module->build( data.shape );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.shape );

        std::mt19937 rng( 123 );
        std::uniform_real_distribution<float> dist( -5.0f, 5.0f );

        auto input_ptr = input.data();

        for (size_t i = 0; i < input.size(); ++i)
        {
            input_ptr[i] = dist( rng );
        }

        // Use getParameters() to access weight/bias; order is weight then bias (if present)
        auto params = data.module->getParameters();
        ASSERT_FALSE( params.empty() );

        auto weight_ptr = static_cast<CpuTensor<TensorDataType::FP32>*>( params[0] )->data();

        if (data.config.hasBias())
        {
            auto bias_ptr = static_cast<CpuTensor<TensorDataType::FP32>*>(params[1])->data();

            for (size_t i = 0; i < static_cast<size_t>( static_cast<CpuTensor<TensorDataType::FP32>*>( params[0] )->size() ); ++i)
            {
                weight_ptr[i] = 1.0f;
                bias_ptr[i] = 0.0f;
            }
        }
        else
        {
            for (size_t i = 0; i < static_cast<size_t>( static_cast<CpuTensor<TensorDataType::FP32>*>( params[0] )->size() ); ++i)
            {
                weight_ptr[i] = 1.0f;
            }
        }

        data.module->forward( input, output );

        ValidateNormalization<TensorDataType::FP32>( output, data.normalized_shape, data.config.getEpsilon() );
    }

    TEST_F( LayerNormCpuTests, Forward_MultipleTrailingDims )
    {
        shape_t shape = { 2, 3, 4, 5 };
        shape_t normalized_shape = { 4, 5 };

        auto data = LayerNormCpuTestData<TensorDataType::FP32>::Create(
            "multi_trailing", shape, normalized_shape );

        data.module->build( data.shape );

        CpuTensor<TensorDataType::FP32> input( "CPU", shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", shape );

        std::mt19937 rng( 456 );
        std::uniform_real_distribution<float> dist( -3.0f, 3.0f );

        auto input_ptr = input.data();

        for (size_t i = 0; i < input.size(); ++i)
        {
            input_ptr[i] = dist( rng );
        }

        EXPECT_NO_THROW( data.module->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( LayerNormCpuTests, WithAxis_Construction )
    {
        auto data = LayerNormCpuTestData<TensorDataType::FP32>::CreateWithAxis(
            "axis_layernorm", medium_shape_, -1 );

        EXPECT_EQ( data.module->getName(), "axis_layernorm" );
        EXPECT_FALSE( data.module->isBuilt() );
    }

    TEST_F( LayerNormCpuTests, WithAxis_Build )
    {
        auto data = LayerNormCpuTestData<TensorDataType::FP32>::CreateWithAxis(
            "axis_layernorm", medium_shape_, -1 );

        EXPECT_NO_THROW( data.module->build( data.shape ) );
        EXPECT_TRUE( data.module->isBuilt() );

        auto params = data.module->getParameters();
        ASSERT_FALSE( params.empty() );
        EXPECT_EQ( static_cast<size_t>(static_cast<CpuTensor<TensorDataType::FP32>*>(params[0])->size()),
            static_cast<size_t>(data.shape.back()) );
    }

    TEST_F( LayerNormCpuTests, WithAxis_Forward )
    {
        auto data = LayerNormCpuTestData<TensorDataType::FP32>::CreateWithAxis(
            "axis_forward", medium_shape_, -1 );

        TestForward( data );
    }

    TEST_F( LayerNormCpuTests, WithContext_Construction )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        auto data = LayerNormCpuTestData<TensorDataType::FP32>::CreateWithContext(
            "context_layernorm", medium_shape_, medium_normalized_shape_, ctx );

        EXPECT_EQ( data.module->getName(), "context_layernorm" );
        EXPECT_EQ( data.exec_context, ctx );
    }

    TEST_F( LayerNormCpuTests, EdgeCase_MinimalShape )
    {
        shape_t shape = { 1, 1, 2 };
        shape_t normalized_shape = { 2 };

        auto data = LayerNormCpuTestData<TensorDataType::FP32>::Create(
            "minimal", shape, normalized_shape );

        TestForward( data );
    }

    TEST_F( LayerNormCpuTests, EdgeCase_LargeNormalizedDim )
    {
        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( LayerNormCpuTests, EdgeCase_TransformerSize )
    {
        auto data = LayerNormCpuTestData<TensorDataType::FP32>::Create(
            "transformer", transformer_shape_, transformer_normalized_shape_ );

        TestForward( data );
    }

    TEST_F( LayerNormCpuTests, EdgeCase_AllZeros )
    {
        auto data = SmallFp32Data();
        data.module->build( data.shape );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.shape );

        auto input_ptr = input.data();

        for (size_t i = 0; i < input.size(); ++i)
        {
            input_ptr[i] = 0.0f;
        }

        EXPECT_NO_THROW( data.module->forward( input, output ) );
    }

    TEST_F( LayerNormCpuTests, EdgeCase_ConstantValues )
    {
        auto data = SmallFp32Data();
        data.module->build( data.shape );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.shape );

        auto input_ptr = input.data();

        for (size_t i = 0; i < input.size(); ++i)
        {
            input_ptr[i] = 5.0f;
        }

        EXPECT_NO_THROW( data.module->forward( input, output ) );
    }

    TEST_F( LayerNormCpuTests, Error_NullExecutionContext )
    {
        LayerNormConfig config;
        config.withName( "test" ).withNormalizedShape( { 4 } );

        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> null_ctx;

        EXPECT_THROW(
            (std::make_shared<LayerNorm<DeviceType::Cpu, TensorDataType::FP32>>( null_ctx, config )),
            std::invalid_argument
        );
    }

    TEST_F( LayerNormCpuTests, Error_InvalidConfig )
    {
        LayerNormConfig invalid_config;
        invalid_config.withName( "invalid" );

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        EXPECT_THROW(
            (std::make_shared<LayerNorm<DeviceType::Cpu, TensorDataType::FP32>>( ctx, invalid_config )),
            std::invalid_argument
        );
    }

    TEST_F( LayerNormCpuTests, Error_ForwardBeforeBuild_WithAxis )
    {
        auto data = LayerNormCpuTestData<TensorDataType::FP32>::CreateWithAxis(
            "unbuild", medium_shape_, -1 );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.shape );

        EXPECT_THROW(
            data.module->forward( input, output ),
            std::runtime_error
        );
    }

    TEST_F( LayerNormCpuTests, Error_ShapeMismatch )
    {
        auto data = SmallFp32Data();
        data.module->build( data.shape );

        shape_t wrong_shape = { 2, 3, 8 };

        CpuTensor<TensorDataType::FP32> input( "CPU", wrong_shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", wrong_shape );

        EXPECT_THROW(
            data.module->forward( input, output ),
            std::invalid_argument
        );
    }

    TEST_F( LayerNormCpuTests, Synchronize )
    {
        auto data = SmallFp32Data();

        EXPECT_NO_THROW( data.module->synchronize() );
    }

    TEST_F( LayerNormCpuTests, SetTrainingMode )
    {
        auto data = SmallFp32Data();

        EXPECT_FALSE( data.module->isTraining() );

        data.module->setTraining( true );
        EXPECT_TRUE( data.module->isTraining() );

        data.module->setTraining( false );
        EXPECT_FALSE( data.module->isTraining() );
    }

    TEST_F( LayerNormCpuTests, MultipleForwardCalls )
    {
        auto data = MediumFp32Data();
        data.module->build( data.shape );

        CpuTensor<TensorDataType::FP32> input( "CPU", data.shape );
        CpuTensor<TensorDataType::FP32> output( "CPU", data.shape );

        std::mt19937 rng( 789 );
        std::uniform_real_distribution<float> dist( -2.0f, 2.0f );

        for (int iter = 0; iter < 10; ++iter)
        {
            auto input_ptr = input.data();

            for (size_t i = 0; i < input.size(); ++i)
            {
                input_ptr[i] = dist( rng );
            }

            EXPECT_NO_THROW( data.module->forward( input, output ) );
        }
    }
}