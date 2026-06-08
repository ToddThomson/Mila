#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

import Mila;

namespace Dnn::Components::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    /**
     * @brief Test data structure for Linear component tests.
     *
     * Name is no longer part of the config. Tests now pass the component name
     * explicitly to the Linear constructor and keep configs name-free.
     */
    template<TensorDataType TPrecision>
    struct LinearCpuTestData
    {
        shape_t input_shape;
        shape_t output_shape;
        LinearConfig config;
        std::shared_ptr<Linear<DeviceType::Cpu, TPrecision>> component;
        int64_t input_features;
        int64_t output_features;
        bool has_bias;

        LinearCpuTestData() : config( 1, 1 ), input_features( 0 ), output_features( 0 ), has_bias( true )
        {}

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
            data.config.withBias( has_bias );

            // Construct using DeviceId directly (component will create/own its context)
            data.component = std::make_shared<Linear<DeviceType::Cpu, TPrecision>>(
                name,
                data.config,
                Device::Cpu()
            );

            return data;
        }

        static LinearCpuTestData CreateWithContext(
            const std::string& name,
            const shape_t& input_shape,
            int64_t input_features,
            int64_t output_features,
            std::unique_ptr<IExecutionContext> context,
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
            data.config.withBias( has_bias );

            // Use DeviceId from provided context to construct component (component creates its own context)
            DeviceId ctx_id = context->getDeviceId();

            data.component = std::make_shared<Linear<DeviceType::Cpu, TPrecision>>(
                name,
                data.config,
                ctx_id
            );

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

            input_features_ = 16;
            output_features_ = 32;
        }

        LinearCpuTestData<TensorDataType::FP32>& SmallFp32Data()
        {
            if ( !small_fp32_.component )
            {
                small_fp32_ = LinearCpuTestData<TensorDataType::FP32>::Create(
                    "small_linear_cpu", small_shape_, input_features_, output_features_ );
            }
            
            return small_fp32_;
        }

        LinearCpuTestData<TensorDataType::FP32>& MediumFp32Data()
        {
            if ( !medium_fp32_.component )
            {
                medium_fp32_ = LinearCpuTestData<TensorDataType::FP32>::Create(
                    "medium_linear_cpu", medium_shape_, 512, 256 );
            }
            return medium_fp32_;
        }

        LinearCpuTestData<TensorDataType::FP32>& NoBiasFp32Data()
        {
            if ( !no_bias_fp32_.component )
            {
                no_bias_fp32_ = LinearCpuTestData<TensorDataType::FP32>::Create(
                    "no_bias_linear_cpu", small_shape_, input_features_, output_features_, false );
            }
            return no_bias_fp32_;
        }

        shape_t small_shape_;
        shape_t medium_shape_;

        int64_t input_features_;
        int64_t output_features_;

        LinearCpuTestData<TensorDataType::FP32> small_fp32_;
        LinearCpuTestData<TensorDataType::FP32> medium_fp32_;
        LinearCpuTestData<TensorDataType::FP32> no_bias_fp32_;
    };

    template<TensorDataType TPrecision>
    void TestGetName( const LinearCpuTestData<TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.component->getName(), expected_name );
    }

    template<TensorDataType TPrecision>
    void TestDeviceType( const LinearCpuTestData<TPrecision>& data )
    {
        EXPECT_EQ( data.component->getDeviceType(), DeviceType::Cpu );

        auto device = data.component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cpu );
    }

    template<TensorDataType TPrecision>
    void TestIsBuilt( const LinearCpuTestData<TPrecision>& data, bool expected_built )
    {
        EXPECT_EQ( data.component->isBuilt(), expected_built );
    }

    template<TensorDataType TPrecision>
    void TestBuild( LinearCpuTestData<TPrecision>& data )
    {
        EXPECT_NO_THROW( data.component->build( data.input_shape ) );
        EXPECT_TRUE( data.component->isBuilt() );
    }

    template<TensorDataType TPrecision>
    void TestParameterCount( const LinearCpuTestData<TPrecision>& data )
    {
        size_t expected_count = data.input_features * data.output_features;
        if ( data.has_bias )
        {
            expected_count += data.output_features;
        }
        EXPECT_EQ( data.component->parameterCount(), expected_count );
    }
    
    template<TensorDataType TPrecision>
    void TestHasBias( const LinearCpuTestData<TPrecision>& data )
    {
        EXPECT_EQ( data.component->hasBias(), data.has_bias );
    }

    template<TensorDataType TPrecision>
    void TestToString( const LinearCpuTestData<TPrecision>& data )
    {
        std::string output = data.component->toString();

        EXPECT_NE( output.find( "Linear" ), std::string::npos );
        EXPECT_NE( output.find( data.component->getName() ), std::string::npos );
        EXPECT_NE( output.find( "Input features:" ), std::string::npos );
        EXPECT_NE( output.find( "Output features:" ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
    }

    template<TensorDataType TPrecision>
    void TestForward( LinearCpuTestData<TPrecision>& data )
    {
        using TensorType = CpuTensor<TPrecision>;

        data.component->build( data.input_shape );

        TensorType input( Device::Cpu(), data.input_shape );

        random( input, -1.0f, 1.0f );

        TensorType* out_ptr = nullptr;
        EXPECT_NO_THROW( out_ptr = &data.component->forward( input ) );
        ASSERT_NE( out_ptr, nullptr );

        auto& out = *out_ptr;

        EXPECT_EQ( out.size(),
            static_cast<size_t>( data.output_shape[ 0 ] * data.output_shape[ 1 ] * data.output_shape[ 2 ] ) );
        EXPECT_EQ( out.shape(), data.output_shape );
    }

    template<TensorDataType TPrecision>
    void TestBackward( LinearCpuTestData<TPrecision>& data )
    {
        using TensorType = CpuTensor<TPrecision>;

        data.component->build( data.input_shape );
        data.component->setTraining( true );

        TensorType input( Device::Cpu(), data.input_shape );
        TensorType output_grad( Device::Cpu(), data.output_shape );

        random( input, -1.0f, 1.0f );
        random( output_grad, -0.1f, 0.1f );

        // Use component-owned output from forward
        TensorType* out_ptr = nullptr;
        EXPECT_NO_THROW( out_ptr = &data.component->forward( input ) );
        ASSERT_NE( out_ptr, nullptr );

        TensorType* in_grad_ptr = nullptr;

        EXPECT_NO_THROW(
            in_grad_ptr = &data.component->backward( input, output_grad )
        ) << "Backward pass should succeed for CPU Linear operation in training mode";

        ASSERT_NE( in_grad_ptr, nullptr );

        auto& in_grad = *in_grad_ptr;

        EXPECT_EQ( in_grad.shape(), data.input_shape );

        bool has_nonzero_grad = false;
        for ( size_t i = 0; i < in_grad.size(); ++i )
        {
            if ( std::abs( in_grad.data()[ i ] ) > 1e-6f )
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

    TEST_F( LinearCpuTests, Construction_WithDeviceId )
    {
        LinearConfig config( 16, 32 );
        config.withBias( true );

        EXPECT_NO_THROW(
            (std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "linear_cpu",
                config,
                Device::Cpu()
            ))
        );
    }

    TEST_F( LinearCpuTests, Construction_NoDeviceId_AllowsSharedMode )
    {
        LinearConfig config( 16, 32 );
        config.withBias( true );

        // Linear requires a device id (or parent must call setExecutionContext).
        EXPECT_NO_THROW(
            (std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "linear_cpu",
                config
            ))
        );
    }

    TEST_F( LinearCpuTests, Construction_DeviceTypeMismatch_Throws )
    {
        LinearConfig config( 16, 32 );
        config.withBias( true );

        EXPECT_THROW(
            (std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "linear_cpu",
                config,
                Device::Cuda( 0 )
            )) ,
            std::invalid_argument
        );
    }

    // ====================================================================
    // Basic Property Tests
    // ====================================================================

    TEST_F( LinearCpuTests, GetName )
    {
        const auto& data = SmallFp32Data();
        TestGetName( data, "small_linear_cpu" );
    }

    TEST_F( LinearCpuTests, DeviceType )
    {
        const auto& data = SmallFp32Data();
        TestDeviceType( data );
    }

    TEST_F( LinearCpuTests, IsBuilt_BeforeBuild )
    {
        TestIsBuilt( SmallFp32Data(), false );
    }

    TEST_F( LinearCpuTests, IsBuilt_AfterBuild )
    {
        auto& data = SmallFp32Data();

        EXPECT_FALSE( data.component->isBuilt() );

        data.component->build( data.input_shape );

        EXPECT_TRUE( data.component->isBuilt() );
    }

    TEST_F( LinearCpuTests, Build )
    {
        auto& data = SmallFp32Data();
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

    TEST_F( LinearCpuTests, HasBias_True )
    {
        TestHasBias( SmallFp32Data() );
    }

    TEST_F( LinearCpuTests, HasBias_False )
    {
        TestHasBias( NoBiasFp32Data() );
    }

    TEST_F( LinearCpuTests, ToString )
    {
        const auto& data = SmallFp32Data();
        TestToString( data );
    }

    // ====================================================================
    // Forward Pass Tests
    // ====================================================================

    TEST_F( LinearCpuTests, Forward_SmallShape )
    {
        auto& data = SmallFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCpuTests, Forward_MediumShape )
    {
        auto& data = MediumFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCpuTests, Forward_WithoutBias )
    {
        auto& data = NoBiasFp32Data();
        TestForward( data );
    }

    TEST_F( LinearCpuTests, WithContext_Construction )
    {
        auto ctx = createExecutionContext( Device::Cpu() );
        auto ctx_id = ctx->getDeviceId();

        auto data = LinearCpuTestData<TensorDataType::FP32>::CreateWithContext(
            "context_linear_cpu", small_shape_, input_features_, output_features_, std::move( ctx ) );

        EXPECT_EQ( data.component->getName(), "context_linear_cpu" );
        EXPECT_EQ( data.component->getDeviceId(), ctx_id );
    }

    TEST_F( LinearCpuTests, EdgeCase_MinimalShape )
    {
        shape_t shape = { 1, 1, 1 };

        auto data = LinearCpuTestData<TensorDataType::FP32>::Create(
            "minimal_cpu", shape, 1, 1 );

        TestForward( data );
    }

    TEST_F( LinearCpuTests, EdgeCase_BatchSize1 )
    {
        shape_t shape = { 1, 8, 16 };

        auto data = LinearCpuTestData<TensorDataType::FP32>::Create(
            "batch1_cpu", shape, 16, 32 );

        TestForward( data );
    }

    // ====================================================================
    // Error Handling Tests
    // ====================================================================

    TEST_F( LinearCpuTests, Error_InvalidConfig_ZeroInputFeatures )
    {
        LinearConfig invalid_config( 0, 32 );
        invalid_config.withBias( true );

        EXPECT_THROW(
            (std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "invalid_fc_cpu",
                invalid_config,
                Device::Cpu()
            )) ,
            std::invalid_argument
        );
    }

    TEST_F( LinearCpuTests, Error_InvalidConfig_ZeroOutputFeatures )
    {
        LinearConfig invalid_config( 16, 0 );
        invalid_config.withBias( true );

        EXPECT_THROW(
            (std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "invalid_fc_cpu",
                invalid_config,
                Device::Cpu()
            )) ,
            std::invalid_argument
        );
    }

    TEST_F( LinearCpuTests, Error_ForwardBeforeBuild )
    {
        auto data = LinearCpuTestData<TensorDataType::FP32>::Create(
            "unbuild_cpu", small_shape_, input_features_, output_features_ );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), data.input_shape );

        EXPECT_THROW(
            data.component->forward( input ),
            std::runtime_error
        );
    }

    TEST_F( LinearCpuTests, Error_ShapeMismatch )
    {
        auto& data = SmallFp32Data();
        data.component->build( data.input_shape );

        shape_t wrong_shape = { 2, 3, 64 };

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), wrong_shape );

        EXPECT_THROW(
            data.component->forward( input ),
            std::invalid_argument
        );
    }

    TEST_F( LinearCpuTests, Synchronize )
    {
        auto& data = SmallFp32Data();

        EXPECT_NO_THROW( data.component->synchronize() );
    }

    TEST_F( LinearCpuTests, SetTrainingMode )
    {
        auto& data = SmallFp32Data();

        EXPECT_FALSE( data.component->isTraining() );

        data.component->build( data.input_shape );
        data.component->setTraining( true );
        EXPECT_TRUE( data.component->isTraining() );

        data.component->setTraining( false );
        EXPECT_FALSE( data.component->isTraining() );
    }

    TEST_F( LinearCpuTests, MultipleForwardCalls )
    {
        auto& data = MediumFp32Data();
        data.component->build( data.input_shape );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), data.input_shape );

        for ( int iter = 0; iter < 10; ++iter )
        {
            random( input, -1.0f, 1.0f );

            CpuTensor<TensorDataType::FP32>* output = nullptr;
            EXPECT_NO_THROW( output = &data.component->forward( input ) );
            ASSERT_NE( output, nullptr );
        }
    }

    // ====================================================================
    // Backward Pass Tests
    // ====================================================================

    TEST_F( LinearCpuTests, Backward_SmallShape )
    {
        auto& data = SmallFp32Data();
        TestBackward( data );
    }

    TEST_F( LinearCpuTests, Backward_MediumShape )
    {
        auto& data = MediumFp32Data();
        TestBackward( data );
    }

    TEST_F( LinearCpuTests, Backward_WithoutBias )
    {
        auto& data = NoBiasFp32Data();
        TestBackward( data );
    }

    TEST_F( LinearCpuTests, Error_BackwardBeforeBuild )
    {
        auto data = LinearCpuTestData<TensorDataType::FP32>::Create(
            "unbuild_backward_cpu", small_shape_, input_features_, output_features_ );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), data.input_shape );
        CpuTensor<TensorDataType::FP32> output_grad( Device::Cpu(), data.output_shape );

        EXPECT_THROW(
            data.component->backward( input, output_grad ),
            std::runtime_error
        );
    }

    TEST_F( LinearCpuTests, Backward_MultipleIterations )
    {
        auto& data = SmallFp32Data();

        data.component->build( data.input_shape );
        data.component->setTraining( true );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), data.input_shape );
        CpuTensor<TensorDataType::FP32> output_grad( Device::Cpu(), data.output_shape );

        for ( int iter = 0; iter < 5; ++iter )
        {
            random( input, -1.0f, 1.0f );
            random( output_grad, -0.1f, 0.1f );

            // establish forward state
            CpuTensor<TensorDataType::FP32>* out_ptr = nullptr;
            EXPECT_NO_THROW( out_ptr = &data.component->forward( input ) );
            ASSERT_NE( out_ptr, nullptr );

            CpuTensor<TensorDataType::FP32>* in_grad_ptr = nullptr;

            EXPECT_NO_THROW(
                in_grad_ptr = &data.component->backward( input, output_grad )
            ) << "Backward iteration " << iter << " failed";

            ASSERT_NE( in_grad_ptr, nullptr );
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
        auto& data = SmallFp32Data();

        EXPECT_FALSE( data.component->isTraining() );
        data.component->build( data.input_shape );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), data.input_shape );
        CpuTensor<TensorDataType::FP32> output_grad( Device::Cpu(), data.output_shape );

        random( input, -1.0f, 1.0f );
        random( output_grad, -0.1f, 0.1f );

        // forward in inference mode
        CpuTensor<TensorDataType::FP32>* out_ptr = nullptr;
        EXPECT_NO_THROW( out_ptr = &data.component->forward( input ) );
        ASSERT_NE( out_ptr, nullptr );

        EXPECT_THROW(
            data.component->backward( input, output_grad ),
            std::runtime_error
        ) << "Backward should fail in inference mode";

        data.component->setTraining( true );
        EXPECT_TRUE( data.component->isTraining() );

        EXPECT_NO_THROW( out_ptr = &data.component->forward( input ) );
        ASSERT_NE( out_ptr, nullptr );

        CpuTensor<TensorDataType::FP32>* in_grad_ptr = nullptr;
        EXPECT_NO_THROW(
            in_grad_ptr = &data.component->backward( input, output_grad )
        ) << "Backward should work after switching to training mode";

        ASSERT_NE( in_grad_ptr, nullptr );

        data.component->setTraining( false );
        EXPECT_FALSE( data.component->isTraining() );

        EXPECT_NO_THROW( out_ptr = &data.component->forward( input ) );
        ASSERT_NE( out_ptr, nullptr );

        EXPECT_THROW(
            data.component->backward( input, output_grad ),
            std::runtime_error
        ) << "Backward should fail after switching back to inference mode";
    }

    TEST_F( LinearCpuTests, Training_EnableBeforeBuild )
    {
        auto& data = SmallFp32Data();

        data.component->build( data.input_shape );
        data.component->setTraining( true );
        
        EXPECT_TRUE( data.component->isTraining() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), data.input_shape );
        CpuTensor<TensorDataType::FP32> output_grad( Device::Cpu(), data.output_shape );

        random( input, -1.0f, 1.0f );
        random( output_grad, -0.1f, 0.1f );

        CpuTensor<TensorDataType::FP32>* out_ptr = nullptr;
        EXPECT_NO_THROW( out_ptr = &data.component->forward( input ) );
        ASSERT_NE( out_ptr, nullptr );

        CpuTensor<TensorDataType::FP32>* in_grad_ptr = nullptr;
        EXPECT_NO_THROW( in_grad_ptr = &data.component->backward( input, output_grad ) );
        ASSERT_NE( in_grad_ptr, nullptr );
    }

    TEST_F( LinearCpuTests, Error_BackwardInInferenceMode )
    {
        auto& data = SmallFp32Data();

        data.component->build( data.input_shape );
        EXPECT_FALSE( data.component->isTraining() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), data.input_shape );
        CpuTensor<TensorDataType::FP32> output_grad( Device::Cpu(), data.output_shape );

        random( input, -1.0f, 1.0f );

        CpuTensor<TensorDataType::FP32>* out_ptr = nullptr;
        EXPECT_NO_THROW( out_ptr = &data.component->forward( input ) );
        ASSERT_NE( out_ptr, nullptr );

        EXPECT_THROW(
            data.component->backward( input, output_grad ),
            std::runtime_error
        ) << "Backward should throw when component is not in training mode";
    }
}