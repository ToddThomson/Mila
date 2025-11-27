#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <type_traits>

import Mila;

namespace Modules::Blocks::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<DeviceType TDevice, TensorDataType TPrecision>
    struct TransformerTestData
    {
        shape_t input_shape;
        size_t num_heads;
        std::shared_ptr<Transformer<TDevice, TPrecision>> transformer_module;
        bool is_training{ false };

        // Create using an ExecutionContext
        static TransformerTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t embedding_dim,
            size_t num_heads,
            bool is_training = false )
        {
            TransformerTestData data;
            data.input_shape = { static_cast<int64_t>(batch_size),
                                 static_cast<int64_t>(sequence_length),
                                 static_cast<int64_t>(embedding_dim) };
            data.num_heads = num_heads;
            data.is_training = is_training;

            auto exec_ctx = std::make_shared<ExecutionContext<TDevice>>( TDevice == DeviceType::Cuda ? 0 : -1 );

            TransformerConfig cfg( static_cast<dim_t>(embedding_dim),
                static_cast<dim_t>(num_heads) );
            cfg.withName( name );

            data.transformer_module = std::make_shared<Transformer<TDevice, TPrecision>>( exec_ctx, cfg );

            data.transformer_module->setTraining( is_training );

            return data;
        }

        // Create with provided execution context
        static TransformerTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t embedding_dim,
            size_t num_heads,
            std::shared_ptr<ExecutionContext<TDevice>> context,
            bool is_training = false )
        {
            TransformerTestData data;
            data.input_shape = { static_cast<int64_t>(batch_size),
                                 static_cast<int64_t>(sequence_length),
                                 static_cast<int64_t>(embedding_dim) };
            data.num_heads = num_heads;
            data.is_training = is_training;

            TransformerConfig cfg( static_cast<dim_t>(embedding_dim),
                static_cast<dim_t>(num_heads) );
            cfg.withName( name );

            data.transformer_module = std::make_shared<Transformer<TDevice, TPrecision>>( context, cfg );
            data.transformer_module->setTraining( is_training );

            return data;
        }
    };

    class TransformerCpuTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // CPU-specific parameters
            cpu_batch_size_ = 1;
            cpu_sequence_length_ = 8;

            // Common parameters
            embedding_dim_ = 256;
            num_heads_ = 8; // Must divide evenly into embedding_dim
        }

        TransformerTestData<DeviceType::Cpu, TensorDataType::FP32> CpuFloatData()
        {
            return TransformerTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                "cpu_transformer_float", cpu_batch_size_, cpu_sequence_length_, embedding_dim_, num_heads_ );
        }

        TransformerTestData<DeviceType::Cpu, TensorDataType::FP32> TrainingCpuFloatData()
        {
            return TransformerTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                "cpu_transformer_float_training", cpu_batch_size_, cpu_sequence_length_, embedding_dim_, num_heads_, true );
        }

        TransformerTestData<DeviceType::Cpu, TensorDataType::FP32> ContextCpuFloatData()
        {
            auto cpu_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );
            return TransformerTestData<DeviceType::Cpu, TensorDataType::FP32>::CreateWithContext(
                "cpu_context_transformer_float", cpu_batch_size_, cpu_sequence_length_, embedding_dim_,
                num_heads_, cpu_context );
        }

        // Test parameters
        size_t cpu_batch_size_{ 0 };
        size_t cpu_sequence_length_{ 0 };
        size_t embedding_dim_{ 0 };
        size_t num_heads_{ 0 };
    };

    TEST_F( TransformerCpuTests, InvalidShape_Throws )
    {
        // Construct config with embedding_dim and num_heads then attempt to build with invalid shape
        shape_t invalid_shape = { 1, 64 }; // Only 2 dimensions
        dim_t embedding_dim = 64;
        dim_t num_heads = 2;

        TransformerConfig cfg( embedding_dim, num_heads );
        cfg.withName( "invalid_transformer" );

        auto exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );

        auto module = std::make_shared<Transformer<DeviceType::Cpu, TensorDataType::FP32>>( exec_ctx, cfg );

        EXPECT_THROW( module->build( invalid_shape ), std::invalid_argument );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_TestName )
    {
        // Setup
        auto data = CpuFloatData();

        // Assertions
        ASSERT_NE( data.transformer_module, nullptr );
        EXPECT_EQ( data.transformer_module->getName(), "cpu_transformer_float" );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_ParameterCount )
    {
        // Setup
        auto data = CpuFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        // Execution
        EXPECT_NO_THROW( data.transformer_module->build( data.input_shape ) );

        // Assertion: expect some parameters
        size_t params_count = data.transformer_module->parameterCount();
        EXPECT_GT( params_count, 0u );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_TestForward )
    {
        // Setup
        auto data = CpuFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // Build and allocate IO
        data.transformer_module->build( data.input_shape );

        auto device = data.transformer_module->getDevice();

        TensorT input( device, data.input_shape );
        TensorT output( device, data.input_shape );

        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[i] = static_cast<TensorT::host_value_t>( i % 10 * 0.1f );
        }

        // Execute
        EXPECT_NO_THROW( data.transformer_module->forward( input, output ) );

        // Verify shapes
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_TestPrint )
    {
        // Setup
        auto data = CpuFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        // Execution
        std::string output = data.transformer_module->toString();

        // Assertion
        EXPECT_NE( output.find( "Transformer: cpu_transformer_float" ), std::string::npos );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_TrainingMode )
    {
        // Setup
        auto data = CpuFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        // Assertion: default should be false
        EXPECT_FALSE( data.transformer_module->isTraining() );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_DeviceType )
    {
        // Setup
        auto data = CpuFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        auto device = data.transformer_module->getDevice();

        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( TransformerCpuTests, Cpu_Training_Float_TrainingMode )
    {
        // Setup: training enabled at creation
        auto data = TrainingCpuFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        EXPECT_TRUE( data.transformer_module->isTraining() );
    }

    TEST_F( TransformerCpuTests, Cpu_Training_Float_TestForward )
    {
        // Setup: training enabled
        auto data = TrainingCpuFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // Build and forward
        data.transformer_module->build( data.input_shape );

        auto device = data.transformer_module->getDevice();

        TensorT input( device, data.input_shape );
        TensorT output( device, data.input_shape );

        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[i] = static_cast<TensorT::host_value_t>( i % 10 * 0.1f );
        }

        EXPECT_NO_THROW( data.transformer_module->forward( input, output ) );
    }

    TEST_F( TransformerCpuTests, Context_Cpu_Float_DeviceType )
    {
        // Setup: explicit context
        auto data = ContextCpuFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        auto device = data.transformer_module->getDevice();

        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( TransformerCpuTests, Forward_FP32 )
    {
        // Setup: explicit context
        auto data = ContextCpuFloatData();

        ASSERT_NE( data.transformer_module, nullptr );

        using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        data.transformer_module->build( data.input_shape );

        auto device = data.transformer_module->getDevice();

        TensorT input( device, data.input_shape );
        TensorT output( device, data.input_shape );

        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[i] = static_cast<TensorT::host_value_t>( i % 10 * 0.1f );
        }

        EXPECT_NO_THROW( data.transformer_module->forward( input, output ) );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_EdgeCases )
    {
        // Minimal case
        {
            shape_t minimal_shape = { 1, 1, 64 };
            size_t minimal_num_heads = 2;

            auto exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );

            TransformerConfig cfg_min( static_cast<dim_t>(minimal_shape[2]),
                static_cast<dim_t>(minimal_num_heads) );
            cfg_min.withName( "minimal_transformer" );

            auto minimal_module = std::make_shared<Transformer<DeviceType::Cpu, TensorDataType::FP32>>( exec_ctx, cfg_min );

            auto device_min = minimal_module->getDevice();

            using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;

            TensorT minimal_input( device_min, minimal_shape );
            TensorT minimal_output( device_min, minimal_shape );

            EXPECT_NO_THROW( minimal_module->build( minimal_shape ) );
            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), minimal_input.size() );
        }

        // Medium case
        {
            shape_t medium_shape = { 1, 2, 128 };
            size_t medium_num_heads = 4;

            auto exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );

            TransformerConfig cfg_med( static_cast<dim_t>(medium_shape[2]),
                static_cast<dim_t>(medium_num_heads) );
            cfg_med.withName( "medium_transformer" );

            auto medium_module = std::make_shared<Transformer<DeviceType::Cpu, TensorDataType::FP32>>( exec_ctx, cfg_med );

            auto device_med = medium_module->getDevice();

            using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;

            TensorT medium_input( device_med, medium_shape );
            TensorT medium_output( device_med, medium_shape );

            EXPECT_NO_THROW( medium_module->build( medium_shape ) );
            EXPECT_NO_THROW( medium_module->forward( medium_input, medium_output ) );
        }
    }
}