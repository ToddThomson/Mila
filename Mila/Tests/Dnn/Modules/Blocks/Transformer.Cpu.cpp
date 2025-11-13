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

            // TransformerConfig now requires embedding_dim and num_heads
            TransformerConfig cfg( static_cast<dim_t>(embedding_dim),
                static_cast<dim_t>(num_heads) );
            cfg.withName( name );

            data.transformer_module = std::make_shared<Transformer<TDevice, TPrecision>>( exec_ctx, cfg );

            // Set training mode if requested
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

            // TransformerConfig now constructed with embedding_dim and num_heads
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

        // Factory methods
        TransformerTestData<DeviceType::Cpu, TensorDataType::FP32>& CpuFloatData()
        {
            if (!cpu_float_data_.transformer_module)
            {
                cpu_float_data_ = TransformerTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                    "cpu_transformer_float", cpu_batch_size_, cpu_sequence_length_, embedding_dim_, num_heads_ );
            }
            return cpu_float_data_;
        }

        TransformerTestData<DeviceType::Cpu, TensorDataType::FP32>& TrainingCpuFloatData()
        {
            if (!training_cpu_float_data_.transformer_module)
            {
                training_cpu_float_data_ = TransformerTestData<DeviceType::Cpu, TensorDataType::FP32>::Create(
                    "cpu_transformer_float_training", cpu_batch_size_, cpu_sequence_length_, embedding_dim_, num_heads_, true );
            }
            return training_cpu_float_data_;
        }

        TransformerTestData<DeviceType::Cpu, TensorDataType::FP32>& ContextCpuFloatData()
        {
            if (!context_cpu_float_data_.transformer_module)
            {
                auto cpu_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );
                context_cpu_float_data_ = TransformerTestData<DeviceType::Cpu, TensorDataType::FP32>::CreateWithContext(
                    "cpu_context_transformer_float", cpu_batch_size_, cpu_sequence_length_, embedding_dim_,
                    num_heads_, cpu_context );
            }
            return context_cpu_float_data_;
        }

        // Test parameters
        size_t cpu_batch_size_{ 0 };
        size_t cpu_sequence_length_{ 0 };
        size_t embedding_dim_{ 0 };
        size_t num_heads_{ 0 };

        // Test data
        TransformerTestData<DeviceType::Cpu, TensorDataType::FP32> cpu_float_data_;
        TransformerTestData<DeviceType::Cpu, TensorDataType::FP32> context_cpu_float_data_;
        TransformerTestData<DeviceType::Cpu, TensorDataType::FP32> training_cpu_float_data_;
    };

    // Common test helpers for CPU
    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestGetName( const TransformerTestData<TDevice, TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.transformer_module->getName(), expected_name );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestParameterCount( const TransformerTestData<TDevice, TPrecision>& data )
    {
        data.transformer_module->build( data.input_shape );
        size_t params_count = data.transformer_module->parameterCount();
        EXPECT_GT( params_count, 0 );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestForward( const TransformerTestData<TDevice, TPrecision>& data )
    {
        // Create tensors using the module's device from its execution context
        using TensorT = Tensor<TPrecision, CpuMemoryResource>;

        data.transformer_module->build( data.input_shape );

        auto device = data.transformer_module->getDevice();

        TensorT input( device, data.input_shape );
        TensorT output( device, data.input_shape );

        for (size_t i = 0; i < input.size(); ++i)
        {
            input.data()[i] = static_cast<typename TensorT::host_value_t>( i % 10 * 0.1f );
        }

        data.transformer_module->forward( input, output );

        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestPrint( const TransformerTestData<TDevice, TPrecision>& data, const std::string& expected_substring )
    {
        std::string output = data.transformer_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestTrainingMode( const TransformerTestData<TDevice, TPrecision>& data, bool expected_mode )
    {
        EXPECT_EQ( data.transformer_module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestDeviceType( const TransformerTestData<TDevice, TPrecision>& data )
    {
        auto device = data.transformer_module->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    template<DeviceType TDevice, TensorDataType TPrecision>
    void TestSubModules( const TransformerTestData<TDevice, TPrecision>& data )
    {
        data.transformer_module->build( data.input_shape );

        EXPECT_NE( data.transformer_module->getAttention(), nullptr );
        EXPECT_NE( data.transformer_module->getLn1(), nullptr );
        EXPECT_NE( data.transformer_module->getLn2(), nullptr );
        EXPECT_NE( data.transformer_module->getFFN(), nullptr );
    }

    template<TensorDataType TPrecision, DeviceType TDevice>
    void TestEdgeCases()
    {
        // Explicit CPU tensor type, constructed from module/device context
        using TensorT = Tensor<TPrecision, CpuMemoryResource>;

        try
        {
            shape_t minimal_shape = { 1, 1, 64 };
            size_t minimal_num_heads = 2;

            auto exec_ctx = std::make_shared<ExecutionContext<TDevice>>( TDevice == DeviceType::Cuda ? 0 : -1 );

            // Provide embedding_dim and num_heads for config
            TransformerConfig cfg_min( static_cast<dim_t>(minimal_shape[2]),
                static_cast<dim_t>(minimal_num_heads) );
            cfg_min.withName( "minimal_transformer" );

            auto minimal_module = std::make_shared<Transformer<TDevice, TPrecision>>( exec_ctx, cfg_min );

            auto device_min = minimal_module->getDevice();

            TensorT minimal_input( device_min, minimal_shape );
            TensorT minimal_output( device_min, minimal_shape );

            EXPECT_NO_THROW( minimal_module->build( minimal_shape ) );
            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), minimal_input.size() );

            shape_t medium_shape = { 1, 2, 128 };
            size_t medium_num_heads = 4;

            TransformerConfig cfg_med( static_cast<dim_t>(medium_shape[2]),
                static_cast<dim_t>(medium_num_heads) );
            cfg_med.withName( "medium_transformer" );

            auto medium_module = std::make_shared<Transformer<TDevice, TPrecision>>( exec_ctx, cfg_med );

            auto device_med = medium_module->getDevice();

            TensorT medium_input( device_med, medium_shape );
            TensorT medium_output( device_med, medium_shape );

            EXPECT_NO_THROW( medium_module->build( medium_shape ) );
            EXPECT_NO_THROW( medium_module->forward( medium_input, medium_output ) );
        }
        catch (const std::exception& e)
        {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    TEST_F( TransformerCpuTests, InvalidShape_Throws )
    {
        // Updated to construct config with embedding_dim and num_heads
        shape_t invalid_shape = { 1, 64 }; // Only 2 dimensions
        dim_t embedding_dim = 64;
        dim_t num_heads = 2;

        TransformerConfig cfg( embedding_dim, num_heads );
        cfg.withName( "invalid_transformer" );

        auto exec_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );

        auto module = std::make_shared<Transformer<DeviceType::Cpu, TensorDataType::FP32>>( exec_ctx, cfg );

        EXPECT_THROW( module->build( invalid_shape ), std::invalid_argument );
    }

    // CPU Tests
    TEST_F( TransformerCpuTests, Cpu_Float_TestName )
    {
        TestGetName<DeviceType::Cpu, TensorDataType::FP32>( CpuFloatData(), "cpu_transformer_float" );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_ParameterCount )
    {
        TestParameterCount<DeviceType::Cpu, TensorDataType::FP32>( CpuFloatData() );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_TestForward )
    {
        TestForward<DeviceType::Cpu, TensorDataType::FP32>( CpuFloatData() );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_TestPrint )
    {
        TestPrint<DeviceType::Cpu, TensorDataType::FP32>( CpuFloatData(), "Transformer: cpu_transformer_float" );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cpu, TensorDataType::FP32>( CpuFloatData(), false );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_DeviceType )
    {
        TestDeviceType<DeviceType::Cpu, TensorDataType::FP32>( CpuFloatData() );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_SubModules )
    {
        TestSubModules<DeviceType::Cpu, TensorDataType::FP32>( CpuFloatData() );
    }

    TEST_F( TransformerCpuTests, Cpu_Training_Float_TrainingMode )
    {
        TestTrainingMode<DeviceType::Cpu, TensorDataType::FP32>( TrainingCpuFloatData(), true );
    }

    TEST_F( TransformerCpuTests, Cpu_Training_Float_TestForward )
    {
        TestForward<DeviceType::Cpu, TensorDataType::FP32>( TrainingCpuFloatData() );
    }

    TEST_F( TransformerCpuTests, Context_Cpu_Float_DeviceType )
    {
        TestDeviceType<DeviceType::Cpu, TensorDataType::FP32>( ContextCpuFloatData() );
    }

    TEST_F( TransformerCpuTests, Context_Cpu_Float_Forward )
    {
        TestForward<DeviceType::Cpu, TensorDataType::FP32>( ContextCpuFloatData() );
    }

    TEST_F( TransformerCpuTests, Cpu_Float_EdgeCases )
    {
        TestEdgeCases<TensorDataType::FP32, DeviceType::Cpu>();
    }
}