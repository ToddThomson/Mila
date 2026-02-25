#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Components::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    using CpuIndexTensor = Tensor<TensorDataType::INT32, CpuMemoryResource>;

    template<TensorDataType TPrecision>
    struct EncoderCpuTestData
    {
        shape_t input_shape;
        shape_t output_shape;
        LpeConfig config;
        std::shared_ptr<Lpe<DeviceType::Cpu, TensorDataType::INT32, TPrecision>> module;
        int64_t channels;
        int64_t max_seq_len;
        int64_t vocab_len;
        bool is_training{ false };

        static EncoderCpuTestData Create(
            const std::string& name,
            int64_t batch_size,
            int64_t seq_len,
            int64_t channels,
            int64_t max_seq_len,
            int64_t vocab_len,
            bool is_training = false )
        {
            EncoderCpuTestData d;
            d.input_shape = { batch_size, seq_len };
            d.output_shape = { batch_size, seq_len, channels };
            d.channels = channels;
            d.max_seq_len = max_seq_len;
            d.vocab_len = vocab_len;
            d.is_training = is_training;

            d.config.withEmbeddingDim( static_cast<size_t>(channels) )
                .withMaxSequenceLength( static_cast<size_t>(max_seq_len) )
                .withVocabularyLength( static_cast<size_t>(vocab_len) );

            // Construct in standalone mode (component creates/uses a DeviceId for Cpu)
            d.module = std::make_shared<Lpe<DeviceType::Cpu, TensorDataType::INT32, TPrecision>>(
                name,
                d.config,
                Device::Cpu() );

            return d;
        }
    };

    class EncoderCpuTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            batch_size_ = 4;
            sequence_length_ = 8;
            channels_ = 32;
            max_seq_len_ = 64;
            vocab_len_ = 1000;
        }

        // Lazily create
        EncoderCpuTestData<TensorDataType::FP32>& CpuFp32()
        {
            if ( !cpu_fp32_.module )
            {
                cpu_fp32_ = EncoderCpuTestData<TensorDataType::FP32>::Create(
                    "cpu_encoder_fp32", batch_size_, sequence_length_, channels_, max_seq_len_, vocab_len_, false );
            }
            return cpu_fp32_;
        }

        EncoderCpuTestData<TensorDataType::FP32>& CpuFp32Training()
        {
            if ( !cpu_fp32_training_.module )
            {
                cpu_fp32_training_ = EncoderCpuTestData<TensorDataType::FP32>::Create(
                    "cpu_encoder_fp32_train", batch_size_, sequence_length_, channels_, max_seq_len_, vocab_len_, true );
            }
            return cpu_fp32_training_;
        }

        int64_t batch_size_{ 0 };
        int64_t sequence_length_{ 0 };
        int64_t channels_{ 0 };
        int64_t max_seq_len_{ 0 };
        int64_t vocab_len_{ 0 };

        EncoderCpuTestData<TensorDataType::FP32> cpu_fp32_;
        EncoderCpuTestData<TensorDataType::FP32> cpu_fp32_training_;
    };

    // Small helper tests

    template<TensorDataType TPrecision>
    void TestParameterCount( const EncoderCpuTestData<TPrecision>& d )
    {
        size_t expected = (static_cast<size_t>(d.vocab_len) * static_cast<size_t>(d.channels)) +
            (static_cast<size_t>(d.max_seq_len) * static_cast<size_t>(d.channels));
        EXPECT_EQ( d.module->parameterCount(), expected );
    }

    template<TensorDataType TPrecision>
    void TestBuildAndForward( EncoderCpuTestData<TPrecision>& d )
    {
        EXPECT_NO_THROW( d.module->build( d.input_shape ) );
        EXPECT_TRUE( d.module->isBuilt() );

        // Prepare input (INT32 token ids)
        CpuIndexTensor input( d.module->getDeviceId(), d.input_shape );

        // Fill token ids
        auto idx_ptr = input.data();
        for ( size_t i = 0; i < input.size(); ++i )
            idx_ptr[ i ] = static_cast<int32_t>( i % static_cast<size_t>( d.vocab_len ) );

        // Forward now returns a reference to the component-owned tensor
        auto& out_tensor = d.module->forward( input );

        EXPECT_EQ( out_tensor.shape(), d.output_shape );
        EXPECT_EQ( out_tensor.size(),
            static_cast<size_t>( d.input_shape[ 0 ] * d.input_shape[ 1 ] * d.channels ) );
    }

    template<TensorDataType TPrecision>
    void TestToStringAndDimensions( const EncoderCpuTestData<TPrecision>& d )
    {
        auto s = d.module->toString();
        EXPECT_NE( s.find( d.module->getName() ), std::string::npos );
        EXPECT_EQ( d.module->getEmbeddingDim(), static_cast<int64_t>(d.channels) );
        EXPECT_EQ( d.module->getVocabularyLength(), static_cast<int64_t>(d.vocab_len) );
        EXPECT_EQ( d.module->getMaxSequenceLength(), static_cast<int64_t>(d.max_seq_len) );
    }

    template<TensorDataType TPrecision>
    void TestTrainingGradAllocation( EncoderCpuTestData<TPrecision>& d )
    {
        d.module->build( d.input_shape );
        d.module->setTraining( true );

        auto wte_grad = d.module->getWteGrad();
        auto wpe_grad = d.module->getWpeGrad();

        ASSERT_NE( wte_grad, nullptr );
        ASSERT_NE( wpe_grad, nullptr );

        EXPECT_EQ( wte_grad->shape()[ 0 ], static_cast<int64_t>(d.vocab_len) );
        EXPECT_EQ( wte_grad->shape()[ 1 ], static_cast<int64_t>(d.channels) );

        EXPECT_EQ( wpe_grad->shape()[ 0 ], static_cast<int64_t>(d.max_seq_len) );
        EXPECT_EQ( wpe_grad->shape()[ 1 ], static_cast<int64_t>(d.channels) );
    }

    // Backward API smoke test: ensures backward returns component-owned input-grad tensor
    template<TensorDataType TPrecision>
    void TestBuildForwardBackward( EncoderCpuTestData<TPrecision>& d )
    {
        EXPECT_NO_THROW( d.module->build( d.input_shape ) );
        EXPECT_TRUE( d.module->isBuilt() );

        d.module->setTraining( true );

        // Prepare input (INT32 token ids)
        CpuIndexTensor input( d.module->getDeviceId(), d.input_shape );

        // Fill token ids
        auto idx_ptr = input.data();
        for ( size_t i = 0; i < input.size(); ++i )
            idx_ptr[ i ] = static_cast<int32_t>( i % static_cast<size_t>( d.vocab_len ) );

        // Forward
        auto& embeddings = d.module->forward( input );

        EXPECT_EQ( embeddings.shape(), d.output_shape );

        // Prepare an output gradient tensor matching embeddings shape
        CpuTensor<TPrecision> output_grad( d.module->getDeviceId(), d.output_shape );

        // Fill deterministic gradient values
        for ( size_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = static_cast<float>( (i % 101) ) * 0.01f;
        }

        // Backward now returns a reference to component-owned input-gradient tensor
        auto& in_grad = d.module->backward( input, output_grad );

        // Input-grad should match input shape and size
        EXPECT_EQ( in_grad.shape(), d.input_shape );
        EXPECT_EQ( in_grad.size(), static_cast<size_t>( d.input_shape[ 0 ] * d.input_shape[ 1 ] ) );
    }

    // Tests

    TEST_F( EncoderCpuTests, ParameterCount )
    {
        TestParameterCount( CpuFp32() );
    }

    TEST_F( EncoderCpuTests, BuildAndForward )
    {
        TestBuildAndForward( CpuFp32() );
    }

    TEST_F( EncoderCpuTests, BuildForwardBackward )
    {
        TestBuildForwardBackward( CpuFp32Training() );
    }

    TEST_F( EncoderCpuTests, ToStringAndDimensions )
    {
        TestToStringAndDimensions( CpuFp32() );
    }

    TEST_F( EncoderCpuTests, Training_GradientsAllocated )
    {
        TestTrainingGradAllocation( CpuFp32Training() );
    }

    TEST_F( EncoderCpuTests, Constructor_WithoutDeviceId_AllowsDeferredContext_BuildThrows )
    {
        // Construct in shared mode (no device id) — build should fail because no execution context set
        LpeConfig config;
        config.withEmbeddingDim( 16 )
            .withMaxSequenceLength( 8 )
            .withVocabularyLength( 100 );

        auto component = std::make_shared<Lpe<DeviceType::Cpu, TensorDataType::INT32, TensorDataType::FP32>>(
            "deferred_ctx_encoder",
            config );

        EXPECT_THROW( component->build( shape_t{ 1,1 } ), std::runtime_error );
    }

    TEST_F( EncoderCpuTests, Constructor_WithInvalidDevice_ThrowsInvalidArgument )
    {
        // Passing a wrong device type for a CPU-instantiated encoder should throw
        LpeConfig config;
        config.withEmbeddingDim( 16 )
            .withMaxSequenceLength( 8 )
            .withVocabularyLength( 100 );

        EXPECT_THROW(
            ((void)std::make_shared<Lpe<DeviceType::Cpu, TensorDataType::INT32, TensorDataType::FP32>>(
                "invalid_device",
                config,
                Device::Cuda( 0 ) ) ),
            std::invalid_argument );
    }

    TEST_F( EncoderCpuTests, EdgeCase_MinimalShape )
    {
        EncoderCpuTestData<TensorDataType::FP32> d = EncoderCpuTestData<TensorDataType::FP32>::Create(
            "minimal_cpu", 1, 1, 16, 8, 50 );

        EXPECT_NO_THROW( d.module->build( d.input_shape ) );
        CpuIndexTensor input( d.module->getDeviceId(), d.input_shape );

        auto idx_ptr = input.data();
        idx_ptr[ 0 ] = 0;

        // Forward returns a reference to the component-owned embeddings tensor
        auto& out_tensor = d.module->forward( input );

        EXPECT_EQ( out_tensor.shape(), d.output_shape );
    }
}