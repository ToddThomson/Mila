#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Modules::Layers::Tests
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
        EncoderConfig config;
        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_context;
        std::shared_ptr<Encoder<DeviceType::Cpu, TensorDataType::INT32, TPrecision>> module;
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

            d.config.withChannels( static_cast<size_t>(channels) )
                .withMaxSequenceLength( static_cast<size_t>(max_seq_len) )
                .withVocabularyLength( static_cast<size_t>(vocab_len) )
                .withName( name );

            d.exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            d.module = std::make_shared<Encoder<DeviceType::Cpu, TensorDataType::INT32, TPrecision>>( d.exec_context, d.config );

            if (is_training)
                d.module->setTraining( true );

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
            if (!cpu_fp32_.module)
            {
                cpu_fp32_ = EncoderCpuTestData<TensorDataType::FP32>::Create(
                    "cpu_encoder_fp32", batch_size_, sequence_length_, channels_, max_seq_len_, vocab_len_, false );
            }
            return cpu_fp32_;
        }

        EncoderCpuTestData<TensorDataType::FP32>& CpuFp32Training()
        {
            if (!cpu_fp32_training_.module)
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
        // Build
        EXPECT_NO_THROW( d.module->build( d.input_shape ) );
        EXPECT_TRUE( d.module->isBuilt() );

        // Prepare input (INT32 token ids) and output
        CpuIndexTensor input( d.exec_context->getDevice(), d.input_shape );
        CpuTensor<TPrecision> output( d.exec_context->getDevice(), d.output_shape );

        // Fill token ids
        auto idx_ptr = static_cast<int32_t*>(input.rawData());
        for (size_t i = 0; i < input.size(); ++i)
            idx_ptr[i] = static_cast<int32_t>( i % static_cast<size_t>( d.vocab_len ) );

        EXPECT_NO_THROW( d.module->forward( input, output ) );
        EXPECT_EQ( output.shape(), d.output_shape );
        EXPECT_EQ( output.size(), static_cast<size_t>( d.input_shape[0] * d.input_shape[1] * d.channels ) );
    }

    template<TensorDataType TPrecision>
    void TestToStringAndDimensions( const EncoderCpuTestData<TPrecision>& d )
    {
        auto s = d.module->toString();
        EXPECT_NE( s.find( d.config.getName() ), std::string::npos );
        EXPECT_EQ( d.module->getChannels(), static_cast<int64_t>(d.channels) );
        EXPECT_EQ( d.module->getVocabularyLength(), static_cast<int64_t>(d.vocab_len) );
        EXPECT_EQ( d.module->getMaxSequenceLength(), static_cast<int64_t>(d.max_seq_len) );
    }

    template<TensorDataType TPrecision>
    void TestTrainingGradAllocation( EncoderCpuTestData<TPrecision>& d )
    {
        d.module->setTraining( true );
        d.module->build( d.input_shape );

        auto wte_grad = d.module->getWteGrad();
        auto wpe_grad = d.module->getWpeGrad();

        ASSERT_NE( wte_grad, nullptr );
        ASSERT_NE( wpe_grad, nullptr );

        EXPECT_EQ( wte_grad->shape()[0], static_cast<int64_t>(d.vocab_len) );
        EXPECT_EQ( wte_grad->shape()[1], static_cast<int64_t>(d.channels) );

        EXPECT_EQ( wpe_grad->shape()[0], static_cast<int64_t>(d.max_seq_len) );
        EXPECT_EQ( wpe_grad->shape()[1], static_cast<int64_t>(d.channels) );
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

    TEST_F( EncoderCpuTests, ToStringAndDimensions )
    {
        TestToStringAndDimensions( CpuFp32() );
    }

    TEST_F( EncoderCpuTests, Training_GradientsAllocated )
    {
        TestTrainingGradAllocation( CpuFp32Training() );
    }

    TEST_F( EncoderCpuTests, Error_NullExecutionContext )
    {
        EncoderConfig config;
        config.withChannels( 16 ).withMaxSequenceLength( 8 ).withVocabularyLength( 100 ).withName( "bad" );

        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> null_ctx;
        EXPECT_THROW( (std::make_shared<Encoder<DeviceType::Cpu, TensorDataType::INT32, TensorDataType::FP32>>( null_ctx, config )), std::invalid_argument );
    }

    TEST_F( EncoderCpuTests, EdgeCase_MinimalShape )
    {
        EncoderCpuTestData<TensorDataType::FP32> d = EncoderCpuTestData<TensorDataType::FP32>::Create(
            "minimal_cpu", 1, 1, 16, 8, 50 );

        EXPECT_NO_THROW( d.module->build( d.input_shape ) );
        CpuIndexTensor input( d.exec_context->getDevice(), d.input_shape );
        CpuTensor<TensorDataType::FP32> output( d.exec_context->getDevice(), d.output_shape );
        auto idx_ptr = static_cast<int32_t*>(input.rawData());
        idx_ptr[0] = 0;
        EXPECT_NO_THROW( d.module->forward( input, output ) );
    }
}