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
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    using CudaIndexTensor = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>;
    using HostIndexTensor = Tensor<TensorDataType::INT32, CpuMemoryResource>;
    template<TensorDataType TPrecision>
    using HostTensor = Tensor<TPrecision, CpuMemoryResource>;

    template<TensorDataType TPrecision>
    struct EncoderCudaTestData
    {
        shape_t input_shape;
        shape_t output_shape;
        EncoderConfig config;
        std::shared_ptr<Encoder<DeviceType::Cuda, TensorDataType::INT32, TPrecision>> module;
        int64_t channels;
        int64_t max_seq_len;
        int64_t vocab_len;
        bool is_training{ false };

        static EncoderCudaTestData Create(
            const std::string& name,
            int64_t batch_size,
            int64_t seq_len,
            int64_t channels,
            int64_t max_seq_len,
            int64_t vocab_len,
            bool is_training = false )
        {
            EncoderCudaTestData d;
            d.input_shape = { batch_size, seq_len };
            d.output_shape = { batch_size, seq_len, channels };
            d.channels = channels;
            d.max_seq_len = max_seq_len;
            d.vocab_len = vocab_len;
            d.is_training = is_training;

            d.config.withChannels( static_cast<size_t>(channels) )
                .withMaxSequenceLength( static_cast<size_t>(max_seq_len) )
                .withVocabularyLength( static_cast<size_t>(vocab_len) );

            // Construct in standalone mode so component owns its ExecutionContext
            d.module = std::make_shared<Encoder<DeviceType::Cuda, TensorDataType::INT32, TPrecision>>(
                name,
                d.config,
                Device::Cuda( 0 )
            );

            return d;
        }
    };

    class EncoderCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = getDeviceCount( DeviceType::Cuda );
            cuda_available_ = (device_count > 0);

            batch_size_ = 8;
            sequence_length_ = 16;
            channels_ = 64;
            max_seq_len_ = 128;
            vocab_len_ = 50257;
        }

        EncoderCudaTestData<TensorDataType::FP32>& CudaFp32()
        {
            if ( !cuda_fp32_.module )
            {
                cuda_fp32_ = EncoderCudaTestData<TensorDataType::FP32>::Create(
                    "cuda_encoder_fp32",
                    batch_size_,
                    sequence_length_,
                    channels_,
                    max_seq_len_,
                    vocab_len_,
                    false );
            }

            return cuda_fp32_;
        }

        EncoderCudaTestData<TensorDataType::FP32>& CudaFp32Training()
        {
            if ( !cuda_fp32_training_.module )
            {
                cuda_fp32_training_ = EncoderCudaTestData<TensorDataType::FP32>::Create(
                    "cuda_encoder_fp32_train",
                    batch_size_,
                    sequence_length_,
                    channels_,
                    max_seq_len_,
                    vocab_len_,
                    true );
            }

            return cuda_fp32_training_;
        }

        bool cuda_available_{ false };
        int64_t batch_size_{ 0 };
        int64_t sequence_length_{ 0 };
        int64_t channels_{ 0 };
        int64_t max_seq_len_{ 0 };
        int64_t vocab_len_{ 0 };

        EncoderCudaTestData<TensorDataType::FP32> cuda_fp32_;
        EncoderCudaTestData<TensorDataType::FP32> cuda_fp32_training_;
    };

    // Helper verifications

    template<TensorDataType TPrecision>
    void TestParameterCount( const EncoderCudaTestData<TPrecision>& d )
    {
        size_t expected = (static_cast<size_t>(d.vocab_len) * static_cast<size_t>(d.channels)) +
            (static_cast<size_t>(d.max_seq_len) * static_cast<size_t>(d.channels));

        EXPECT_EQ( d.module->parameterCount(), expected );
    }

    template<TensorDataType TPrecision>
    void TestBuildAndForward( EncoderCudaTestData<TPrecision>& d )
    {
        if ( !d.module )
        {
            GTEST_SKIP() << "Module not initialized";
        }

        // Ensure module is built before forward
        EXPECT_NO_THROW( d.module->build( d.input_shape ) );
        EXPECT_TRUE( d.module->isBuilt() );

        // Device tensors created from module's device id
        CudaIndexTensor device_input( d.module->getDeviceId(), d.input_shape );

        // Fill host input then copy to device
        HostIndexTensor host_input( Device::Cpu(), d.input_shape );
        auto hptr = host_input.data();
        for ( size_t i = 0; i < host_input.size(); ++i )
        {
            hptr[ i ] = static_cast<int32_t>( i % static_cast<size_t>( d.vocab_len ) );
        }

        copy( host_input, device_input );

        // Forward now returns a reference to the component-owned embeddings tensor
        auto& out_tensor = d.module->forward( device_input );

        EXPECT_EQ( out_tensor.shape(), d.output_shape );
    }

    template<TensorDataType TPrecision>
    void TestBuildForwardBackward( EncoderCudaTestData<TPrecision>& d )
    {
        if ( !d.module )
        {
            GTEST_SKIP() << "Module not initialized";
        }

        // Build first, then enable training according to component lifecycle rules
        EXPECT_NO_THROW( d.module->build( d.input_shape ) );
        EXPECT_TRUE( d.module->isBuilt() );

        d.module->setTraining( true );

        // Prepare token indices on host and copy to device
        HostIndexTensor host_input( Device::Cpu(), d.input_shape );
        auto hptr = host_input.data();
        for ( size_t i = 0; i < host_input.size(); ++i )
        {
            hptr[ i ] = static_cast<int32_t>( i % static_cast<size_t>( d.vocab_len ) );
        }

        CudaIndexTensor device_input( d.module->getDeviceId(), d.input_shape );
        copy( host_input, device_input );

        // Forward -> embeddings (device tensor)
        auto& embeddings = d.module->forward( device_input );
        EXPECT_EQ( embeddings.shape(), d.output_shape );

        // Prepare output gradient on host then copy to device
        HostTensor<TPrecision> host_out_grad( Device::Cpu(), d.output_shape );
        for ( size_t i = 0; i < host_out_grad.size(); ++i )
        {
            host_out_grad.data()[ i ] = static_cast<float>( (i % 101) ) * 0.01f;
        }

        CudaTensor<TPrecision> device_out_grad( d.module->getDeviceId(), d.output_shape );
        copy( host_out_grad, device_out_grad );

        // Backward returns reference to component-owned input-gradient tensor (token indices type)
        auto& in_grad = d.module->backward( device_input, device_out_grad );

        // Input-grad should match input shape and size
        EXPECT_EQ( in_grad.shape(), d.input_shape );
        EXPECT_EQ( in_grad.size(), static_cast<size_t>( d.input_shape[ 0 ] * d.input_shape[ 1 ] ) );
    }

    template<TensorDataType TPrecision>
    void TestTrainingGradAllocation( EncoderCudaTestData<TPrecision>& d )
    {
        if ( !d.module )
        {
            GTEST_SKIP() << "Module not initialized";
        }

        d.module->build( d.input_shape );
        d.module->setTraining( true );

        auto wte_grad = d.module->getWteGrad();
        auto wpe_grad = d.module->getWpeGrad();

        ASSERT_NE( wte_grad, nullptr );
        ASSERT_NE( wpe_grad, nullptr );

        EXPECT_EQ( wte_grad->shape()[ 0 ], static_cast<int64_t>(d.vocab_len) );
        EXPECT_EQ( wpe_grad->shape()[ 0 ], static_cast<int64_t>(d.max_seq_len) );
    }

    // Tests

    TEST_F( EncoderCudaTests, ParameterCount )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        TestParameterCount( CudaFp32() );
    }

    TEST_F( EncoderCudaTests, BuildAndForward )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        TestBuildAndForward( CudaFp32() );
    }

    TEST_F( EncoderCudaTests, BuildForwardBackward )
    {
        if ( !cuda_available_ ) 
            GTEST_SKIP() << "CUDA not available";

        TestBuildForwardBackward( CudaFp32Training() );
    }

    TEST_F( EncoderCudaTests, Training_GradientsAllocated )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        TestTrainingGradAllocation( CudaFp32Training() );
    }

    TEST_F( EncoderCudaTests, ToStringContainsName )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        auto d = CudaFp32();
        auto s = d.module->toString();
        EXPECT_NE( s.find( d.module->getName() ), std::string::npos );
    }

    TEST_F( EncoderCudaTests, Constructor_WithoutDeviceId_AllowsDeferredContext_BuildThrows )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        EncoderConfig cfg;
        cfg.withChannels( 16 ).withMaxSequenceLength( 8 ).withVocabularyLength( 100 );

        auto component = std::make_shared<Encoder<DeviceType::Cuda, TensorDataType::INT32, TensorDataType::FP32>>(
            "deferred_cuda_encoder",
            cfg );

        EXPECT_THROW( component->build( shape_t{ 1,1 } ), std::runtime_error );
    }

    TEST_F( EncoderCudaTests, Constructor_WithInvalidDevice_ThrowsInvalidArgument )
    {
        if ( !cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        EncoderConfig cfg;
        cfg.withChannels( 16 )
            .withMaxSequenceLength( 8 )
            .withVocabularyLength( 100 );

        EXPECT_THROW(
            ( (void)std::make_shared<Encoder<DeviceType::Cuda, TensorDataType::INT32, TensorDataType::FP32>>(
                "invalid_device",
                cfg,
                Device::Cpu() ) ),
            std::invalid_argument );
    }
}