#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cstdint>

import Mila;

namespace Modules::Layers::Tests
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
        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> exec_context;
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
                .withVocabularyLength( static_cast<size_t>(vocab_len) )
                .withName( name );

            d.exec_context = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            d.module = std::make_shared<Encoder<DeviceType::Cuda, TensorDataType::INT32, TPrecision>>( d.exec_context, d.config );

            if (is_training)
                d.module->setTraining( true );

            return d;
        }
    };

    class EncoderCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount( &device_count );
            cuda_available_ = (err == cudaSuccess && device_count > 0);

            batch_size_ = 8;
            sequence_length_ = 16;
            channels_ = 64;
            max_seq_len_ = 128;
            vocab_len_ = 50257;
        }

        EncoderCudaTestData<TensorDataType::FP32>& CudaFp32()
        {
            if (!cuda_fp32_.module)
            {
                cuda_fp32_ = EncoderCudaTestData<TensorDataType::FP32>::Create(
                    "cuda_encoder_fp32", batch_size_, sequence_length_, channels_, max_seq_len_, vocab_len_, false );
            }
            return cuda_fp32_;
        }

        EncoderCudaTestData<TensorDataType::FP32>& CudaFp32Training()
        {
            if (!cuda_fp32_training_.module)
            {
                cuda_fp32_training_ = EncoderCudaTestData<TensorDataType::FP32>::Create(
                    "cuda_encoder_fp32_train", batch_size_, sequence_length_, channels_, max_seq_len_, vocab_len_, true );
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
        if (!d.module)
            GTEST_SKIP() << "Module not initialized";

        d.module->build( d.input_shape );

        CudaIndexTensor device_input( d.exec_context->getDevice(), d.input_shape );
        CudaTensor<TPrecision> device_output( d.exec_context->getDevice(), d.output_shape );

        // Fill host input then copy to device
        HostIndexTensor host_input( d.exec_context->getDevice(), d.input_shape );
        auto hptr = static_cast<int32_t*>(host_input.rawData());
        for (size_t i = 0; i < host_input.size(); ++i)
            hptr[i] = static_cast<int32_t>( i % static_cast<size_t>( d.vocab_len ) );

        copy( host_input, device_input );

        EXPECT_NO_THROW( d.module->forward( device_input, device_output ) );
        EXPECT_EQ( device_output.shape(), d.output_shape );
    }

    template<TensorDataType TPrecision>
    void TestTrainingGradAllocation( EncoderCudaTestData<TPrecision>& d )
    {
        d.module->setTraining( true );
        d.module->build( d.input_shape );

        auto wte_grad = d.module->getWteGrad();
        auto wpe_grad = d.module->getWpeGrad();

        ASSERT_NE( wte_grad, nullptr );
        ASSERT_NE( wpe_grad, nullptr );
    }

    // Tests

    TEST_F( EncoderCudaTests, ParameterCount )
    {
        if (!cuda_available_) GTEST_SKIP() << "CUDA not available";
        TestParameterCount( CudaFp32() );
    }

    TEST_F( EncoderCudaTests, BuildAndForward )
    {
        if (!cuda_available_) GTEST_SKIP() << "CUDA not available";
        TestBuildAndForward( CudaFp32() );
    }

    TEST_F( EncoderCudaTests, Training_GradientsAllocated )
    {
        if (!cuda_available_) GTEST_SKIP() << "CUDA not available";
        TestTrainingGradAllocation( CudaFp32Training() );
    }

    TEST_F( EncoderCudaTests, ToStringContainsName )
    {
        if (!cuda_available_) GTEST_SKIP() << "CUDA not available";
        auto d = CudaFp32();
        auto s = d.module->toString();
        EXPECT_NE( s.find( d.config.getName() ), std::string::npos );
    }

    TEST_F( EncoderCudaTests, Error_NullExecutionContext )
    {
        if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

        EncoderConfig config;
        config.withChannels( 16 ).withMaxSequenceLength( 8 ).withVocabularyLength( 100 ).withName( "bad" );

        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> null_ctx;
        EXPECT_THROW( (std::make_shared<Encoder<DeviceType::Cuda, TensorDataType::INT32, TensorDataType::FP32>>( null_ctx, config )), std::invalid_argument );
    }
}