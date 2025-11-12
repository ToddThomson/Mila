#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

    class AttentionCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Use DeviceRegistry to determine whether CUDA support was registered.
            // Tests must not call CUDA runtime APIs directly.
            if (!DeviceRegistry::instance().hasDeviceType( "CUDA" ))
            {
                cuda_available_ = false;
                return;
            }

            cuda_available_ = true;
            exec_ctx_ = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        }

        bool cuda_available_{ false };
        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> exec_ctx_;
    };

    TEST_F( AttentionCudaTests, SkipWhenCudaMissing )
    {
        if (!cuda_available_) GTEST_SKIP() << "CUDA not available (DeviceRegistry)";
    }

    TEST_F( AttentionCudaTests, Constructor_NullContext_Throws )
    {
        if (!DeviceRegistry::instance().hasDeviceType( "CUDA" )) GTEST_SKIP() << "CUDA not available";

        AttentionConfig cfg( 64, 8 );
        cfg.withName( "attn_null_cuda" );

        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> null_ctx;
        EXPECT_THROW(
            (std::make_shared<Attention<DeviceType::Cuda, TensorDataType::FP32>>( null_ctx, cfg )),
            std::invalid_argument );
    }

    TEST_F( AttentionCudaTests, BuildAndForward_Fp32 )
    {
        if (!DeviceRegistry::instance().hasDeviceType( "CUDA" )) GTEST_SKIP() << "CUDA not available";

        const int64_t B = 2;
        const int64_t T = 8;
        const int64_t C = 64;
        const int64_t heads = 8;
        const int64_t hs = C / heads;

        AttentionConfig cfg( C, heads );
        cfg.withName( "attn_cuda_fp32" );

        auto module = std::make_shared<Attention<DeviceType::Cuda, TensorDataType::FP32>>( exec_ctx_, cfg );

        // Head-major shapes: [B, NH, T, hs]
        shape_t input_shape = { B, heads, T, hs };
        shape_t output_shape = { B, heads, T, hs };

        EXPECT_NO_THROW( module->build( input_shape ) );

        // Prepare host inputs Q/K/V and device tensors
        HostTensor host_Q( "CPU", input_shape );
        HostTensor host_K( "CPU", input_shape );
        HostTensor host_V( "CPU", input_shape );

        for (size_t i = 0; i < host_Q.size(); ++i)
        {
            host_Q.data()[i] = static_cast<float>( (i % 100) ) * 0.01f;
            host_K.data()[i] = static_cast<float>( (i % 97) ) * 0.02f;
            host_V.data()[i] = static_cast<float>( (i % 89) ) * 0.03f;
        }

        CudaTensor<TensorDataType::FP32> device_Q( exec_ctx_->getDevice(), input_shape );
        CudaTensor<TensorDataType::FP32> device_K( exec_ctx_->getDevice(), input_shape );
        CudaTensor<TensorDataType::FP32> device_V( exec_ctx_->getDevice(), input_shape );
        CudaTensor<TensorDataType::FP32> device_output( exec_ctx_->getDevice(), output_shape );

        copy( host_Q, device_Q );
        copy( host_K, device_K );
        copy( host_V, device_V );

        EXPECT_NO_THROW( module->forward( device_Q, device_K, device_V, device_output ) );

        // Copy back to host to verify shape
        HostTensor host_output = toHost<TensorDataType::FP32>( device_output );
        EXPECT_EQ( host_output.size(), device_output.size() );
        EXPECT_EQ( host_output.shape(), output_shape );
    }

    TEST_F( AttentionCudaTests, BuildAndForward_Fp16 )
    {
        if (!DeviceRegistry::instance().hasDeviceType( "CUDA" )) GTEST_SKIP() << "CUDA not available";

        const int64_t B = 2;
        const int64_t T = 8;
        const int64_t C = 64;
        const int64_t heads = 8;
        const int64_t hs = C / heads;

        AttentionConfig cfg( C, heads );
        cfg.withName( "attn_cuda_fp16" );

        auto module = std::make_shared<Attention<DeviceType::Cuda, TensorDataType::FP16>>( exec_ctx_, cfg );

        // Head-major shapes: [B, NH, T, hs]
        shape_t input_shape = { B, heads, T, hs };
        shape_t output_shape = { B, heads, T, hs };

        EXPECT_NO_THROW( module->build( input_shape ) );

        // Prepare host inputs (FP32) and device FP16 tensors
        HostTensor host_Q( "CPU", input_shape );
        HostTensor host_K( "CPU", input_shape );
        HostTensor host_V( "CPU", input_shape );

        for (size_t i = 0; i < host_Q.size(); ++i)
        {
            host_Q.data()[i] = static_cast<float>( (i % 100) ) * 0.01f;
            host_K.data()[i] = static_cast<float>( (i % 97) ) * 0.02f;
            host_V.data()[i] = static_cast<float>( (i % 89) ) * 0.03f;
        }

        CudaTensor<TensorDataType::FP16> device_Q( exec_ctx_->getDevice(), input_shape );
        CudaTensor<TensorDataType::FP16> device_K( exec_ctx_->getDevice(), input_shape );
        CudaTensor<TensorDataType::FP16> device_V( exec_ctx_->getDevice(), input_shape );
        CudaTensor<TensorDataType::FP16> device_output( exec_ctx_->getDevice(), output_shape );

        // Copy from FP32 host tensors to FP16 device tensors (runtime conversion expected)
        copy( host_Q, device_Q );
        copy( host_K, device_K );
        copy( host_V, device_V );

        EXPECT_NO_THROW( module->forward( device_Q, device_K, device_V, device_output ) );

        // Copy device FP16 output back to a FP32 host tensor for verification (conversion expected)
        HostTensor host_output = toHost<TensorDataType::FP32>( device_output );
        EXPECT_EQ( host_output.size(), device_output.size() );
        EXPECT_EQ( host_output.shape(), output_shape );
    }
}