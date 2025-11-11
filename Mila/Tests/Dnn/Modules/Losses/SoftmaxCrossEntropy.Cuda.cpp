#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <cuda_runtime.h>

import Mila;

namespace Modules::Losses::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    template<TensorDataType TLogits, TensorDataType TTargets = TensorDataType::INT32>
    struct SoftmaxCrossEntropyCudaTestData
    {
        shape_t logits_shape;
        shape_t targets_shape;
        shape_t output_shape;
        CrossEntropyConfig config;
        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> exec_context;
        std::shared_ptr<SoftmaxCrossEntropy<DeviceType::Cuda, TLogits, TTargets>> module;
        int64_t vocab_size;

        SoftmaxCrossEntropyCudaTestData() : config( 1 ), vocab_size( 0 )
        {
        }

        static SoftmaxCrossEntropyCudaTestData Create(
            const std::string& name,
            const shape_t& logits_shape,
            int64_t vocab_size )
        {
            SoftmaxCrossEntropyCudaTestData data;
            data.logits_shape = logits_shape;
            data.vocab_size = vocab_size;

            data.targets_shape = logits_shape;
            data.targets_shape.pop_back();

            data.output_shape = data.targets_shape;

            data.config = CrossEntropyConfig( vocab_size );
            data.config.withName( name );

            data.exec_context = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            data.module = std::make_shared<SoftmaxCrossEntropy<DeviceType::Cuda, TLogits, TTargets>>(
                data.exec_context, data.config );

            return data;
        }

        static SoftmaxCrossEntropyCudaTestData CreateWithContext(
            const std::string& name,
            const shape_t& logits_shape,
            int64_t vocab_size,
            std::shared_ptr<ExecutionContext<DeviceType::Cuda>> context )
        {
            SoftmaxCrossEntropyCudaTestData data;
            data.logits_shape = logits_shape;
            data.vocab_size = vocab_size;

            data.targets_shape = logits_shape;
            data.targets_shape.pop_back();

            data.output_shape = data.targets_shape;

            data.config = CrossEntropyConfig( vocab_size );
            data.config.withName( name );

            data.exec_context = context;
            data.module = std::make_shared<SoftmaxCrossEntropy<DeviceType::Cuda, TLogits, TTargets>>(
                data.exec_context, data.config );

            return data;
        }
    };

    class SoftmaxCrossEntropyCudaTests : public ::testing::Test
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

            small_shape_ = { 2, 4, 16 };
            medium_shape_ = { 8, 32, 128 };
            large_shape_ = { 16, 64, 512 };

            small_vocab_ = 16;
            medium_vocab_ = 128;
            large_vocab_ = 512;
        }

        auto& SmallFp32Data()
        {
            if (!small_fp32_.module)
            {
                small_fp32_ = SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32>::Create(
                    "small_sce_cuda", small_shape_, small_vocab_ );
            }
            return small_fp32_;
        }

        auto& MediumFp32Data()
        {
            if (!medium_fp32_.module)
            {
                medium_fp32_ = SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32>::Create(
                    "medium_sce_cuda", medium_shape_, medium_vocab_ );
            }
            return medium_fp32_;
        }

        auto& LargeFp32Data()
        {
            if (!large_fp32_.module)
            {
                large_fp32_ = SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32>::Create(
                    "large_sce_cuda", large_shape_, large_vocab_ );
            }
            return large_fp32_;
        }

        auto& SmallFp16Data()
        {
            if (!small_fp16_.module)
            {
                small_fp16_ = SoftmaxCrossEntropyCudaTestData<TensorDataType::FP16>::Create(
                    "small_sce_cuda_fp16", small_shape_, small_vocab_ );
            }
            return small_fp16_;
        }

        bool cuda_available_{ false };

        shape_t small_shape_;
        shape_t medium_shape_;
        shape_t large_shape_;
        int64_t small_vocab_;
        int64_t medium_vocab_;
        int64_t large_vocab_;

        SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32> small_fp32_;
        SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32> medium_fp32_;
        SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32> large_fp32_;
        SoftmaxCrossEntropyCudaTestData<TensorDataType::FP16> small_fp16_;
    };

    template<TensorDataType TLogits, TensorDataType TTargets>
    void TestGetName( const SoftmaxCrossEntropyCudaTestData<TLogits, TTargets>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.module->getName(), expected_name );
    }

    template<TensorDataType TLogits, TensorDataType TTargets>
    void TestDeviceType( const SoftmaxCrossEntropyCudaTestData<TLogits, TTargets>& data )
    {
        EXPECT_EQ( data.module->getDeviceType(), DeviceType::Cuda );
        ASSERT_NE( data.exec_context, nullptr );

        auto device = data.exec_context->getDevice();
        ASSERT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), DeviceType::Cuda );
    }

    template<TensorDataType TLogits, TensorDataType TTargets>
    void TestIsBuilt( const SoftmaxCrossEntropyCudaTestData<TLogits, TTargets>& data, bool expected_built )
    {
        EXPECT_EQ( data.module->isBuilt(), expected_built );
    }

    template<TensorDataType TLogits, TensorDataType TTargets>
    void TestBuild( SoftmaxCrossEntropyCudaTestData<TLogits, TTargets>& data )
    {
        EXPECT_NO_THROW( data.module->build( data.logits_shape ) );
        EXPECT_TRUE( data.module->isBuilt() );

        data.module->build( data.logits_shape );
        EXPECT_TRUE( data.module->isBuilt() );
    }

    template<TensorDataType TLogits, TensorDataType TTargets>
    void TestParameterCount( const SoftmaxCrossEntropyCudaTestData<TLogits, TTargets>& data )
    {
        EXPECT_EQ( data.module->parameterCount(), 0 );
    }

    template<TensorDataType TLogits, TensorDataType TTargets>
    void TestGetParameters( const SoftmaxCrossEntropyCudaTestData<TLogits, TTargets>& data )
    {
        auto params = data.module->getParameters();
        EXPECT_EQ( params.size(), 0 );
    }

    template<TensorDataType TLogits, TensorDataType TTargets>
    void TestToString( const SoftmaxCrossEntropyCudaTestData<TLogits, TTargets>& data )
    {
        std::string output = data.module->toString();

        EXPECT_NE( output.find( "SoftmaxCrossEntropy" ), std::string::npos );
        EXPECT_NE( output.find( data.config.getName() ), std::string::npos );
        EXPECT_NE( output.find( "Vocabulary Size:" ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
    }

    template<TensorDataType TLogits, TensorDataType TTargets>
    void TestForward( SoftmaxCrossEntropyCudaTestData<TLogits, TTargets>& data )
    {
        using LogitsTensorType = CudaTensor<TLogits>;
        using TargetsTensorType = CudaTensor<TTargets>;
        using HostLogitsTensorType = CpuTensor<TensorDataType::FP32>;
        using HostTargetsTensorType = CpuTensor<TensorDataType::INT32>;

        data.module->build( data.logits_shape );

        HostLogitsTensorType host_logits( "CPU", data.logits_shape );
        HostTargetsTensorType host_targets( "CPU", data.targets_shape );

        random( host_logits, -2.0f, 2.0f );

        int64_t batch_size = data.targets_shape[0];
        int64_t seq_len = data.targets_shape[1];
        for (int64_t i = 0; i < batch_size * seq_len; ++i)
        {
            host_targets.data()[i] = static_cast<int32_t>( rand() % data.vocab_size );
        }

        LogitsTensorType device_logits( "CUDA:0", data.logits_shape );
        TargetsTensorType device_targets( "CUDA:0", data.targets_shape );
        LogitsTensorType device_output( "CUDA:0", data.output_shape );

        copy( host_logits, device_logits );
        copy( host_targets, device_targets );

        EXPECT_NO_THROW( data.module->forward( device_logits, device_targets, device_output ) );
        EXPECT_EQ( device_output.size(), data.output_shape[0] * data.output_shape[1] );
        EXPECT_EQ( device_output.shape(), data.output_shape );

        HostLogitsTensorType host_output = toHost<TensorDataType::FP32>( device_output );
        EXPECT_EQ( host_output.size(), device_output.size() );

        for (size_t i = 0; i < host_output.size(); ++i)
        {
            EXPECT_GE( host_output.data()[i], 0.0f ) << "Loss should be non-negative";
        }
    }

    template<TensorDataType TLogits, TensorDataType TTargets>
    void TestBackward( SoftmaxCrossEntropyCudaTestData<TLogits, TTargets>& data )
    {
        using LogitsTensorType = CudaTensor<TLogits>;
        using TargetsTensorType = CudaTensor<TTargets>;
        using HostLogitsTensorType = CpuTensor<TensorDataType::FP32>;
        using HostTargetsTensorType = CpuTensor<TensorDataType::INT32>;

        data.module->setTraining( true );
        data.module->build( data.logits_shape );

        HostLogitsTensorType host_logits( "CPU", data.logits_shape );
        HostTargetsTensorType host_targets( "CPU", data.targets_shape );
        HostLogitsTensorType host_output_grad( "CPU", data.output_shape );

        random( host_logits, -2.0f, 2.0f );
        fill( host_output_grad, 1.0f );

        int64_t batch_size = data.targets_shape[0];
        int64_t seq_len = data.targets_shape[1];
        for (int64_t i = 0; i < batch_size * seq_len; ++i)
        {
            host_targets.data()[i] = static_cast<int32_t>( rand() % data.vocab_size );
        }

        LogitsTensorType device_logits( "CUDA:0", data.logits_shape );
        TargetsTensorType device_targets( "CUDA:0", data.targets_shape );
        LogitsTensorType device_output( "CUDA:0", data.output_shape );
        LogitsTensorType device_output_grad( "CUDA:0", data.output_shape );
        LogitsTensorType device_input_grad( "CUDA:0", data.logits_shape );

        copy( host_logits, device_logits );
        copy( host_targets, device_targets );
        copy( host_output_grad, device_output_grad );
        zeros( device_input_grad );

        data.module->forward( device_logits, device_targets, device_output );

        EXPECT_NO_THROW(
            data.module->backward( device_logits, device_targets, device_output_grad, device_input_grad )
        ) << "Backward pass should succeed for CUDA SoftmaxCrossEntropy operation in training mode";

        EXPECT_EQ( device_input_grad.shape(), data.logits_shape );

        HostLogitsTensorType host_input_grad = toHost<TensorDataType::FP32>( device_input_grad );
        EXPECT_EQ( host_input_grad.size(), device_input_grad.size() );

        bool has_nonzero_grad = false;
        for (size_t i = 0; i < host_input_grad.size(); ++i)
        {
            if (std::abs( host_input_grad.data()[i] ) > 1e-6f)
            {
                has_nonzero_grad = true;
                break;
            }
        }
        EXPECT_TRUE( has_nonzero_grad ) << "Input gradients should contain non-zero values";
    }

    // ====================================================================
    // Module Interface Tests
    // ====================================================================

    TEST_F( SoftmaxCrossEntropyCudaTests, GetName )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetName( SmallFp32Data(), "small_sce_cuda" );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, DeviceType )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestDeviceType( SmallFp32Data() );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, IsBuilt_BeforeBuild )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestIsBuilt( SmallFp32Data(), false );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, IsBuilt_AfterBuild )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_FALSE( data.module->isBuilt() );

        data.module->build( data.logits_shape );

        EXPECT_TRUE( data.module->isBuilt() );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Build )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestBuild( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, ParameterCount )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestParameterCount( SmallFp32Data() );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, GetParameters )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        TestGetParameters( SmallFp32Data() );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, ToString )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestToString( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, GetVocabSize )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        EXPECT_EQ( data.module->getVocabSize(), data.vocab_size );
    }

    // ====================================================================
    // Forward Pass Tests
    // ====================================================================

    TEST_F( SoftmaxCrossEntropyCudaTests, Forward_SmallShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestForward( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Forward_MediumShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        TestForward( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Forward_LargeShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LargeFp32Data();
        TestForward( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Forward_FP16 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp16Data();
        TestForward( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, WithContext_Construction )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        auto data = SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32>::CreateWithContext(
            "context_sce_cuda", small_shape_, small_vocab_, ctx );

        EXPECT_EQ( data.module->getName(), "context_sce_cuda" );
        EXPECT_EQ( data.exec_context, ctx );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, EdgeCase_MinimalShape )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 1, 1, 2 };

        auto data = SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32>::Create(
            "minimal_cuda", shape, 2 );

        TestForward( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, EdgeCase_BatchSize1 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 1, 8, 16 };

        auto data = SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32>::Create(
            "batch1_cuda", shape, 16 );

        TestForward( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Error_NullExecutionContext )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        CrossEntropyConfig config( 16 );
        config.withName( "test_cuda" );

        std::shared_ptr<ExecutionContext<DeviceType::Cuda>> null_ctx;

        EXPECT_THROW(
            (std::make_shared<SoftmaxCrossEntropy<DeviceType::Cuda, TensorDataType::FP32>>( null_ctx, config )),
            std::invalid_argument
        );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Error_ForwardBeforeBuild )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32>::Create(
            "unbuild_cuda", small_shape_, small_vocab_ );

        CudaTensor<TensorDataType::FP32> logits( "CUDA:0", data.logits_shape );
        CudaTensor<TensorDataType::INT32> targets( "CUDA:0", data.targets_shape );
        CudaTensor<TensorDataType::FP32> output( "CUDA:0", data.output_shape );

        EXPECT_THROW(
            data.module->forward( logits, targets, output ),
            std::runtime_error
        );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Error_ShapeMismatch_VocabSize )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        shape_t wrong_shape = { 2, 4, 32 };

        EXPECT_THROW(
            data.module->build( wrong_shape ),
            std::invalid_argument
        );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Synchronize )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_NO_THROW( data.module->synchronize() );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, SetTrainingMode )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_FALSE( data.module->isTraining() );

        data.module->setTraining( true );
        EXPECT_TRUE( data.module->isTraining() );

        data.module->setTraining( false );
        EXPECT_FALSE( data.module->isTraining() );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, MultipleForwardCalls )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        data.module->build( data.logits_shape );

        CpuTensor<TensorDataType::FP32> host_logits( "CPU", data.logits_shape );
        CpuTensor<TensorDataType::INT32> host_targets( "CPU", data.targets_shape );
        CudaTensor<TensorDataType::FP32> device_logits( "CUDA:0", data.logits_shape );
        CudaTensor<TensorDataType::INT32> device_targets( "CUDA:0", data.targets_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.output_shape );

        for (int iter = 0; iter < 10; ++iter)
        {
            random( host_logits, -2.0f, 2.0f );

            int64_t batch_size = data.targets_shape[0];
            int64_t seq_len = data.targets_shape[1];
            for (int64_t i = 0; i < batch_size * seq_len; ++i)
            {
                host_targets.data()[i] = static_cast<int32_t>( rand() % data.vocab_size );
            }

            copy( host_logits, device_logits );
            copy( host_targets, device_targets );

            EXPECT_NO_THROW( data.module->forward( device_logits, device_targets, device_output ) );
        }
    }

    // ====================================================================
    // Backward Pass Tests
    // ====================================================================

    TEST_F( SoftmaxCrossEntropyCudaTests, Backward_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();
        TestBackward( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Backward_FP16 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp16Data();
        TestBackward( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Backward_MediumShape_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = MediumFp32Data();
        TestBackward( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Backward_LargeShape_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = LargeFp32Data();
        TestBackward( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Error_BackwardBeforeBuild_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32>::Create(
            "unbuild_backward_cuda", small_shape_, small_vocab_ );

        CudaTensor<TensorDataType::FP32> logits( "CUDA:0", data.logits_shape );
        CudaTensor<TensorDataType::INT32> targets( "CUDA:0", data.targets_shape );
        CudaTensor<TensorDataType::FP32> output_grad( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> input_grad( "CUDA:0", data.logits_shape );

        EXPECT_THROW(
            data.module->backward( logits, targets, output_grad, input_grad ),
            std::runtime_error
        );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Backward_EdgeCase_MinimalShape_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 1, 1, 2 };

        auto data = SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32>::Create(
            "minimal_backward_cuda", shape, 2 );

        TestBackward( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Backward_EdgeCase_BatchSize1_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        shape_t shape = { 1, 8, 16 };

        auto data = SoftmaxCrossEntropyCudaTestData<TensorDataType::FP32>::Create(
            "batch1_backward_cuda", shape, 16 );

        TestBackward( data );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Backward_MultipleIterations_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        data.module->setTraining( true );
        data.module->build( data.logits_shape );

        CpuTensor<TensorDataType::FP32> host_logits( "CPU", data.logits_shape );
        CpuTensor<TensorDataType::INT32> host_targets( "CPU", data.targets_shape );
        CpuTensor<TensorDataType::FP32> host_output_grad( "CPU", data.output_shape );

        CudaTensor<TensorDataType::FP32> device_logits( "CUDA:0", data.logits_shape );
        CudaTensor<TensorDataType::INT32> device_targets( "CUDA:0", data.targets_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_output_grad( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_input_grad( "CUDA:0", data.logits_shape );

        for (int iter = 0; iter < 5; ++iter)
        {
            random( host_logits, -2.0f, 2.0f );
            fill( host_output_grad, 1.0f );

            int64_t batch_size = data.targets_shape[0];
            int64_t seq_len = data.targets_shape[1];
            for (int64_t i = 0; i < batch_size * seq_len; ++i)
            {
                host_targets.data()[i] = static_cast<int32_t>( rand() % data.vocab_size );
            }

            copy( host_logits, device_logits );
            copy( host_targets, device_targets );
            copy( host_output_grad, device_output_grad );
            zeros( device_input_grad );

            data.module->forward( device_logits, device_targets, device_output );

            EXPECT_NO_THROW(
                data.module->backward( device_logits, device_targets, device_output_grad, device_input_grad )
            ) << "Backward iteration " << iter << " failed";
        }
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Training_InferenceToTrainingToInference_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        EXPECT_FALSE( data.module->isTraining() );
        data.module->build( data.logits_shape );

        CudaTensor<TensorDataType::FP32> device_logits( "CUDA:0", data.logits_shape );
        CudaTensor<TensorDataType::INT32> device_targets( "CUDA:0", data.targets_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_output_grad( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_input_grad( "CUDA:0", data.logits_shape );

        CpuTensor<TensorDataType::FP32> host_logits( "CPU", data.logits_shape );
        CpuTensor<TensorDataType::INT32> host_targets( "CPU", data.targets_shape );
        random( host_logits, -2.0f, 2.0f );

        int64_t batch_size = data.targets_shape[0];
        int64_t seq_len = data.targets_shape[1];
        for (int64_t i = 0; i < batch_size * seq_len; ++i)
        {
            host_targets.data()[i] = static_cast<int32_t>( rand() % data.vocab_size );
        }

        copy( host_logits, device_logits );
        copy( host_targets, device_targets );

        EXPECT_NO_THROW( data.module->forward( device_logits, device_targets, device_output ) );

        EXPECT_THROW(
            data.module->backward( device_logits, device_targets, device_output_grad, device_input_grad ),
            std::runtime_error
        ) << "Backward should fail in inference mode";

        data.module->setTraining( true );
        EXPECT_TRUE( data.module->isTraining() );

        EXPECT_NO_THROW( data.module->forward( device_logits, device_targets, device_output ) );

        CpuTensor<TensorDataType::FP32> host_output_grad( "CPU", data.output_shape );
        fill( host_output_grad, 1.0f );
        copy( host_output_grad, device_output_grad );
        zeros( device_input_grad );

        EXPECT_NO_THROW(
            data.module->backward( device_logits, device_targets, device_output_grad, device_input_grad )
        ) << "Backward should work after switching to training mode";

        data.module->setTraining( false );
        EXPECT_FALSE( data.module->isTraining() );

        EXPECT_NO_THROW( data.module->forward( device_logits, device_targets, device_output ) );

        EXPECT_THROW(
            data.module->backward( device_logits, device_targets, device_output_grad, device_input_grad ),
            std::runtime_error
        ) << "Backward should fail after switching back to inference mode";
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Training_EnableBeforeBuild_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        data.module->setTraining( true );
        EXPECT_TRUE( data.module->isTraining() );

        data.module->build( data.logits_shape );

        CudaTensor<TensorDataType::FP32> device_logits( "CUDA:0", data.logits_shape );
        CudaTensor<TensorDataType::INT32> device_targets( "CUDA:0", data.targets_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_output_grad( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_input_grad( "CUDA:0", data.logits_shape );

        CpuTensor<TensorDataType::FP32> host_logits( "CPU", data.logits_shape );
        CpuTensor<TensorDataType::INT32> host_targets( "CPU", data.targets_shape );
        CpuTensor<TensorDataType::FP32> host_output_grad( "CPU", data.output_shape );
        random( host_logits, -2.0f, 2.0f );
        fill( host_output_grad, 1.0f );

        int64_t batch_size = data.targets_shape[0];
        int64_t seq_len = data.targets_shape[1];
        for (int64_t i = 0; i < batch_size * seq_len; ++i)
        {
            host_targets.data()[i] = static_cast<int32_t>( rand() % data.vocab_size );
        }

        copy( host_logits, device_logits );
        copy( host_targets, device_targets );
        copy( host_output_grad, device_output_grad );
        zeros( device_input_grad );

        EXPECT_NO_THROW( data.module->forward( device_logits, device_targets, device_output ) );
        EXPECT_NO_THROW( data.module->backward( device_logits, device_targets, device_output_grad, device_input_grad ) );
    }

    TEST_F( SoftmaxCrossEntropyCudaTests, Error_BackwardInInferenceMode_FP32 )
    {
        if (!cuda_available_)
        {
            GTEST_SKIP() << "CUDA not available";
        }

        auto data = SmallFp32Data();

        data.module->build( data.logits_shape );
        EXPECT_FALSE( data.module->isTraining() );

        CudaTensor<TensorDataType::FP32> device_logits( "CUDA:0", data.logits_shape );
        CudaTensor<TensorDataType::INT32> device_targets( "CUDA:0", data.targets_shape );
        CudaTensor<TensorDataType::FP32> device_output( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_output_grad( "CUDA:0", data.output_shape );
        CudaTensor<TensorDataType::FP32> device_input_grad( "CUDA:0", data.logits_shape );

        CpuTensor<TensorDataType::FP32> host_logits( "CPU", data.logits_shape );
        CpuTensor<TensorDataType::INT32> host_targets( "CPU", data.targets_shape );
        random( host_logits, -2.0f, 2.0f );

        int64_t batch_size = data.targets_shape[0];
        int64_t seq_len = data.targets_shape[1];
        for (int64_t i = 0; i < batch_size * seq_len; ++i)
        {
            host_targets.data()[i] = static_cast<int32_t>( rand() % data.vocab_size );
        }

        copy( host_logits, device_logits );
        copy( host_targets, device_targets );

        data.module->forward( device_logits, device_targets, device_output );

        EXPECT_THROW(
            data.module->backward( device_logits, device_targets, device_output_grad, device_input_grad ),
            std::runtime_error
        ) << "Backward should throw when module is not in training mode";
    }
}