/**
 * @file Sampling.Cuda.cpp
 * @brief Injected-r oracle for the device token sampler (CudaSamplingOp<FP32>).
 *
 * The TokenSampler facade owns its RNG, so the rigorous sampling checks live at the
 * op layer, where forward() takes the host-drawn uniform `r` as an explicit argument.
 * Rather than a fragile device-vs-double-host exact match (FP32 reduction order differs
 * from a double reference), these lock the algorithm by property: greedy = argmax
 * (exact), inverse-CDF boundaries, determinism, top-k / top-p support restriction, and
 * softcap monotonicity. Small, well-separated synthetic logits keep every expectation
 * unambiguous. See Specifications/TokenSampling.md section 7.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <vector>

import Mila;
import Compute.ExecutionContext;
import Compute.CudaSamplingOp;
import Dnn.Samplers.SamplingConfig;
import Dnn.GenerateParams;

namespace Mila::Tests::Dnn::Samplers
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Compute::Cuda::Sampling;

    class SamplingCudaTests : public ::testing::Test
    {
    protected:
        static constexpr int64_t kVocab = 8;

        using OpType = CudaSamplingOp<TensorDataType::FP32>;
        using DeviceLogits = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
        using DeviceToken = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>;
        using HostToken = Tensor<TensorDataType::INT32, CpuMemoryResource>;

        void SetUp() override
        {
            try
            {
                ctx_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                ctx_ = nullptr;
            }

            if ( !ctx_ )
            {
                GTEST_SKIP() << "CUDA device not available";
            }
        }

        std::unique_ptr<OpType> makeOp( float softcap = 0.0f )
        {
            SamplingConfig config = SamplingConfig{}
                .withVocabularySize( kVocab )
                .withFinalLogitSoftcap( softcap );

            return std::make_unique<OpType>( ctx_.get(), config );
        }

        DeviceLogits deviceLogits( const std::vector<float>& values )
        {
            const shape_t shape{ 1, 1, kVocab };

            Tensor<TensorDataType::FP32, CpuMemoryResource> host( Device::Cpu(), shape );
            for ( int64_t i = 0; i < kVocab; ++i )
                host.data()[ i ] = values[ static_cast<size_t>( i ) ];

            DeviceLogits device( Device::Cuda( 0 ), shape );
            copy( host, device, ctx_.get() );
            ctx_->synchronize();

            return device;
        }

        // Runs the op and reads the sampled token back. The op samples on the default
        // stream; a context-free copy is a synchronous default-stream readback (mirrors
        // the TokenSampler facade), so the host value is valid without an extra sync.
        int32_t sample( OpType& op, const DeviceLogits& logits, const SamplingParams& params, float r )
        {
            DeviceToken token_device( Device::Cuda( 0 ), shape_t{ 1, 1 } );
            op.forward( logits, token_device, params, r );

            HostToken token_host( Device::Cpu(), shape_t{ 1, 1 } );
            copy( token_device, token_host );

            return token_host.data()[ 0 ];
        }

        std::unique_ptr<IExecutionContext> ctx_;
    };

    // Greedy (temperature <= 0) is an exact argmax.
    TEST_F( SamplingCudaTests, Greedy_PicksArgmax )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 2, 3, 4, 5, 6, 7, 8 } );

        SamplingParams params;
        params.temperature = 0.0f;

        EXPECT_EQ( sample( *op, logits, params, 0.5f ), 7 );
    }

    // top_k == 1 is also greedy (argmax), independent of the max position.
    TEST_F( SamplingCudaTests, TopK1_PicksArgmax )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 8, 2, 3, 4, 5, 6, 7, 1 } );

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 1;

        EXPECT_EQ( sample( *op, logits, params, 0.99f ), 0 );
    }

    // Full-multinomial inverse-CDF boundaries: r -> 0 selects the lowest-index
    // positive-probability token; r -> 1 selects the last.
    TEST_F( SamplingCudaTests, FullMultinomial_BoundaryR )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 2, 3, 4, 5, 6, 7, 8 } );

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 0;
        params.top_p = 1.0f;

        EXPECT_EQ( sample( *op, logits, params, 0.0f ), 0 );
        EXPECT_EQ( sample( *op, logits, params, 0.999999f ), 7 );
    }

    // Deterministic: identical (logits, params, r) -> identical token.
    TEST_F( SamplingCudaTests, Stochastic_Deterministic )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 5, 2, 8, 3, 6, 4, 7 } );

        SamplingParams params;
        params.temperature = 0.8f;
        params.top_k = 0;
        params.top_p = 1.0f;

        const int32_t first = sample( *op, logits, params, 0.42f );
        const int32_t second = sample( *op, logits, params, 0.42f );

        EXPECT_EQ( first, second );
    }

    // Top-k restricts the support: with the top-2 logits far above the rest, only those
    // two indices carry non-negligible probability across an r sweep.
    TEST_F( SamplingCudaTests, TopK_RestrictsSupport )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 2, 3, 4, 5, 6, 70, 80 } );

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 2;
        params.top_p = 1.0f;

        for ( int i = 0; i < 20; ++i )
        {
            const float r = static_cast<float>( i ) / 20.0f;
            const int32_t token = sample( *op, logits, params, r );

            EXPECT_TRUE( token == 6 || token == 7 )
                << "top_k=2 selected out-of-support token " << token << " at r=" << r;
        }
    }

    // Top-p (nucleus) restricts the support: with one dominant token the 0.5 nucleus is
    // just that token, so every r maps to it.
    TEST_F( SamplingCudaTests, TopP_RestrictsSupport )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 0, 0, 0, 0, 0, 0, 0, 20 } );

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 0;
        params.top_p = 0.5f;

        for ( int i = 0; i < 20; ++i )
        {
            const float r = static_cast<float>( i ) / 20.0f;
            EXPECT_EQ( sample( *op, logits, params, r ), 7 )
                << "top_p=0.5 should select only the dominant token";
        }
    }

    // Softcap squashes a huge outlier but is monotonic, so greedy argmax is preserved.
    TEST_F( SamplingCudaTests, Softcap_PreservesArgmax )
    {
        auto op = makeOp( 30.0f );
        auto logits = deviceLogits( { 1, 2, 3, 4, 5, 6, 7, 1000 } );

        SamplingParams params;
        params.temperature = 0.0f;

        EXPECT_EQ( sample( *op, logits, params, 0.5f ), 7 );
    }
}
