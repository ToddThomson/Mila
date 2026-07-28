/**
 * @file AdamW.MixedPrecision.Cuda.cpp
 * @brief Contract tests for CudaAdamWOptimizer's FP32 master-parameter path (BF16 / FP16).
 *
 * Net-new coverage, not a revival. The mixed-precision master path had no test in any
 * file, active or disabled -- AdamW.Cuda.cpp is 40 FP32 cases and never mentions masters --
 * which is why a master copy that was allocated and then zeroed survived unnoticed.
 *
 * The contract, from the kernel (CudaAdamW.cu): when a master exists it is the
 * authoritative parameter value, not a mirror. The kernel reads
 *   old_param = master ? master[idx] : (float)params[idx]
 * applies the update to that, writes the narrowed result back to the parameter and the
 * full-precision result back to the master. A master starting at zero therefore discards
 * the initialized weights on the very first step and trains the model from zero.
 *
 * The oracle below is chosen so a correct optimizer is exactly a no-op: zero gradients
 * and zero weight decay make `param = old_param - lr * (0 / (0 + eps) + 0 * old_param)`,
 * which is old_param. Anything that moves is wrong, and the specific defect moves it all
 * the way to zero.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <vector>

import Mila;
import Compute.ExecutionContext;

namespace Mila::Tests::Dnn::Optimizers
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Optimizers;

    namespace
    {
        // The device-agnostic wrapper, as AdamW.Cpu.cpp uses: it is what the umbrella
        // exports (Compute.CudaAdamWOptimizer is not public) and it takes IExecutionContext*,
        // so the test drives the same surface a consumer would.
        using AdamWBf16Cuda = AdamWOptimizer<DeviceType::Cuda, TensorDataType::BF16>;
    }

    class AdamWMixedPrecisionCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            try
            {
                context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                context_ = nullptr;
            }
        }

        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // Fills a device BF16 tensor from host FP32 values, converting on the way in.
        DeviceBf16 makeDeviceTensor( const shape_t& shape, float value )
        {
            HostFp32 host( Device::Cpu(), shape );
            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = value;
            }

            DeviceBf16 device( Device::Cuda( 0 ), shape );
            copy( host, device, context_.get() );
            context_->synchronize();

            return device;
        }

        std::vector<float> readBack( const DeviceBf16& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, context_.get() );
            context_->synchronize();

            return std::vector<float>( host.data(), host.data() + host.size() );
        }

        std::unique_ptr<IExecutionContext> context_;
    };

#define SKIP_WITHOUT_CUDA()                                            \
    if ( context_ == nullptr )                                         \
    {                                                                  \
        GTEST_SKIP() << "CUDA device not available.";                  \
    }

    // The regression guard for the zeroed master. With no gradient and no weight decay a
    // correct step is a no-op; a master initialized to zero instead drives every element
    // to zero on step one.
    TEST_F( AdamWMixedPrecisionCudaTests, Step_WithZeroGradient_PreservesBf16Parameters )
    {
        SKIP_WITHOUT_CUDA();

        constexpr float kInitial = 1.0f;   // exactly representable in BF16
        const shape_t shape{ 64, 32 };

        auto param = makeDeviceTensor( shape, kInitial );
        auto grad = makeDeviceTensor( shape, 0.0f );

        auto config = AdamWConfig()
            .withLearningRate( 0.01f )
            .withBeta1( 0.9f )
            .withBeta2( 0.999f )
            .withWeightDecay( 0.0f );

        auto optimizer = std::make_shared<AdamWBf16Cuda>( context_.get(), config );

        optimizer->addParameter( &param, &grad );
        optimizer->step();
        context_->synchronize();

        auto values = readBack( param );
        ASSERT_EQ( values.size(), static_cast<size_t>( 64 * 32 ) );

        for ( size_t i = 0; i < values.size(); ++i )
        {
            // A zeroed master collapses this to 0.0 -- the assertion the defect fails.
            ASSERT_NEAR( values[ i ], kInitial, 1e-2f )
                << "parameter moved on a zero-gradient, zero-decay step at index " << i;
        }
    }

    // With weight decay on and still no gradient, the update reduces to pure decay:
    // param = old * (1 - lr * wd). That pins the master as the value the decay is applied
    // TO -- decaying a zeroed master would leave zero rather than a slightly shrunk value.
    TEST_F( AdamWMixedPrecisionCudaTests, Step_WithWeightDecayOnly_ShrinksFromTheLiveValue )
    {
        SKIP_WITHOUT_CUDA();

        constexpr float kInitial = 1.0f;
        constexpr float kLearningRate = 0.1f;
        constexpr float kWeightDecay = 0.1f;
        const shape_t shape{ 128 };

        auto param = makeDeviceTensor( shape, kInitial );
        auto grad = makeDeviceTensor( shape, 0.0f );

        auto config = AdamWConfig()
            .withLearningRate( kLearningRate )
            .withBeta1( 0.9f )
            .withBeta2( 0.999f )
            .withWeightDecay( kWeightDecay );

        auto optimizer = std::make_shared<AdamWBf16Cuda>( context_.get(), config );

        optimizer->addParameter( &param, &grad );
        optimizer->step();
        context_->synchronize();

        const float expected = kInitial * ( 1.0f - kLearningRate * kWeightDecay );

        auto values = readBack( param );

        for ( size_t i = 0; i < values.size(); ++i )
        {
            // BF16 carries 8 mantissa bits and the writeback is stochastically rounded,
            // so allow a couple of ulps around 1.0 while still excluding both 0.0 and an
            // undecayed 1.0.
            ASSERT_NEAR( values[ i ], expected, 2e-2f ) << "at index " << i;
        }
    }

    // Repeated zero-gradient steps must stay stable: the master is written back each step,
    // so an initialization defect compounds rather than washing out.
    TEST_F( AdamWMixedPrecisionCudaTests, RepeatedSteps_WithZeroGradient_DoNotDrift )
    {
        SKIP_WITHOUT_CUDA();

        constexpr float kInitial = 0.5f;   // exactly representable in BF16
        const shape_t shape{ 256 };

        auto param = makeDeviceTensor( shape, kInitial );
        auto grad = makeDeviceTensor( shape, 0.0f );

        auto config = AdamWConfig()
            .withLearningRate( 0.01f )
            .withBeta1( 0.9f )
            .withBeta2( 0.999f )
            .withWeightDecay( 0.0f );

        auto optimizer = std::make_shared<AdamWBf16Cuda>( context_.get(), config );

        optimizer->addParameter( &param, &grad );

        for ( int i = 0; i < 5; ++i )
        {
            optimizer->step();
        }

        context_->synchronize();

        auto values = readBack( param );

        for ( size_t i = 0; i < values.size(); ++i )
        {
            ASSERT_NEAR( values[ i ], kInitial, 2e-2f )
                << "drift after five no-op steps at index " << i;
        }
    }

#undef SKIP_WITHOUT_CUDA
}
