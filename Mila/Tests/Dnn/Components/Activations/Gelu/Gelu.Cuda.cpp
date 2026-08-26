/**
 * @file Gelu.Cuda.cpp
 * @brief Concrete-component tests for Gelu<DeviceType::Cuda, FP32>.
 *
 * The CUDA companion to Gelu.Cpu.cpp (see Specifications/Testing.md). CUDA GeluOp
 * has a single working precision today -- the kernel dispatch (cuda_gelu_impl)
 * supports only float/half, so FP32 is the one precision the component can
 * instantiate -- so by the methodology this file is explicit (pragmatic), not
 * precision-parameterized. The TYPED_TEST mechanism appears first on a component
 * that genuinely supports more than one CUDA precision.
 *
 * Compiled only under MILA_ENABLE_CUDA (the Tests/CMakeLists.txt CUDA block), so
 * no #ifdef. CUDA may still be absent at runtime; SetUp() skips if no device.
 *
 * Numerics stage inputs host->device (copy) and read outputs device->host
 * (toHost). FP32 is exact host-side, so the reference is taken from the original
 * host input. Base-contract behavior is covered once in Core/Component.cpp and is
 * not retested here.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::Activations::Gelu
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using GeluCuda = Mila::Dnn::Gelu<DeviceType::Cuda, TensorDataType::FP32>;
        using DeviceTensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // GELU tanh-approximation reference, computed independently in float.
        constexpr float kSqrt2OverPi = 0.7978845608f;
        constexpr float kGeluCoeff = 0.044715f;

        float geluReference( float x )
        {
            const float x_cubed = x * x * x;

            return 0.5f * x * (1.0f + std::tanh( kSqrt2OverPi * (x + kGeluCoeff * x_cubed) ));
        }

        float geluGradientReference( float x )
        {
            const float x_squared = x * x;
            const float arg = kSqrt2OverPi * (x + kGeluCoeff * x * x_squared);
            const float tanh_value = std::tanh( arg );
            const float sech_squared = 1.0f - tanh_value * tanh_value;
            const float d_arg = kSqrt2OverPi * (1.0f + 3.0f * kGeluCoeff * x_squared);

            return 0.5f * (1.0f + tanh_value) + 0.5f * x * sech_squared * d_arg;
        }

        static_assert( GeluCuda::getDeviceType() == DeviceType::Cuda );
        static_assert( GeluCuda::getPrecision() == TensorDataType::FP32 );
    }

    class GeluCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            cpu_context_ = createExecutionContext( Device::Cpu() );

            try
            {
                cuda_context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                cuda_context_ = nullptr;
            }

            if ( !cuda_context_ )
            {
                GTEST_SKIP() << "CUDA device not available";
            }
        }

        HostTensor makeSpreadHost( const shape_t& shape )
        {
            HostTensor host( cpu_context_->getDeviceId(), shape );
            auto* data = host.data();

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                data[ i ] = static_cast<float>( i ) / host.size() * 4.0f - 2.0f;
            }

            return host;
        }

        DeviceTensor toDevice( const HostTensor& host, const shape_t& shape )
        {
            DeviceTensor device( cuda_context_->getDeviceId(), shape );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        std::unique_ptr<IExecutionContext> cpu_context_;
        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    // ====================================================================
    // A. Construction
    // ====================================================================

    TEST_F( GeluCudaTests, Construct_StandaloneSucceeds )
    {
        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( gelu.getApproximationMethod(), ApproximationMethod::Tanh );
        EXPECT_EQ( gelu.getDeviceId().type, DeviceType::Cuda );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( GeluCudaTests, Forward_MatchesReference )
    {
        const shape_t shape{ 2, 3, 4 };

        auto host_in = makeSpreadHost( shape );
        auto device_in = toDevice( host_in, shape );

        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );
        gelu.build( BuildContext( shape, RuntimeMode::Inference ) );

        auto& device_out = gelu.forward( device_in );
        gelu.synchronize();

        auto host_out = toHost<TensorDataType::FP32>( device_out, cuda_context_.get() );

        ASSERT_EQ( host_out.size(), host_in.size() );

        constexpr float tolerance = 1e-4f;

        for ( dim_t i = 0; i < host_out.size(); ++i )
        {
            const float expected = geluReference( host_in.data()[ i ] );

            EXPECT_NEAR( host_out.data()[ i ], expected, tolerance )
                << "forward mismatch at index " << i << " input=" << host_in.data()[ i ];
        }
    }

    // ====================================================================
    // F. Backward (numeric vs analytic gradient)
    // ====================================================================

    TEST_F( GeluCudaTests, Backward_MatchesGradientReference )
    {
        const shape_t shape{ 2, 3, 4 };

        auto host_in = makeSpreadHost( shape );

        HostTensor host_grad( cpu_context_->getDeviceId(), shape );
        for ( dim_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = 1.0f;
        }

        auto device_in = toDevice( host_in, shape );
        auto device_grad = toDevice( host_grad, shape );

        // Training build allocates the input-gradient buffer backward needs.
        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );
        gelu.build( BuildContext( shape, RuntimeMode::Training ) );

        gelu.forward( device_in );
        auto& device_in_grad = gelu.backward( device_in, device_grad );
        gelu.synchronize();

        auto host_in_grad = toHost<TensorDataType::FP32>( device_in_grad, cuda_context_.get() );

        ASSERT_EQ( host_in_grad.size(), host_in.size() );

        constexpr float tolerance = 1e-3f;

        for ( dim_t i = 0; i < host_in_grad.size(); ++i )
        {
            const float expected = geluGradientReference( host_in.data()[ i ] ) * host_grad.data()[ i ];

            EXPECT_NEAR( host_in_grad.data()[ i ], expected, tolerance )
                << "backward mismatch at index " << i << " input=" << host_in.data()[ i ];
        }
    }

    // ====================================================================
    // G. Parameters & Gradients (Gelu is stateless)
    // ====================================================================

    TEST_F( GeluCudaTests, Parameters_AreEmpty )
    {
        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );
        gelu.build( BuildContext( shape_t{ 2, 4 }, RuntimeMode::Inference ) );

        EXPECT_EQ( gelu.parameterCount(), 0 );
        EXPECT_TRUE( gelu.getParameters().empty() );
        EXPECT_TRUE( gelu.getGradients().empty() );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( GeluCudaTests, GetType_IsGelu )
    {
        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( gelu.getType(), ComponentType::Gelu );
    }

    // ====================================================================
    // K. Observability (Observability.md sections 6.3, 6.4)
    // ====================================================================

    TEST_F( GeluCudaTests, Observation_DescribesItsOutputAndStage )
    {
        const shape_t shape{ 2, 3, 4 };

        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );

        // Before build there is no allocation to describe, and empty is the whole answer.
        EXPECT_TRUE( gelu.getOutputs().empty() );

        gelu.build( BuildContext( shape, RuntimeMode::Inference ) );

        const auto outputs = gelu.getOutputs();

        ASSERT_EQ( outputs.size(), 1u );
        EXPECT_EQ( outputs[ 0 ]->shape(), shape );
        EXPECT_EQ( outputs[ 0 ]->getDataType(), TensorDataType::FP32 );

        // The stage vocabulary is derived from the tensor's own name, not invented here.
        const auto stages = gelu.getObservableStages();

        ASSERT_EQ( stages.size(), 1u );
        EXPECT_EQ( stages[ 0 ].name, "output" );
        EXPECT_TRUE( stages[ 0 ].passes.contains( ComputePass::Forward ) );
        EXPECT_FALSE( stages[ 0 ].passes.contains( ComputePass::Decode ) );
    }

    TEST_F( GeluCudaTests, Observation_PublishesTheLiveViewNotTheCeiling )
    {
        const shape_t built_shape{ 1, 8, 4 };
        const shape_t narrow_shape{ 1, 2, 4 };

        auto host_in = makeSpreadHost( narrow_shape );
        auto device_in = toDevice( host_in, narrow_shape );

        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );
        gelu.build( BuildContext( built_shape, RuntimeMode::Inference ) );

        struct Record
        {
            std::string path;
            ComputePass pass;
            std::string stage;
            shape_t shape;
        };

        std::vector<Record> seen;

        gelu.getExecutionContext()->setActivationObserver(
            [&seen]( std::string_view path, ComputePass pass, std::string_view stage,
                const ITensor& value )
            {
                seen.push_back( { std::string( path ), pass, std::string( stage ), value.shape() } );
            } );

        // Attached but not selected: the observer must not fire.
        gelu.forward( device_in );

        EXPECT_TRUE( seen.empty() ) << "published without an attach walk selecting the component";

        gelu.setObservedPasses( ComputePassMask{ ComputePass::Forward } );
        gelu.forward( device_in );

        ASSERT_EQ( seen.size(), 1u );
        EXPECT_EQ( seen[ 0 ].path, "gelu" );
        EXPECT_EQ( seen[ 0 ].pass, ComputePass::Forward );
        EXPECT_EQ( seen[ 0 ].stage, "output" );

        // The published tensor is this call's narrowed view, not the built ceiling that
        // getOutputs() reports -- publishing the ceiling would hand over a stale tail.
        EXPECT_EQ( seen[ 0 ].shape, narrow_shape );
        EXPECT_EQ( gelu.getOutputs()[ 0 ]->shape(), built_shape );
    }

    TEST_F( GeluCudaTests, Observation_PassFilterExcludesUnselectedPasses )
    {
        const shape_t shape{ 1, 2, 4 };

        auto host_in = makeSpreadHost( shape );
        auto device_in = toDevice( host_in, shape );

        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );
        gelu.build( BuildContext( shape, RuntimeMode::Inference ) );

        int publications = 0;

        gelu.getExecutionContext()->setActivationObserver(
            [&publications]( std::string_view, ComputePass, std::string_view, const ITensor& )
            {
                ++publications;
            } );

        // Selected for a pass this component does not run: nothing fires.
        gelu.setObservedPasses( ComputePassMask{ ComputePass::Decode } );
        gelu.forward( device_in );

        EXPECT_EQ( publications, 0 );

        gelu.setObservedPasses( ComputePassMask::inference() );
        gelu.forward( device_in );

        EXPECT_EQ( publications, 1 );
    }

    TEST_F( GeluCudaTests, Observation_OutputsAgreeWithReportedMemory )
    {
        const shape_t shape{ 2, 16, 32 };

        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );
        gelu.build( BuildContext( shape, RuntimeMode::Inference ) );

        const auto outputs = gelu.getOutputs();

        ASSERT_EQ( outputs.size(), 1u );

        // The cross-check the description exists to make possible: what a component says it
        // produces has to account for what it reports allocating.
        EXPECT_EQ( outputs[ 0 ]->getStorageSize(), gelu.getMemoryStats().device_state_bytes );
    }

    // ====================================================================
    // L. Observability publication cost (Observability.md section 7)
    //
    // Measures host-side enqueue cost per forward() with no observer attached.
    // The loop deliberately does not synchronize per call: the branch under test
    // is host-side, so what matters is its share of enqueue cost, not of kernel
    // time. Run with --gtest_also_run_disabled_tests.
    // ====================================================================

    namespace
    {
        struct EnqueueTiming
        {
            double enqueue_nanoseconds_per_call;
            double total_nanoseconds_per_call;
        };

        EnqueueTiming timeForward( GeluCuda& gelu, const DeviceTensor& input, int iterations )
        {
            const auto start = std::chrono::steady_clock::now();

            for ( int i = 0; i < iterations; ++i )
            {
                gelu.forward( input );
            }

            const auto enqueued = std::chrono::steady_clock::now();
            gelu.synchronize();
            const auto done = std::chrono::steady_clock::now();

            const double enqueue_ns =
                std::chrono::duration<double, std::nano>( enqueued - start ).count();
            const double total_ns =
                std::chrono::duration<double, std::nano>( done - start ).count();

            return { enqueue_ns / iterations, total_ns / iterations };
        }

        void reportPublishCost( const char* label, GeluCuda& gelu, const DeviceTensor& input )
        {
            constexpr int warmup = 2000;
            constexpr int iterations = 50000;
            constexpr int repeats = 5;

            for ( int i = 0; i < warmup; ++i )
            {
                gelu.forward( input );
            }

            gelu.synchronize();

            std::vector<double> enqueue;
            std::vector<double> total;

            for ( int r = 0; r < repeats; ++r )
            {
                const auto timing = timeForward( gelu, input, iterations );

                enqueue.push_back( timing.enqueue_nanoseconds_per_call );
                total.push_back( timing.total_nanoseconds_per_call );
            }

            std::sort( enqueue.begin(), enqueue.end() );
            std::sort( total.begin(), total.end() );

            std::cout << "[publish-cost] " << label
                << " enqueue_ns/call median=" << enqueue[ repeats / 2 ]
                << " min=" << enqueue.front() << " max=" << enqueue.back()
                << " | total_ns/call median=" << total[ repeats / 2 ]
                << " min=" << total.front() << " max=" << total.back()
                << std::endl;
        }
    }

    TEST_F( GeluCudaTests, DISABLED_PublishCost )
    {
        // Decode-shaped: one token, GPT-2 MLP width. Smallest kernel, so the
        // host-side branch is the largest fraction of the call it can ever be.
        {
            const shape_t shape{ 1, 1, 3072 };

            auto host_in = makeSpreadHost( shape );
            auto device_in = toDevice( host_in, shape );

            GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );
            gelu.build( BuildContext( shape, RuntimeMode::Inference ) );

            reportPublishCost( "decode[1,1,3072]", gelu, device_in );
        }

        // Prefill-shaped: 512 tokens of the same width.
        {
            const shape_t shape{ 1, 512, 3072 };

            auto host_in = makeSpreadHost( shape );
            auto device_in = toDevice( host_in, shape );

            GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );
            gelu.build( BuildContext( shape, RuntimeMode::Inference ) );

            reportPublishCost( "prefill[1,512,3072]", gelu, device_in );
        }
    }
}
