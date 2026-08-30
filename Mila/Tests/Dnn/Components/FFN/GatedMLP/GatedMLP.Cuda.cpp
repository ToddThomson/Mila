/**
 * @file GatedMLP.Cuda.cpp
 * @brief Concrete-component tests for GatedMLP<DeviceType::Cuda, {FP32, BF16}>.
 *
 * GatedMLP's gate is the CUDA-only Swiglu op, so there is no CPU companion -- this
 * is the whole functional surface (the config contract lives in GatedMLPConfig.cpp).
 * A precision sweep (FP32, BF16) covers the composite wiring at the value-agnostic
 * level (build shape contract, aggregate parameter count, backward shape). The exact
 * numeric identity -- bias-free, a zero input yields a zero output -- is FP32-only
 * and Training-built: it needs finite (initialized) weights, and BF16 train-from-
 * scratch init is the known-buggy path (BACKLOG fill_normal FP32-only).
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device is present.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

import Mila;
// Instantiating GatedMLP<Cuda, P> forces the child Linear/Swiglu CUDA op member
// bodies, which call concrete ExecutionContext<Cuda> methods; complete the type here.

namespace Mila::Tests::Dnn::Components::FFN::GatedMLP
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        // hidden_size is the Swiglu half_width; must be a multiple of the BF16 kernel
        // vector width (8). in_features is unconstrained but 8 keeps the GEMMs small.
        constexpr int64_t kInFeatures = 8;
        constexpr int64_t kHiddenSize = 8;

        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr const char* name = "Bf16";
        };

        using GatedMLPPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

        class PrecisionNames
        {
        public:
            template<typename TPrecisionTag>
            static std::string GetName( int ) { return TPrecisionTag::name; }
        };

        // Aggregate trainable count: fc_gate_up (2H x in) + fc_down (in x H), bias-free.
        constexpr size_t kExpectedParameters =
            static_cast<size_t>( 2 * kHiddenSize * kInFeatures + kInFeatures * kHiddenSize );
    }

    template<typename TPrecisionTag>
    class GatedMLPCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using GatedMLPType = Mila::Dnn::GatedMLP<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( GatedMLPType::getDeviceType() == DeviceType::Cuda );
        static_assert( GatedMLPType::getPrecision() == P );

        void SetUp() override
        {
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

        // Training build so the child Linear weights are initialized (finite).
        std::unique_ptr<GatedMLPType> built( const shape_t& shape )
        {
            auto gated = std::make_unique<GatedMLPType>(
                "gmlp", GatedMLPConfig( kInFeatures, kHiddenSize ), Device::Cuda( 0 ) );
            gated->build( BuildContext( shape, RuntimeMode::Training ) );

            return gated;
        }

        DeviceTensor zeros( const shape_t& shape )
        {
            HostFp32 host( Device::Cpu(), shape );
            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = 0.0f;
            }

            DeviceTensor device( Device::Cuda( 0 ), shape );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TYPED_TEST_SUITE( GatedMLPCudaTests, GatedMLPPrecisions, PrecisionNames );

    TYPED_TEST( GatedMLPCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::GatedMLPType gated(
            "gmlp", GatedMLPConfig( kInFeatures, kHiddenSize ), Device::Cuda( 0 ) );

        EXPECT_EQ( gated.getDeviceId().type, DeviceType::Cuda );
        EXPECT_EQ( gated.getType(), ComponentType::GatedMlp );
    }

    TYPED_TEST( GatedMLPCudaTests, Forward_PreservesInputShape )
    {
        const shape_t shape{ 2, 3, kInFeatures };

        auto gated = this->built( shape );
        auto input = this->zeros( shape );

        auto& output = gated->forward( input );
        gated->synchronize();

        ASSERT_EQ( output.shape().size(), 3u );
        EXPECT_EQ( output.shape()[ 0 ], 2 );
        EXPECT_EQ( output.shape()[ 1 ], 3 );
        EXPECT_EQ( output.shape()[ 2 ], kInFeatures );
    }

    TYPED_TEST( GatedMLPCudaTests, ParameterCount_IsSumOfProjections )
    {
        auto gated = this->built( shape_t{ 2, 3, kInFeatures } );

        EXPECT_EQ( gated->parameterCount(), kExpectedParameters );
    }

    TYPED_TEST( GatedMLPCudaTests, Backward_ProducesInputShapeGradient )
    {
        const shape_t shape{ 2, 3, kInFeatures };

        auto gated = this->built( shape );
        auto input = this->zeros( shape );
        auto output_grad = this->zeros( shape );

        gated->forward( input );
        auto& input_grad = gated->backward( input, output_grad );
        gated->synchronize();

        ASSERT_EQ( input_grad.shape().size(), 3u );
        EXPECT_EQ( input_grad.shape()[ 2 ], kInFeatures );
    }

    // ====================================================================
    // FP32 numeric identity: bias-free, zero input -> zero output.
    //   0 -> fc_gate_up=0 -> SiLU(0)*0 = 0 -> fc_down(0) = 0.
    // FP32 + Training build so weights are finite; 0 * finite = 0 exactly.
    // ====================================================================

    class GatedMLPCudaFp32Tests : public ::testing::Test
    {
    protected:
        using GatedMLPType = Mila::Dnn::GatedMLP<DeviceType::Cuda, TensorDataType::FP32>;
        using DeviceFp32 = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        void SetUp() override
        {
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

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TEST_F( GatedMLPCudaFp32Tests, Forward_ZeroInputYieldsZero )
    {
        const shape_t shape{ 2, 3, kInFeatures };

        GatedMLPType gated( "gmlp", GatedMLPConfig( kInFeatures, kHiddenSize ), Device::Cuda( 0 ) );
        gated.build( BuildContext( shape, RuntimeMode::Training ) );

        HostFp32 host_zero( Device::Cpu(), shape );
        for ( dim_t i = 0; i < host_zero.size(); ++i )
        {
            host_zero.data()[ i ] = 0.0f;
        }

        DeviceFp32 input( Device::Cuda( 0 ), shape );
        copy( host_zero, input, cuda_context_.get() );
        cuda_context_->synchronize();

        auto& output = gated.forward( input );
        gated.synchronize();

        auto host_out = toHost<TensorDataType::FP32>( output, cuda_context_.get() );

        for ( dim_t i = 0; i < host_out.size(); ++i )
        {
            EXPECT_NEAR( host_out.data()[ i ], 0.0f, 1e-6f )
                << "non-zero output at index " << i;
        }
    }
}
