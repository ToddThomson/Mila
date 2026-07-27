/**
 * @file Linear.Cuda.cpp
 * @brief Concrete-component tests for Linear<DeviceType::Cuda, {FP32, BF16}>.
 *
 * The CUDA companion to Linear.Cpu.cpp (see Specifications/Testing.md). Unlike
 * Gelu (single CUDA precision), the CUDA LinearOp supports BOTH FP32 and BF16 in
 * the unquantized path, so this is the reference instance of the precision-sweep
 * mechanism the methodology describes: a TYPED_TEST runs the same bodies over
 * each supported precision via ::testing::Types, with the only per-precision
 * variation -- tolerance and the read-as-float accessor -- isolated in a small
 * traits tag. The supported-precision list (LinearPrecisions) is the single point
 * of change: add a CUDA precision to the OperationTraits and to that list, and the
 * suite re-runs for it.
 *
 * Compiled only under MILA_ENABLE_CUDA (the Tests/CMakeLists.txt CUDA block), so
 * no #ifdef. CUDA may still be absent at runtime; SetUp() skips if no device.
 *
 * Dimensions are realistic (not toy): the batch GEMM path goes through cuBLASLt,
 * whose heuristic finds no algorithm for tiny dimensions, so the forward/backward
 * tests use feature sizes a real layer would. The decode path (outer_size == 1)
 * uses the matvec kernel and needs no cuBLASLt -- it is covered separately as the
 * prefill->decode build-context regime.
 *
 * Numeric strategy: deterministic weights/bias/input are uploaded with element
 * conversion to the device precision, then read back to float so the host
 * reference sees exactly the (precision-rounded) values the kernel consumed. Only
 * the GEMM accumulation precision then differs, which the per-precision tolerance
 * accommodates. Base-contract behavior is covered once in Core/Component.cpp.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

import Mila;
// Instantiating Linear<Cuda, P> forces CudaLinearOp's member bodies, which call
// concrete ExecutionContext<Cuda> methods (getCublasLtHandle, etc.). The Mila
// umbrella does not complete that type for a consumer TU, so import it directly.
import Compute.ExecutionContext;
// Same reason: the weight-quant policy structs (PerChannelFp8 / PerGroupFp4) and the
// serialization blob types (TensorMetadata / TensorBlobView) are not re-exported through
// the Mila umbrella, so import their modules directly (clang requires it; MSVC did not).
import Dnn.Quantization.Weight.Policies;
import Serialization.Tensor;

namespace Mila::Tests::Dnn::Components::Linear
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        // Realistic feature sizes — cuBLASLt has no algorithm for toy GEMM dims.
        constexpr int64_t kInFeatures = 64;
        constexpr int64_t kOutFeatures = 32;

        // Deterministic, bounded parameter values (kept small so BF16 rounding of a
        // 64-wide dot product stays well within the per-precision tolerance).
        float weightValue( int64_t out_index, int64_t in_index )
        {
            const int64_t h = ( out_index * 13 + in_index * 7 ) % 17;

            return 0.1f * ( static_cast<float>( h ) - 8.0f ) / 17.0f;
        }

        float biasValue( int64_t out_index )
        {
            return 0.1f * ( static_cast<float>( out_index % 5 ) - 2.0f ) / 5.0f;
        }

        // Host reference: Y = X * W^T + B. W is [out, in], X is [batch, in].
        void referenceForward(
            const float* X, const float* W, const float* B,
            int64_t batch, int64_t in_features, int64_t out_features,
            std::vector<float>& Y )
        {
            Y.assign( static_cast<size_t>( batch * out_features ), 0.0f );

            for ( int64_t b = 0; b < batch; ++b )
            {
                for ( int64_t o = 0; o < out_features; ++o )
                {
                    double acc = B ? static_cast<double>( B[ o ] ) : 0.0;

                    for ( int64_t i = 0; i < in_features; ++i )
                    {
                        acc += static_cast<double>( X[ b * in_features + i ] ) *
                            static_cast<double>( W[ o * in_features + i ] );
                    }

                    Y[ b * out_features + o ] = static_cast<float>( acc );
                }
            }
        }

        // -- Per-precision traits tags -------------------------------------
        // value: the device precision. atol/rtol: the comparison budget. name:
        // the TYPED_TEST instance suffix (LinearCudaTests/Bf16 rather than /1).

        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr float forward_atol = 1e-3f;
            static constexpr float forward_rtol = 1e-4f;
            static constexpr float backward_atol = 1e-2f;
            static constexpr float backward_rtol = 1e-3f;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr float forward_atol = 5e-2f;
            static constexpr float forward_rtol = 5e-2f;
            static constexpr float backward_atol = 1.5e-1f;
            static constexpr float backward_rtol = 1e-1f;
            static constexpr const char* name = "Bf16";
        };

        using LinearPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

        class PrecisionNames
        {
        public:
            template<typename TPrecisionTag>
            static std::string GetName( int )
            {
                return TPrecisionTag::name;
            }
        };
    }

    template<typename TPrecisionTag>
    class LinearCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using LinearType = Mila::Dnn::Linear<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( LinearType::getDeviceType() == DeviceType::Cuda );
        static_assert( LinearType::getPrecision() == P );

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

        std::unique_ptr<LinearType> builtLinear(
            const shape_t& shape, bool has_bias, RuntimeMode mode )
        {
            LinearConfig config( kInFeatures, kOutFeatures );
            config.withBias( has_bias );

            auto linear = std::make_unique<LinearType>( "linear", config, Device::Cuda( 0 ) );
            linear->build( BuildContext( shape, mode, false ) );

            return linear;
        }

        // Upload deterministic known weights/bias (FP32 host -> device precision).
        void setKnownParameters( LinearType& linear, bool has_bias )
        {
            auto params = linear.getParameters();

            HostFp32 host_weight( Device::Cpu(), shape_t{ kOutFeatures, kInFeatures } );
            for ( int64_t o = 0; o < kOutFeatures; ++o )
            {
                for ( int64_t i = 0; i < kInFeatures; ++i )
                {
                    host_weight.data()[ o * kInFeatures + i ] = weightValue( o, i );
                }
            }
            copy( host_weight, *static_cast<DeviceTensor*>( params[ 0 ] ), cuda_context_.get() );

            if ( has_bias )
            {
                HostFp32 host_bias( Device::Cpu(), shape_t{ kOutFeatures } );
                for ( int64_t o = 0; o < kOutFeatures; ++o )
                {
                    host_bias.data()[ o ] = biasValue( o );
                }
                copy( host_bias, *static_cast<DeviceTensor*>( params[ 1 ] ), cuda_context_.get() );
            }

            cuda_context_->synchronize();
        }

        HostFp32 spreadHost( const shape_t& shape )
        {
            HostFp32 host( Device::Cpu(), shape );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = static_cast<float>( i ) / host.size() * 2.0f - 1.0f;
            }

            return host;
        }

        DeviceTensor toDevice( const HostFp32& host )
        {
            DeviceTensor device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        HostFp32 toFloat( const DeviceTensor& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, cuda_context_.get() );
            cuda_context_->synchronize();

            return host;
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TYPED_TEST_SUITE( LinearCudaTests, LinearPrecisions, PrecisionNames );

    // ====================================================================
    // A. Construction
    // ====================================================================

    TYPED_TEST( LinearCudaTests, Construct_StandaloneSucceeds )
    {
        LinearConfig config( kInFeatures, kOutFeatures );
        typename TestFixture::LinearType linear( "linear", config, Device::Cuda( 0 ) );

        EXPECT_EQ( linear.getDeviceId().type, DeviceType::Cuda );
        EXPECT_TRUE( linear.hasBias() );
    }

    // ====================================================================
    // E. Forward (numeric vs reference) — prefill (batch GEMM via cuBLASLt)
    // ====================================================================

    TYPED_TEST( LinearCudaTests, Forward_MatchesReference )
    {
        // Re-enabled 2026-06-17: the non-quantized cuBLASLt batch GEMM
        // CUBLAS_STATUS_NOT_SUPPORTED was root-caused to the bias epilogue (not the
        // NT row-major layout) and fixed 2026-06-16 by adding bias post-GEMM. The
        // decode (matvec) oracle is covered by Forward_DecodeMatchesReference.
        const shape_t shape{ 2, 4, kInFeatures };

        auto linear = this->builtLinear( shape, true, RuntimeMode::Inference );
        this->setKnownParameters( *linear, true );

        auto host_in = this->spreadHost( shape );
        auto device_in = this->toDevice( host_in );

        auto& device_out = linear->forward( device_in );
        linear->synchronize();

        auto out = this->toFloat( device_out );

        // Read back the precision-rounded values the kernel actually consumed.
        auto params = linear->getParameters();
        auto weight = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 0 ] ) );
        auto bias = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 1 ] ) );
        auto in = this->toFloat( device_in );

        const int64_t batch = 2 * 4;
        std::vector<float> expected;
        referenceForward( in.data(), weight.data(), bias.data(), batch, kInFeatures, kOutFeatures, expected );

        ASSERT_EQ( out.size(), expected.size() );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            const float tolerance = TypeParam::forward_atol + TypeParam::forward_rtol * std::fabs( expected[ i ] );

            EXPECT_NEAR( out.data()[ i ], expected[ i ], tolerance )
                << "forward mismatch at index " << i;
        }
    }

    // ====================================================================
    // E/D. Forward decode regime — build prefill shape, forward outer_size == 1
    //      (matvec path, no cuBLASLt). The build-context prefill->decode axis.
    // ====================================================================

    TYPED_TEST( LinearCudaTests, Forward_DecodeMatchesReference )
    {
        // Built for a multi-token prefill shape...
        const shape_t prefill_shape{ 1, 4, kInFeatures };
        auto linear = this->builtLinear( prefill_shape, true, RuntimeMode::Inference );
        this->setKnownParameters( *linear, true );

        // ...then driven with a single-token decode shape.
        const shape_t decode_shape{ 1, 1, kInFeatures };
        auto host_in = this->spreadHost( decode_shape );
        auto device_in = this->toDevice( host_in );

        auto& device_out = linear->forward( device_in );
        linear->synchronize();

        auto out = this->toFloat( device_out );

        ASSERT_EQ( out.shape(), ( shape_t{ 1, 1, kOutFeatures } ) );

        auto params = linear->getParameters();
        auto weight = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 0 ] ) );
        auto bias = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 1 ] ) );
        auto in = this->toFloat( device_in );

        std::vector<float> expected;
        referenceForward( in.data(), weight.data(), bias.data(), 1, kInFeatures, kOutFeatures, expected );

        ASSERT_EQ( out.size(), expected.size() );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            const float tolerance = TypeParam::forward_atol + TypeParam::forward_rtol * std::fabs( expected[ i ] );

            EXPECT_NEAR( out.data()[ i ], expected[ i ], tolerance )
                << "decode mismatch at index " << i;
        }
    }

    // ====================================================================
    // F. Backward (numeric vs analytic gradient)
    // ====================================================================

    TYPED_TEST( LinearCudaTests, Backward_MatchesReferenceGradients )
    {
        // Re-enabled 2026-06-17: backward shares the non-quantized cuBLASLt plans with
        // the batch forward; the CUBLAS_STATUS_NOT_SUPPORTED was the bias epilogue,
        // fixed 2026-06-16 (backward plans carry no bias epilogue, so once forward
        // builds, use_cublaslt_ stays true). CPU backward numerics: Linear.Cpu.cpp.
        const shape_t input_shape{ 2, 4, kInFeatures };
        const shape_t output_shape{ 2, 4, kOutFeatures };

        auto linear = this->builtLinear( input_shape, true, RuntimeMode::Training );
        this->setKnownParameters( *linear, true );

        auto host_in = this->spreadHost( input_shape );
        auto device_in = this->toDevice( host_in );

        typename TestFixture::HostFp32 host_grad( Device::Cpu(), output_shape );
        for ( dim_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = 0.01f * static_cast<float>( ( i % 13 ) + 1 );
        }
        auto device_grad = this->toDevice( host_grad );

        linear->forward( device_in );
        auto& device_in_grad = linear->backward( device_in, device_grad );
        linear->synchronize();

        auto input_grad = this->toFloat( device_in_grad );

        // Read back precision-rounded inputs the kernels actually consumed.
        auto params = linear->getParameters();
        auto weight = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 0 ] ) );
        auto in = this->toFloat( device_in );
        auto grad = this->toFloat( device_grad );

        const int64_t batch = 2 * 4;

        // dX[b,i] = sum_o dY[b,o] * W[o,i]
        std::vector<float> expected_dx( static_cast<size_t>( batch * kInFeatures ), 0.0f );
        for ( int64_t b = 0; b < batch; ++b )
        {
            for ( int64_t i = 0; i < kInFeatures; ++i )
            {
                double acc = 0.0;
                for ( int64_t o = 0; o < kOutFeatures; ++o )
                {
                    acc += static_cast<double>( grad.data()[ b * kOutFeatures + o ] ) *
                        static_cast<double>( weight.data()[ o * kInFeatures + i ] );
                }
                expected_dx[ b * kInFeatures + i ] = static_cast<float>( acc );
            }
        }

        ASSERT_EQ( input_grad.size(), expected_dx.size() );

        for ( dim_t i = 0; i < input_grad.size(); ++i )
        {
            const float tolerance = TypeParam::backward_atol + TypeParam::backward_rtol * std::fabs( expected_dx[ i ] );

            EXPECT_NEAR( input_grad.data()[ i ], expected_dx[ i ], tolerance )
                << "input-gradient mismatch at index " << i;
        }

        // dW[o,i] = sum_b dY[b,o] * X[b,i] ; dB[o] = sum_b dY[b,o]
        auto grads = linear->getGradients();
        ASSERT_EQ( grads.size(), 2u );
        auto weight_grad = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( grads[ 0 ] ) );
        auto bias_grad = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( grads[ 1 ] ) );

        for ( int64_t o = 0; o < kOutFeatures; ++o )
        {
            double db = 0.0;
            for ( int64_t b = 0; b < batch; ++b )
            {
                db += static_cast<double>( grad.data()[ b * kOutFeatures + o ] );
            }
            const float bias_tol = TypeParam::backward_atol + TypeParam::backward_rtol * std::fabs( static_cast<float>( db ) );
            EXPECT_NEAR( bias_grad.data()[ o ], static_cast<float>( db ), bias_tol )
                << "bias-gradient mismatch at o=" << o;

            for ( int64_t i = 0; i < kInFeatures; ++i )
            {
                double dw = 0.0;
                for ( int64_t b = 0; b < batch; ++b )
                {
                    dw += static_cast<double>( grad.data()[ b * kOutFeatures + o ] ) *
                        static_cast<double>( in.data()[ b * kInFeatures + i ] );
                }
                const float weight_tol = TypeParam::backward_atol + TypeParam::backward_rtol * std::fabs( static_cast<float>( dw ) );
                EXPECT_NEAR( weight_grad.data()[ o * kInFeatures + i ], static_cast<float>( dw ), weight_tol )
                    << "weight-gradient mismatch at o=" << o << " i=" << i;
            }
        }
    }

    // ====================================================================
    // G. Parameters & Gradients
    // ====================================================================

    TYPED_TEST( LinearCudaTests, Parameters_CountReflectsBias )
    {
        auto with_bias = this->builtLinear( shape_t{ 2, kInFeatures }, true, RuntimeMode::Inference );
        EXPECT_EQ( with_bias->getParameters().size(), 2u );

        auto without_bias = this->builtLinear( shape_t{ 2, kInFeatures }, false, RuntimeMode::Inference );
        EXPECT_EQ( without_bias->getParameters().size(), 1u );
    }

    TYPED_TEST( LinearCudaTests, Gradients_PresentOnlyForTrainingBuild )
    {
        auto inference = this->builtLinear( shape_t{ 2, kInFeatures }, true, RuntimeMode::Inference );
        EXPECT_TRUE( inference->getGradients().empty() );

        auto training = this->builtLinear( shape_t{ 2, kInFeatures }, true, RuntimeMode::Training );
        EXPECT_EQ( training->getGradients().size(), 2u );
    }

    // ====================================================================
    // H. Shared weight installation (weight tying — WeightTying.md)
    // ====================================================================

    // installSharedWeight replaces the owned weight with a caller-owned tensor (the
    // tied lm_head sharing the token embedding table). Assert the component exposes the
    // shared tensor as its parameter (pointer identity), reports its bytes (so the
    // owning transformer's tie-aware getMemoryStats can subtract them once), and that
    // forward actually consumes the installed weight.
    TYPED_TEST( LinearCudaTests, InstallSharedWeight_SetsParameterAndMatchesDirectLoad )
    {
        const shape_t shape{ 2, 4, kInFeatures };

        // No bias — the lm_head that motivates this path is unbiased.
        auto linear = this->builtLinear( shape, false, RuntimeMode::Inference );

        auto shared = std::make_shared<typename TestFixture::DeviceTensor>(
            Device::Cuda( 0 ), shape_t{ kOutFeatures, kInFeatures }, "shared.weight" );

        typename TestFixture::HostFp32 host_weight( Device::Cpu(), shape_t{ kOutFeatures, kInFeatures } );
        for ( int64_t o = 0; o < kOutFeatures; ++o )
        {
            for ( int64_t i = 0; i < kInFeatures; ++i )
            {
                host_weight.data()[ o * kInFeatures + i ] = weightValue( o, i );
            }
        }
        copy( host_weight, *shared, this->cuda_context_.get() );
        this->cuda_context_->synchronize();

        linear->installSharedWeight( shared );

        auto params = linear->getParameters();
        ASSERT_EQ( params.size(), 1u );
        EXPECT_EQ( static_cast<typename TestFixture::DeviceTensor*>( params[ 0 ] ), shared.get() );
        EXPECT_EQ( linear->getMemoryStats().device_parameter_bytes, shared->getStorageSize() );

        auto host_in = this->spreadHost( shape );
        auto device_in = this->toDevice( host_in );

        auto& device_out = linear->forward( device_in );
        linear->synchronize();

        auto out = this->toFloat( device_out );

        // Read back the precision-rounded weight the kernel actually consumed.
        auto weight = this->toFloat( *shared );
        auto in = this->toFloat( device_in );

        std::vector<float> expected;
        referenceForward( in.data(), weight.data(), nullptr, 2 * 4, kInFeatures, kOutFeatures, expected );

        ASSERT_EQ( out.size(), expected.size() );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            const float tolerance = TypeParam::forward_atol + TypeParam::forward_rtol * std::fabs( expected[ i ] );

            EXPECT_NEAR( out.data()[ i ], expected[ i ], tolerance )
                << "shared-weight forward mismatch at index " << i;
        }
    }

    // Tying contract on quantized instantiations (D4 Design B): per-channel FP8
    // accepts the (weight, scales) overload -- the per-output-channel scale axis IS
    // the vocab row a tied embedding gathers -- while per-group policies stay
    // excluded (input-axis scales do not transfer to a row gather), and a quantized
    // weight without scales is always rejected. Deferred construction (no device)
    // keeps the throw tests GPU-independent: the throw precedes any op or context
    // use, and the arguments are never dereferenced.
    TEST( LinearCudaQuantizedTests, InstallSharedWeight_PerGroupPath_Throws )
    {
        using QuantizedLinear =
            Mila::Dnn::Linear<DeviceType::Cuda, TensorDataType::BF16, Mila::Dnn::Quant::Weight::PerGroupFp4<128>>;

        LinearConfig config( kInFeatures, kOutFeatures );
        config.withBias( false );
        QuantizedLinear linear( "linear_quantized", config );

        EXPECT_THROW( linear.installSharedWeight( nullptr ), std::logic_error );
        EXPECT_THROW( linear.installSharedWeight( nullptr, nullptr ), std::logic_error );
    }

    TEST( LinearCudaQuantizedTests, InstallSharedWeight_PerChannelWithoutScales_Throws )
    {
        using QuantizedLinear =
            Mila::Dnn::Linear<DeviceType::Cuda, TensorDataType::BF16, Mila::Dnn::Quant::Weight::PerChannelFp8<>>;

        LinearConfig config( kInFeatures, kOutFeatures );
        config.withBias( false );
        QuantizedLinear linear( "linear_quantized", config );

        EXPECT_THROW( linear.installSharedWeight( nullptr ), std::logic_error );
    }

    // The real D4 wiring end to end: a quantized TokenEmbedding donor produces the
    // FP8 table + per-vocab-row scales via quantize-on-load; a per-channel FP8
    // lm_head that adopts both handles must compute exactly what a head that
    // quantized the same BF16 blob through its own loadParameter computes -- the
    // two paths run the same quantization kernel on the same data, so the outputs
    // are identical, not merely close.
    TEST( LinearCudaQuantizedTests, InstallSharedWeight_PerChannelFp8_MatchesDirectQuantizedLoad )
    {
        using QuantizedLinear =
            Mila::Dnn::Linear<DeviceType::Cuda, TensorDataType::BF16, Mila::Dnn::Quant::Weight::PerChannelFp8<>>;
        using QuantizedEmbedding = Mila::Dnn::TokenEmbedding<
            DeviceType::Cuda, TensorDataType::INT32, TensorDataType::BF16, Mila::Dnn::Quant::Weight::PerChannelFp8<>>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;

        std::unique_ptr<IExecutionContext> context;
        try
        {
            context = createExecutionContext( Device::Cuda( 0 ) );
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "CUDA device not available";
        }

        // BF16 weight blob shared by all three consumers.
        std::vector<uint16_t> weight_bits( static_cast<size_t>( kOutFeatures * kInFeatures ) );
        for ( int64_t o = 0; o < kOutFeatures; ++o )
        {
            for ( int64_t i = 0; i < kInFeatures; ++i )
            {
                const uint32_t bits = std::bit_cast<uint32_t>( weightValue( o, i ) );
                const uint32_t rounding = 0x7FFF + ( ( bits >> 16 ) & 1 );
                weight_bits[ o * kInFeatures + i ] = static_cast<uint16_t>( ( bits + rounding ) >> 16 );
            }
        }

        const size_t blob_bytes = weight_bits.size() * sizeof( uint16_t );

        // Decode-shaped build (outer_size == 1): the FP8 matvec path the tied
        // lm_head actually runs.
        const shape_t input_shape{ 1, kInFeatures };
        LinearConfig config( kInFeatures, kOutFeatures );
        config.withBias( false );

        // Head A: direct quantize-on-load through its own loadParameter.
        QuantizedLinear direct( "lm_head_direct", config, Device::Cuda( 0 ) );
        direct.build( BuildContext( input_shape, RuntimeMode::Inference, false ) );

        Serialization::TensorMetadata weight_meta{
            TensorDataType::BF16, shape_t{ kOutFeatures, kInFeatures }, blob_bytes };
        Serialization::TensorBlobView weight_blob( weight_meta, weight_bits.data(), blob_bytes );
        direct.loadParameter( "weight", weight_blob );
        context->synchronize();

        // Donor: quantized TokenEmbedding over the same table ([vocab=out, d=in]).
        auto embedding_config = TokenEmbeddingConfig()
            .withVocabSize( static_cast<size_t>( kOutFeatures ) )
            .withEmbeddingDim( static_cast<size_t>( kInFeatures ) );
        QuantizedEmbedding embedding( "token_embedding_fp8", embedding_config, Device::Cuda( 0 ) );
        embedding.build( BuildContext( shape_t{ 1, 2 }, RuntimeMode::Inference, false ) );

        Serialization::TensorBlobView wte_blob( weight_meta, weight_bits.data(), blob_bytes );
        embedding.loadParameter( "wte", wte_blob );
        context->synchronize();

        // Head B: adopts the donor's FP8 table and row scales.
        QuantizedLinear tied( "lm_head_tied", config, Device::Cuda( 0 ) );
        tied.build( BuildContext( input_shape, RuntimeMode::Inference, false ) );
        tied.installSharedWeight( embedding.getWeightTensorShared(), embedding.getWeightScalesTensorShared() );

        // Pointer identity: the tied head exposes the shared table as its parameter.
        auto tied_params = tied.getParameters();
        ASSERT_EQ( tied_params.size(), 1u );
        EXPECT_EQ( tied_params[ 0 ], static_cast<ITensor*>( embedding.getWeightTensorShared().get() ) );

        // Same input through both heads.
        HostFp32 host_input( Device::Cpu(), input_shape );
        for ( int64_t i = 0; i < kInFeatures; ++i )
        {
            host_input.data()[ i ] = 0.5f * weightValue( i % kOutFeatures, i );
        }

        DeviceBf16 device_input( Device::Cuda( 0 ), input_shape );
        copy( host_input, device_input, context.get() );
        context->synchronize();

        auto& direct_out = direct.forward( device_input );
        direct.synchronize();
        auto direct_host = toHost<TensorDataType::FP32>( direct_out, context.get() );
        context->synchronize();

        auto& tied_out = tied.forward( device_input );
        tied.synchronize();
        auto tied_host = toHost<TensorDataType::FP32>( tied_out, context.get() );
        context->synchronize();

        ASSERT_EQ( direct_host.size(), tied_host.size() );

        for ( dim_t i = 0; i < direct_host.size(); ++i )
        {
            EXPECT_EQ( tied_host.data()[ i ], direct_host.data()[ i ] )
                << "tied FP8 head diverged from direct quantized load at index " << i;
        }
    }

    // W4A8-FP8 prefill numerics oracle (Fp8ActivationPrefill.md section 6): the
    // batched prefill forward must match the decode matvec -- an independent, proven
    // FP4 path over the SAME loaded weights -- row for row. Input rows span fifteen
    // decades of magnitude: under this fixture per-TENSOR activation scaling fails
    // (the +98 incoherence -- one outlier row crushes every other row's FP8
    // resolution) while per-TOKEN scaling passes. The comparison budget is per-row,
    // proportional to the row's reference absmax, because FP8 activation error is
    // relative to each token's own scale. The test stays valid with
    // kUseFp8ActivationPrefill=false (BF16 staging GEMM vs matvec, passes with
    // margin), so it does not depend on the toggle state.
    TEST( LinearCudaQuantizedTests, Forward_Fp4PrefillMatchesDecodeAcrossTokenMagnitudes )
    {
        using QuantizedLinear =
            Mila::Dnn::Linear<DeviceType::Cuda, TensorDataType::BF16, Mila::Dnn::Quant::Weight::PerGroupFp4<128>>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;

        // Realistic layer-ish dims: in_features spans four FP4 groups of 128, and
        // both GEMM dims are large enough for the cuBLASLt FP8 heuristic.
        constexpr int64_t kFp8Rows = 16;
        constexpr int64_t kFp8InFeatures = 512;
        constexpr int64_t kFp8OutFeatures = 256;

        std::unique_ptr<IExecutionContext> context;
        try
        {
            context = createExecutionContext( Device::Cuda( 0 ) );
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "CUDA device not available";
        }

        // BF16 weight blob, quantized to FP4 by loadParameter.
        std::vector<uint16_t> weight_bits( static_cast<size_t>( kFp8OutFeatures * kFp8InFeatures ) );
        for ( int64_t o = 0; o < kFp8OutFeatures; ++o )
        {
            for ( int64_t i = 0; i < kFp8InFeatures; ++i )
            {
                const uint32_t bits = std::bit_cast<uint32_t>( weightValue( o, i ) );
                const uint32_t rounding = 0x7FFF + ( ( bits >> 16 ) & 1 );
                weight_bits[ o * kFp8InFeatures + i ] = static_cast<uint16_t>( ( bits + rounding ) >> 16 );
            }
        }

        const size_t blob_bytes = weight_bits.size() * sizeof( uint16_t );

        LinearConfig config( kFp8InFeatures, kFp8OutFeatures );
        config.withBias( false );

        QuantizedLinear linear( "linear_fp4_prefill", config, Device::Cuda( 0 ) );
        linear.build( BuildContext( shape_t{ kFp8Rows, kFp8InFeatures }, RuntimeMode::Inference, false ) );

        Serialization::TensorMetadata weight_meta{
            TensorDataType::BF16, shape_t{ kFp8OutFeatures, kFp8InFeatures }, blob_bytes };
        Serialization::TensorBlobView weight_blob( weight_meta, weight_bits.data(), blob_bytes );
        linear.loadParameter( "weight", weight_blob );
        context->synchronize();

        // Row m carries magnitude 10^(m-8): 1e-8 .. 1e+7 across the batch.
        HostFp32 host_input( Device::Cpu(), shape_t{ kFp8Rows, kFp8InFeatures } );
        for ( int64_t m = 0; m < kFp8Rows; ++m )
        {
            const float row_scale = std::pow( 10.0f, static_cast<float>( m ) - 8.0f );

            for ( int64_t k = 0; k < kFp8InFeatures; ++k )
            {
                const float spread = static_cast<float>( ( m * 31 + k * 17 ) % 257 ) / 128.0f - 1.0f;
                host_input.data()[ m * kFp8InFeatures + k ] = row_scale * spread;
            }
        }

        DeviceBf16 device_input( Device::Cuda( 0 ), shape_t{ kFp8Rows, kFp8InFeatures } );
        copy( host_input, device_input, context.get() );
        context->synchronize();

        // Prefill leg: batched forward (the W4A8-FP8 GEMM when the toggle is on).
        auto& prefill_out = linear.forward( device_input );
        linear.synchronize();
        auto prefill_host = toHost<TensorDataType::FP32>( prefill_out, context.get() );
        context->synchronize();

        // The op reuses its output tensor across forwards -- keep a copy before the
        // decode legs overwrite it.
        std::vector<float> prefill(
            prefill_host.data(), prefill_host.data() + kFp8Rows * kFp8OutFeatures );

        // Decode legs: one matvec forward per row, same op instance, same weights.
        for ( int64_t m = 0; m < kFp8Rows; ++m )
        {
            HostFp32 host_row( Device::Cpu(), shape_t{ 1, kFp8InFeatures } );
            for ( int64_t k = 0; k < kFp8InFeatures; ++k )
            {
                host_row.data()[ k ] = host_input.data()[ m * kFp8InFeatures + k ];
            }

            DeviceBf16 device_row( Device::Cuda( 0 ), shape_t{ 1, kFp8InFeatures } );
            copy( host_row, device_row, context.get() );
            context->synchronize();

            auto& decode_out = linear.forward( device_row );
            linear.synchronize();
            auto decode_host = toHost<TensorDataType::FP32>( decode_out, context.get() );
            context->synchronize();

            float row_absmax = 0.0f;
            for ( int64_t n = 0; n < kFp8OutFeatures; ++n )
            {
                row_absmax = std::max( row_absmax, std::fabs( decode_host.data()[ n ] ) );
            }

            const float tolerance = 5e-2f * row_absmax;

            for ( int64_t n = 0; n < kFp8OutFeatures; ++n )
            {
                EXPECT_NEAR( prefill[ m * kFp8OutFeatures + n ], decode_host.data()[ n ], tolerance )
                    << "prefill/decode mismatch at row " << m << " (magnitude 1e"
                    << ( m - 8 ) << "), column " << n;
            }
        }
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TYPED_TEST( LinearCudaTests, GetType_IsLinear )
    {
        LinearConfig config( kInFeatures, kOutFeatures );
        typename TestFixture::LinearType linear( "linear", config, Device::Cuda( 0 ) );

        EXPECT_EQ( linear.getType(), ComponentType::Linear );
    }
}
