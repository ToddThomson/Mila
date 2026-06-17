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

            for ( size_t i = 0; i < host.size(); ++i )
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

        for ( size_t i = 0; i < out.size(); ++i )
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

        for ( size_t i = 0; i < out.size(); ++i )
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
        for ( size_t i = 0; i < host_grad.size(); ++i )
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

        for ( size_t i = 0; i < input_grad.size(); ++i )
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
    // J. Type identity
    // ====================================================================

    TYPED_TEST( LinearCudaTests, GetType_IsLinear )
    {
        LinearConfig config( kInFeatures, kOutFeatures );
        typename TestFixture::LinearType linear( "linear", config, Device::Cuda( 0 ) );

        EXPECT_EQ( linear.getType(), ComponentType::Linear );
    }
}
