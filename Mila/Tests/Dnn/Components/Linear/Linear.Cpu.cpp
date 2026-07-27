/**
 * @file Linear.Cpu.cpp
 * @brief Concrete-component tests for Linear<DeviceType::Cpu, FP32>.
 *
 * Reference instance of the concrete-component archetype for a leaf WITH
 * parameters (see Specifications/Testing.md). Where Gelu.Cpu.cpp covers the
 * stateless case, Linear adds the substance of section G (weight/bias parameters
 * and gradients) and section H (parameter load). It tests only the Linear DELTA
 * over the Component base contract: Linear-specific construction/validation, the
 * build->forward->backward numerics (y = x*W^T + b), parameter/gradient access,
 * and the loadParameter path. The inherited base machinery is covered once in
 * Core/Component.cpp and is not retested here.
 *
 * CPU device, so this rides the MILA_ENABLE_CUDA=OFF CI gate. CUDA-instantiation
 * tests (and the FP32/BF16 precision sweep) belong in Linear.Cuda.cpp.
 *
 * Numeric strategy: weights and bias are set to deterministic known values
 * (rather than relying on parameter init), so the host reference matmul is fully
 * controlled. CPU FP32 is exact, so a tight tolerance applies.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

#include "Common/GradientCheck.h"

import Mila;
// TensorMetadata / TensorBlobView live in the Serialization.Tensor module, which
// the Mila umbrella does not re-export; import it directly for the loadParameter tests.
import Serialization.Tensor;

namespace Mila::Tests::Dnn::Components::Linear
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using Mila::Dnn::Serialization::TensorMetadata;
    using Mila::Dnn::Serialization::TensorBlobView;

    namespace
    {
        using LinearCpu = Mila::Dnn::Linear<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // Deterministic parameter values, independent of the component under test.
        float weightValue( int64_t out_index, int64_t in_index )
        {
            return 0.05f * static_cast<float>( out_index + 1 ) - 0.03f * static_cast<float>( in_index );
        }

        float biasValue( int64_t out_index )
        {
            return 0.2f * static_cast<float>( out_index ) - 0.1f;
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

        static_assert( LinearCpu::getDeviceType() == DeviceType::Cpu );
        static_assert( LinearCpu::getPrecision() == TensorDataType::FP32 );
    }

    class LinearCpuTests : public ::testing::Test
    {
    protected:
        // Standalone Linear (owns its CPU execution context), built in the given
        // mode. Parameter init is disabled because the numeric tests set known
        // weights explicitly via setKnownParameters().
        std::unique_ptr<LinearCpu> builtLinear(
            const shape_t& shape, int64_t in_features, int64_t out_features,
            bool has_bias, RuntimeMode mode )
        {
            LinearConfig config( in_features, out_features );
            config.withBias( has_bias );

            auto linear = std::make_unique<LinearCpu>( "linear", config, Device::Cpu() );
            linear->build( BuildContext( shape, mode, false ) );

            return linear;
        }

        // Overwrite the (allocated) weight and bias with deterministic values.
        static void setKnownParameters(
            LinearCpu& linear, int64_t in_features, int64_t out_features, bool has_bias )
        {
            auto params = linear.getParameters();
            float* weight = static_cast<float*>( params[ 0 ]->rawData() );

            for ( int64_t o = 0; o < out_features; ++o )
            {
                for ( int64_t i = 0; i < in_features; ++i )
                {
                    weight[ o * in_features + i ] = weightValue( o, i );
                }
            }

            if ( has_bias )
            {
                float* bias = static_cast<float*>( params[ 1 ]->rawData() );

                for ( int64_t o = 0; o < out_features; ++o )
                {
                    bias[ o ] = biasValue( o );
                }
            }
        }

        static void fillSpread( TensorFp32& t )
        {
            auto* data = t.data();

            for ( dim_t i = 0; i < t.size(); ++i )
            {
                data[ i ] = static_cast<float>( i ) / t.size() * 4.0f - 2.0f;
            }
        }

        static int64_t leadingProduct( const shape_t& shape )
        {
            int64_t batch = 1;

            for ( size_t i = 0; i + 1 < shape.size(); ++i )
            {
                batch *= shape[ i ];
            }

            return batch;
        }
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( LinearCpuTests, Construct_StandaloneSucceeds )
    {
        LinearConfig config( 16, 32 );
        LinearCpu linear( "linear", config, Device::Cpu() );

        EXPECT_EQ( linear.getDeviceId().type, DeviceType::Cpu );
        EXPECT_TRUE( linear.hasBias() );
    }

    TEST_F( LinearCpuTests, Construct_InvalidConfigThrows )
    {
        LinearConfig bad( 0, 32 );

        EXPECT_THROW( LinearCpu( "linear", bad, Device::Cpu() ), std::invalid_argument );
    }

    TEST_F( LinearCpuTests, Construct_DeviceTypeMismatchThrows )
    {
        // A CUDA DeviceId on a CPU-typed component is rejected before any device
        // work, so this is safe on a host without a GPU.
        LinearConfig config( 16, 32 );

        EXPECT_THROW( LinearCpu( "linear", config, Device::Cuda( 0 ) ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle (Linear preconditions)
    // ====================================================================

    TEST_F( LinearCpuTests, Forward_ThrowsBeforeBuild )
    {
        LinearConfig config( 16, 32 );
        LinearCpu linear( "linear", config, Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ 2, 16 } );

        EXPECT_THROW( linear.forward( input ), std::runtime_error );
    }

    TEST_F( LinearCpuTests, Backward_ThrowsBeforeBuild )
    {
        LinearConfig config( 16, 32 );
        LinearCpu linear( "linear", config, Device::Cpu() );
        TensorFp32 input( Device::Cpu(), shape_t{ 2, 16 } );
        TensorFp32 output_grad( Device::Cpu(), shape_t{ 2, 32 } );

        EXPECT_THROW( linear.backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // C. Execution Context / Device
    // ====================================================================

    TEST_F( LinearCpuTests, Synchronize_Succeeds )
    {
        LinearConfig config( 16, 32 );
        LinearCpu linear( "linear", config, Device::Cpu() );

        EXPECT_NO_THROW( linear.synchronize() );
    }

    // ====================================================================
    // D. Runtime + Training Mode
    // ====================================================================

    TEST_F( LinearCpuTests, Backward_ThrowsWhenBuiltForInference )
    {
        const int64_t in_features = 4;
        const int64_t out_features = 3;
        const shape_t shape{ 2, in_features };

        auto linear = builtLinear( shape, in_features, out_features, true, RuntimeMode::Inference );
        setKnownParameters( *linear, in_features, out_features, true );

        TensorFp32 input( Device::Cpu(), shape );
        TensorFp32 output_grad( Device::Cpu(), shape_t{ 2, out_features } );
        fillSpread( input );

        linear->forward( input );

        EXPECT_THROW( linear->backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( LinearCpuTests, Forward_MatchesReference )
    {
        const int64_t in_features = 4;
        const int64_t out_features = 3;
        const shape_t shape{ 2, 3, in_features };

        auto linear = builtLinear( shape, in_features, out_features, true, RuntimeMode::Inference );
        setKnownParameters( *linear, in_features, out_features, true );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input );

        auto& output = linear->forward( input );

        const int64_t batch = leadingProduct( shape );
        ASSERT_EQ( output.size(), static_cast<size_t>( batch * out_features ) );

        std::vector<float> weight( out_features * in_features );
        std::vector<float> bias( out_features );
        for ( int64_t o = 0; o < out_features; ++o )
        {
            bias[ o ] = biasValue( o );
            for ( int64_t i = 0; i < in_features; ++i )
            {
                weight[ o * in_features + i ] = weightValue( o, i );
            }
        }

        std::vector<float> expected;
        referenceForward( input.data(), weight.data(), bias.data(), batch, in_features, out_features, expected );

        constexpr float tolerance = 1e-4f;

        for ( dim_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], expected[ i ], tolerance )
                << "forward mismatch at index " << i;
        }
    }

    TEST_F( LinearCpuTests, Forward_MatchesReferenceWithoutBias )
    {
        const int64_t in_features = 4;
        const int64_t out_features = 3;
        const shape_t shape{ 2, 3, in_features };

        auto linear = builtLinear( shape, in_features, out_features, false, RuntimeMode::Inference );
        setKnownParameters( *linear, in_features, out_features, false );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input );

        auto& output = linear->forward( input );

        const int64_t batch = leadingProduct( shape );

        std::vector<float> weight( out_features * in_features );
        for ( int64_t o = 0; o < out_features; ++o )
        {
            for ( int64_t i = 0; i < in_features; ++i )
            {
                weight[ o * in_features + i ] = weightValue( o, i );
            }
        }

        std::vector<float> expected;
        referenceForward( input.data(), weight.data(), nullptr, batch, in_features, out_features, expected );

        constexpr float tolerance = 1e-4f;

        for ( dim_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], expected[ i ], tolerance )
                << "no-bias forward mismatch at index " << i;
        }
    }

    TEST_F( LinearCpuTests, Forward_ExercisesUnrolledBatchPath )
    {
        // batch == 16 is a multiple of the CpuLinearOp LOOP_UNROLL factor (8),
        // selecting the unrolled kernel path that the small shapes above skip.
        const int64_t in_features = 4;
        const int64_t out_features = 3;
        const shape_t shape{ 16, in_features };

        auto linear = builtLinear( shape, in_features, out_features, true, RuntimeMode::Inference );
        setKnownParameters( *linear, in_features, out_features, true );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input );

        auto& output = linear->forward( input );

        std::vector<float> weight( out_features * in_features );
        std::vector<float> bias( out_features );
        for ( int64_t o = 0; o < out_features; ++o )
        {
            bias[ o ] = biasValue( o );
            for ( int64_t i = 0; i < in_features; ++i )
            {
                weight[ o * in_features + i ] = weightValue( o, i );
            }
        }

        std::vector<float> expected;
        referenceForward( input.data(), weight.data(), bias.data(), 16, in_features, out_features, expected );

        constexpr float tolerance = 1e-4f;

        for ( dim_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], expected[ i ], tolerance )
                << "unrolled-path mismatch at index " << i;
        }
    }

    TEST_F( LinearCpuTests, Forward_ThrowsOnInputFeatureMismatch )
    {
        const shape_t build_shape{ 2, 16 };
        auto linear = builtLinear( build_shape, 16, 32, true, RuntimeMode::Inference );

        TensorFp32 wrong_input( Device::Cpu(), shape_t{ 2, 64 } );

        EXPECT_THROW( linear->forward( wrong_input ), std::invalid_argument );
    }

    // ====================================================================
    // F. Backward (numeric vs analytic gradient)
    // ====================================================================

    TEST_F( LinearCpuTests, Backward_MatchesReferenceGradients )
    {
        const int64_t in_features = 4;
        const int64_t out_features = 3;
        const shape_t input_shape{ 2, 3, in_features };
        const shape_t output_shape{ 2, 3, out_features };

        auto linear = builtLinear( input_shape, in_features, out_features, true, RuntimeMode::Training );
        setKnownParameters( *linear, in_features, out_features, true );

        TensorFp32 input( Device::Cpu(), input_shape );
        TensorFp32 output_grad( Device::Cpu(), output_shape );
        fillSpread( input );

        for ( dim_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = 0.1f * static_cast<float>( i + 1 );
        }

        linear->forward( input );
        auto& input_grad = linear->backward( input, output_grad );

        const int64_t batch = leadingProduct( input_shape );
        ASSERT_EQ( input_grad.shape(), input_shape );

        std::vector<float> weight( out_features * in_features );
        for ( int64_t o = 0; o < out_features; ++o )
        {
            for ( int64_t i = 0; i < in_features; ++i )
            {
                weight[ o * in_features + i ] = weightValue( o, i );
            }
        }

        // dX[b,i] = sum_o dY[b,o] * W[o,i]
        std::vector<float> expected_dx( static_cast<size_t>( batch * in_features ), 0.0f );
        for ( int64_t b = 0; b < batch; ++b )
        {
            for ( int64_t i = 0; i < in_features; ++i )
            {
                double acc = 0.0;
                for ( int64_t o = 0; o < out_features; ++o )
                {
                    acc += static_cast<double>( output_grad.data()[ b * out_features + o ] ) *
                        static_cast<double>( weight[ o * in_features + i ] );
                }
                expected_dx[ b * in_features + i ] = static_cast<float>( acc );
            }
        }

        constexpr float tolerance = 1e-3f;

        for ( dim_t i = 0; i < input_grad.size(); ++i )
        {
            EXPECT_NEAR( input_grad.data()[ i ], expected_dx[ i ], tolerance )
                << "input-gradient mismatch at index " << i;
        }

        // dW[o,i] = sum_b dY[b,o] * X[b,i] ; dB[o] = sum_b dY[b,o]
        auto grads = linear->getGradients();
        ASSERT_EQ( grads.size(), 2u );
        const float* weight_grad = static_cast<const float*>( grads[ 0 ]->rawData() );
        const float* bias_grad = static_cast<const float*>( grads[ 1 ]->rawData() );

        for ( int64_t o = 0; o < out_features; ++o )
        {
            double db = 0.0;
            for ( int64_t b = 0; b < batch; ++b )
            {
                db += static_cast<double>( output_grad.data()[ b * out_features + o ] );
            }
            EXPECT_NEAR( bias_grad[ o ], static_cast<float>( db ), tolerance )
                << "bias-gradient mismatch at o=" << o;

            for ( int64_t i = 0; i < in_features; ++i )
            {
                double dw = 0.0;
                for ( int64_t b = 0; b < batch; ++b )
                {
                    dw += static_cast<double>( output_grad.data()[ b * out_features + o ] ) *
                        static_cast<double>( input.data()[ b * in_features + i ] );
                }
                EXPECT_NEAR( weight_grad[ o * in_features + i ], static_cast<float>( dw ), tolerance )
                    << "weight-gradient mismatch at o=" << o << " i=" << i;
            }
        }
    }

    // Reference application of the finite-difference gradient-check archetype for a
    // leaf WITH parameters (Specifications/Testing.md). Where the analytic case above
    // checks backward() against a hand-derived matmul gradient, this verifies dX, dW
    // and dB against numeric gradients of Linear's OWN forward() -- no hand-derived
    // reference, so it generalizes verbatim and is a second independent oracle.
    TEST_F( LinearCpuTests, Backward_MatchesNumericGradient )
    {
        const int64_t in_features = 4;
        const int64_t out_features = 3;
        const shape_t input_shape{ 2, 3, in_features };
        const shape_t output_shape{ 2, 3, out_features };

        auto linear = builtLinear( input_shape, in_features, out_features, true, RuntimeMode::Training );
        setKnownParameters( *linear, in_features, out_features, true );

        TensorFp32 input( Device::Cpu(), input_shape );
        TensorFp32 output_grad( Device::Cpu(), output_shape );
        fillSpread( input );
        for ( dim_t i = 0; i < output_grad.size(); ++i )
        {
            output_grad.data()[ i ] = 0.1f * static_cast<float>( i + 1 );
        }

        linear->forward( input );
        auto& input_grad = linear->backward( input, output_grad );

        // Snapshot analytic gradients before the probe re-runs forward().
        std::vector<float> analytic_dx( input_grad.data(), input_grad.data() + input_grad.size() );

        auto params = linear->getParameters();
        auto grads = linear->getGradients();
        ASSERT_EQ( grads.size(), 2u );
        float* weight = static_cast<float*>( params[ 0 ]->rawData() );
        float* bias = static_cast<float*>( params[ 1 ]->rawData() );
        const size_t weight_size = static_cast<size_t>( out_features * in_features );
        const float* weight_grad = static_cast<const float*>( grads[ 0 ]->rawData() );
        const float* bias_grad = static_cast<const float*>( grads[ 1 ]->rawData() );
        std::vector<float> analytic_dw( weight_grad, weight_grad + weight_size );
        std::vector<float> analytic_db( bias_grad, bias_grad + static_cast<size_t>( out_features ) );

        auto evaluate = [&]() -> const float* { return linear->forward( input ).data(); };

        const auto numeric_dx = Mila::Tests::Common::centralDifferenceGradient(
            input.data(), input.size(), output_grad.data(), output_grad.size(), evaluate, 1e-2f );
        Mila::Tests::Common::expectGradientsClose( analytic_dx.data(), numeric_dx, 1e-2f, 1e-2f, "Linear dX" );

        const auto numeric_dw = Mila::Tests::Common::centralDifferenceGradient(
            weight, weight_size, output_grad.data(), output_grad.size(), evaluate, 1e-2f );
        Mila::Tests::Common::expectGradientsClose( analytic_dw.data(), numeric_dw, 1e-2f, 1e-2f, "Linear dW" );

        const auto numeric_db = Mila::Tests::Common::centralDifferenceGradient(
            bias, static_cast<size_t>( out_features ), output_grad.data(), output_grad.size(), evaluate, 1e-2f );
        Mila::Tests::Common::expectGradientsClose( analytic_db.data(), numeric_db, 1e-2f, 1e-2f, "Linear dB" );
    }

    // ====================================================================
    // G. Parameters & Gradients
    // ====================================================================

    TEST_F( LinearCpuTests, ParameterCount_WithBias )
    {
        LinearConfig config( 16, 32 );
        LinearCpu linear( "linear", config, Device::Cpu() );
        linear.build( BuildContext( shape_t{ 2, 16 }, RuntimeMode::Inference, false ) );

        EXPECT_EQ( linear.parameterCount(), static_cast<size_t>( 16 * 32 + 32 ) );
    }

    TEST_F( LinearCpuTests, ParameterCount_WithoutBias )
    {
        LinearConfig config( 16, 32 );
        config.withBias( false );
        LinearCpu linear( "linear", config, Device::Cpu() );
        linear.build( BuildContext( shape_t{ 2, 16 }, RuntimeMode::Inference, false ) );

        EXPECT_EQ( linear.parameterCount(), static_cast<size_t>( 16 * 32 ) );
    }

    TEST_F( LinearCpuTests, GetParameters_WeightAndBias )
    {
        auto linear = builtLinear( shape_t{ 2, 16 }, 16, 32, true, RuntimeMode::Inference );

        EXPECT_EQ( linear->getParameters().size(), 2u );
    }

    TEST_F( LinearCpuTests, GetParameters_WeightOnlyWithoutBias )
    {
        auto linear = builtLinear( shape_t{ 2, 16 }, 16, 32, false, RuntimeMode::Inference );

        EXPECT_EQ( linear->getParameters().size(), 1u );
    }

    TEST_F( LinearCpuTests, GetGradients_EmptyWhenBuiltForInference )
    {
        auto linear = builtLinear( shape_t{ 2, 16 }, 16, 32, true, RuntimeMode::Inference );

        EXPECT_TRUE( linear->getGradients().empty() );
    }

    TEST_F( LinearCpuTests, GetGradients_PresentWhenBuiltForTraining )
    {
        auto linear = builtLinear( shape_t{ 2, 16 }, 16, 32, true, RuntimeMode::Training );

        EXPECT_EQ( linear->getGradients().size(), 2u );
    }

    // ====================================================================
    // G/H. Parameter load
    // ====================================================================

    TEST_F( LinearCpuTests, LoadParameter_SetsWeightAndBias )
    {
        const int64_t in_features = 4;
        const int64_t out_features = 3;
        const shape_t shape{ 2, in_features };

        auto linear = builtLinear( shape, in_features, out_features, true, RuntimeMode::Inference );

        std::vector<float> weight( out_features * in_features );
        std::vector<float> bias( out_features );
        for ( int64_t o = 0; o < out_features; ++o )
        {
            bias[ o ] = biasValue( o );
            for ( int64_t i = 0; i < in_features; ++i )
            {
                weight[ o * in_features + i ] = weightValue( o, i );
            }
        }

        TensorMetadata weight_meta{ TensorDataType::FP32, shape_t{ out_features, in_features }, weight.size() * sizeof( float ) };
        TensorBlobView weight_blob( weight_meta, weight.data(), weight.size() * sizeof( float ) );
        linear->loadParameter( "weight", weight_blob );

        TensorMetadata bias_meta{ TensorDataType::FP32, shape_t{ out_features }, bias.size() * sizeof( float ) };
        TensorBlobView bias_blob( bias_meta, bias.data(), bias.size() * sizeof( float ) );
        linear->loadParameter( "bias", bias_blob );

        TensorFp32 input( Device::Cpu(), shape );
        fillSpread( input );

        auto& output = linear->forward( input );

        std::vector<float> expected;
        referenceForward( input.data(), weight.data(), bias.data(), 2, in_features, out_features, expected );

        constexpr float tolerance = 1e-4f;

        for ( dim_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], expected[ i ], tolerance )
                << "loaded-weight forward mismatch at index " << i;
        }
    }

    TEST_F( LinearCpuTests, LoadParameter_ThrowsOnDtypeMismatch )
    {
        auto linear = builtLinear( shape_t{ 2, 4 }, 4, 3, true, RuntimeMode::Inference );

        std::vector<float> weight( 3 * 4, 0.0f );

        // Declaring the blob as BF16 while the unquantized weight expects FP32
        // must be rejected before any copy.
        TensorMetadata bad_meta{ TensorDataType::BF16, shape_t{ 3, 4 }, weight.size() * sizeof( float ) };
        TensorBlobView bad_blob( bad_meta, weight.data(), weight.size() * sizeof( float ) );

        EXPECT_THROW( linear->loadParameter( "weight", bad_blob ), std::invalid_argument );
    }

    TEST_F( LinearCpuTests, LoadParameter_ThrowsOnShapeMismatch )
    {
        auto linear = builtLinear( shape_t{ 2, 4 }, 4, 3, true, RuntimeMode::Inference );

        std::vector<float> weight( 5 * 4, 0.0f );

        TensorMetadata bad_meta{ TensorDataType::FP32, shape_t{ 5, 4 }, weight.size() * sizeof( float ) };
        TensorBlobView bad_blob( bad_meta, weight.data(), weight.size() * sizeof( float ) );

        EXPECT_THROW( linear->loadParameter( "weight", bad_blob ), std::invalid_argument );
    }

    TEST_F( LinearCpuTests, LoadParameter_ThrowsOnUnknownName )
    {
        auto linear = builtLinear( shape_t{ 2, 4 }, 4, 3, true, RuntimeMode::Inference );

        std::vector<float> data( 3 * 4, 0.0f );
        TensorMetadata meta{ TensorDataType::FP32, shape_t{ 3, 4 }, data.size() * sizeof( float ) };
        TensorBlobView blob( meta, data.data(), data.size() * sizeof( float ) );

        EXPECT_THROW( linear->loadParameter( "bogus", blob ), std::invalid_argument );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( LinearCpuTests, ToString_DescribesComponent )
    {
        LinearConfig config( 16, 32 );
        LinearCpu linear( "my_linear", config, Device::Cpu() );

        const std::string text = linear.toString();

        EXPECT_NE( text.find( "Linear" ), std::string::npos );
        EXPECT_NE( text.find( "my_linear" ), std::string::npos );
        EXPECT_NE( text.find( "Input features:" ), std::string::npos );
        EXPECT_NE( text.find( "Output features:" ), std::string::npos );
        EXPECT_NE( text.find( "Has Bias:" ), std::string::npos );
    }

    // ====================================================================
    // J. Type identity & accessors
    // ====================================================================

    TEST_F( LinearCpuTests, GetType_IsLinear )
    {
        LinearConfig config( 16, 32 );
        LinearCpu linear( "linear", config, Device::Cpu() );

        EXPECT_EQ( linear.getType(), ComponentType::Linear );
    }

    TEST_F( LinearCpuTests, HasBias_ReflectsConfig )
    {
        LinearConfig with_bias( 16, 32 );
        LinearCpu a( "a", with_bias, Device::Cpu() );
        EXPECT_TRUE( a.hasBias() );

        LinearConfig without_bias( 16, 32 );
        without_bias.withBias( false );
        LinearCpu b( "b", without_bias, Device::Cpu() );
        EXPECT_FALSE( b.hasBias() );
    }

    TEST_F( LinearCpuTests, GetConfig_ReturnsFeatureDimensions )
    {
        LinearConfig config( 16, 32 );
        LinearCpu linear( "linear", config, Device::Cpu() );

        EXPECT_EQ( linear.getConfig().getInputFeatures(), 16 );
        EXPECT_EQ( linear.getConfig().getOutputFeatures(), 32 );
    }
}
