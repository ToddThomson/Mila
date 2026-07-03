/**
 * @file Residual.Cpu.cpp
 * @brief Concrete-component tests for Residual<DeviceType::Cpu, FP32>.
 *
 * Concrete-component archetype for a stateless BINARY leaf (see
 * Specifications/Testing.md). Residual computes out = a + b (Addition,
 * scaling 1.0) and backward returns the pair {da, db}; for plain addition both
 * input gradients equal the upstream gradient. CPU is FP32-only.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

#include "Common/GradientCheck.h"

import Mila;

namespace Mila::Tests::Dnn::Components::Connections
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using ResidualCpu = Mila::Dnn::Residual<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( ResidualCpu::getDeviceType() == DeviceType::Cpu );
        static_assert( ResidualCpu::getPrecision() == TensorDataType::FP32 );

        void fillRamp( TensorFp32& t, float start, float step )
        {
            for ( size_t i = 0; i < t.size(); ++i )
            {
                t.data()[ i ] = start + step * static_cast<float>( i );
            }
        }
    }

    class ResidualCpuTests : public ::testing::Test
    {
    protected:
        std::unique_ptr<ResidualCpu> builtResidual( const shape_t& shape, RuntimeMode mode )
        {
            auto residual = std::make_unique<ResidualCpu>( "residual", ResidualConfig(), Device::Cpu() );
            residual->build( BuildContext( shape, mode, false ) );

            return residual;
        }
    };

    // ====================================================================
    // A. Construction
    // ====================================================================

    TEST_F( ResidualCpuTests, Construct_StandaloneSucceeds )
    {
        ResidualCpu residual( "residual", ResidualConfig(), Device::Cpu() );

        EXPECT_EQ( residual.getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( ResidualCpuTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( ResidualCpu( "residual", ResidualConfig(), Device::Cuda( 0 ) ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( ResidualCpuTests, Forward_ThrowsBeforeBuild )
    {
        ResidualCpu residual( "residual", ResidualConfig(), Device::Cpu() );
        TensorFp32 a( Device::Cpu(), shape_t{ 2, 4 } );
        TensorFp32 b( Device::Cpu(), shape_t{ 2, 4 } );

        EXPECT_THROW( residual.forward( a, b ), std::runtime_error );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( ResidualCpuTests, Forward_MatchesSum )
    {
        const shape_t shape{ 2, 3, 4 };
        auto residual = builtResidual( shape, RuntimeMode::Inference );

        TensorFp32 a( Device::Cpu(), shape );
        TensorFp32 b( Device::Cpu(), shape );
        fillRamp( a, -1.0f, 0.1f );
        fillRamp( b, 0.5f, -0.05f );

        auto& output = residual->forward( a, b );

        ASSERT_EQ( output.size(), a.size() );

        for ( size_t i = 0; i < output.size(); ++i )
        {
            EXPECT_NEAR( output.data()[ i ], a.data()[ i ] + b.data()[ i ], 1e-5f )
                << "sum mismatch at index " << i;
        }
    }

    // ====================================================================
    // D/F. Backward (training mode + gradient routing)
    // ====================================================================

    TEST_F( ResidualCpuTests, Backward_ThrowsWhenBuiltForInference )
    {
        const shape_t shape{ 2, 4 };
        auto residual = builtResidual( shape, RuntimeMode::Inference );

        TensorFp32 a( Device::Cpu(), shape );
        TensorFp32 b( Device::Cpu(), shape );
        TensorFp32 grad( Device::Cpu(), shape );

        residual->forward( a, b );

        EXPECT_THROW( residual->backward( a, b, grad ), std::runtime_error );
    }

    TEST_F( ResidualCpuTests, Backward_RoutesGradientToBothInputs )
    {
        const shape_t shape{ 2, 3, 4 };
        auto residual = builtResidual( shape, RuntimeMode::Training );

        TensorFp32 a( Device::Cpu(), shape );
        TensorFp32 b( Device::Cpu(), shape );
        TensorFp32 grad( Device::Cpu(), shape );
        fillRamp( a, -1.0f, 0.1f );
        fillRamp( b, 0.5f, -0.05f );
        fillRamp( grad, 0.2f, 0.03f );

        residual->forward( a, b );
        auto [da, db] = residual->backward( a, b, grad );

        ASSERT_EQ( da.size(), grad.size() );
        ASSERT_EQ( db.size(), grad.size() );

        // d(a+b)/da = d(a+b)/db = 1, so both gradients equal the upstream gradient.
        for ( size_t i = 0; i < grad.size(); ++i )
        {
            EXPECT_NEAR( da.data()[ i ], grad.data()[ i ], 1e-5f ) << "da mismatch at " << i;
            EXPECT_NEAR( db.data()[ i ], grad.data()[ i ], 1e-5f ) << "db mismatch at " << i;
        }
    }

    // Finite-difference gradient-check archetype for a BINARY leaf
    // (Specifications/Testing.md): the shared helper verifies each input gradient
    // against a numeric gradient of the component's own forward(), perturbing one
    // input while the other is held fixed. For plain addition both gradients equal
    // the upstream gradient, but this exercises the archetype's binary adaptation.
    TEST_F( ResidualCpuTests, Backward_MatchesNumericGradient )
    {
        const shape_t shape{ 2, 3, 4 };
        auto residual = builtResidual( shape, RuntimeMode::Training );

        TensorFp32 a( Device::Cpu(), shape );
        TensorFp32 b( Device::Cpu(), shape );
        TensorFp32 grad( Device::Cpu(), shape );
        fillRamp( a, -1.0f, 0.1f );
        fillRamp( b, 0.5f, -0.05f );
        fillRamp( grad, 0.2f, 0.03f );

        residual->forward( a, b );
        auto [da, db] = residual->backward( a, b, grad );

        // Snapshot analytic gradients before the probe re-runs forward().
        std::vector<float> analytic_da( da.data(), da.data() + da.size() );
        std::vector<float> analytic_db( db.data(), db.data() + db.size() );

        auto evaluate = [&]() -> const float* { return residual->forward( a, b ).data(); };

        const auto numeric_da = Mila::Tests::Common::centralDifferenceGradient(
            a.data(), a.size(), grad.data(), grad.size(), evaluate, 1e-2f );
        Mila::Tests::Common::expectGradientsClose( analytic_da.data(), numeric_da, 1e-3f, 1e-2f, "Residual da" );

        const auto numeric_db = Mila::Tests::Common::centralDifferenceGradient(
            b.data(), b.size(), grad.data(), grad.size(), evaluate, 1e-2f );
        Mila::Tests::Common::expectGradientsClose( analytic_db.data(), numeric_db, 1e-3f, 1e-2f, "Residual db" );
    }

    // ====================================================================
    // G. Parameters (Residual is stateless)
    // ====================================================================

    TEST_F( ResidualCpuTests, Parameters_AreEmpty )
    {
        auto residual = builtResidual( shape_t{ 2, 4 }, RuntimeMode::Inference );

        EXPECT_EQ( residual->parameterCount(), 0u );
        EXPECT_TRUE( residual->getParameters().empty() );
        EXPECT_TRUE( residual->getGradients().empty() );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( ResidualCpuTests, GetType_IsResidual )
    {
        ResidualCpu residual( "residual", ResidualConfig(), Device::Cpu() );

        EXPECT_EQ( residual.getType(), ComponentType::Residual );
    }
}
