/**
 * @file Random.Cpu.cpp
 * @brief Tests for the TensorOps random-initialization area (CPU): fill_uniform,
 *        fill_normal, xavier.
 *
 * These are the LIVE train-from-scratch init entry points (module Dnn.TensorOps:Random)
 * that Linear::initializeParameters drives for weight/bias setup -- the MNIST/Bard
 * training init path. Distinct from the obsolete Dnn.TensorInitializers free-function
 * layer (random/normal/ones), which is commented out in Src pending re-authoring; the
 * old Tensors/Tensor.Initializers.cpp targets that dead module and stays retired.
 *
 * The ops are float-only (requires is_float_type). CPU generation is host-side
 * std::mt19937 + std distributions seeded from Core::RandomGenerator, converted to the
 * tensor's native element type. FP32 here is the MNIST path; BF16 is addable once a
 * CPU read-as-float helper exists (the CPU backend already converts element-wise).
 *
 * Value-type / god-module archetype (see Specifications/Testing.md): the TensorOps
 * area split, exhaustive against fill_uniform / fill_normal / xavier. CPU device, so
 * this rides the MILA_ENABLE_CUDA=OFF CI gate; the CUDA cuRAND path (and the known
 * BF16 fill_normal gap) belong in a Random.Cuda.cpp companion.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <limits>

import Mila;
// Core.RandomGenerator is not part of the public umbrella; import it directly to
// drive the reproducibility (seed) contract.
import Core.RandomGenerator;

namespace Mila::Tests::Dnn::Tensors::TensorOps
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using TensorFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
    }

    class RandomCpuTests : public ::testing::Test
    {
    protected:
        // A 100x100 tensor gives 10k samples -- enough for tight statistical bounds.
        static TensorFp32 makeTensor()
        {
            return TensorFp32( Device::Cpu(), shape_t{ 100, 100 } );
        }
    };

    // ====================================================================
    // fill_uniform
    // ====================================================================

    TEST_F( RandomCpuTests, FillUniform_ValuesWithinRange )
    {
        auto tensor = makeTensor();

        const float min_val = -1.0f;
        const float max_val = 1.0f;
        fill_uniform( tensor, min_val, max_val );

        const float* data = tensor.data();
        float actual_min = std::numeric_limits<float>::max();
        float actual_max = std::numeric_limits<float>::lowest();

        for ( size_t i = 0; i < tensor.size(); ++i )
        {
            EXPECT_GE( data[ i ], min_val );
            EXPECT_LE( data[ i ], max_val );
            actual_min = std::min( actual_min, data[ i ] );
            actual_max = std::max( actual_max, data[ i ] );
        }

        // The sample should span most of the range (not collapse to a point).
        EXPECT_LT( actual_min, min_val + 0.2f );
        EXPECT_GT( actual_max, max_val - 0.2f );
    }

    TEST_F( RandomCpuTests, FillUniform_EmptyTensorIsNoOp )
    {
        TensorFp32 empty( Device::Cpu(), shape_t{ 0 } );

        EXPECT_NO_THROW( fill_uniform( empty, -1.0f, 1.0f ) );
    }

    // ====================================================================
    // fill_normal
    // ====================================================================

    TEST_F( RandomCpuTests, FillNormal_MatchesRequestedMomentsAndIsFinite )
    {
        auto tensor = makeTensor();

        const float mean = 0.0f;
        const float stddev = 1.0f;
        fill_normal( tensor, mean, stddev );

        const float* data = tensor.data();
        const size_t n = tensor.size();

        double sum = 0.0;
        for ( size_t i = 0; i < n; ++i )
        {
            EXPECT_FALSE( std::isnan( data[ i ] ) );
            EXPECT_FALSE( std::isinf( data[ i ] ) );
            sum += data[ i ];
        }

        const double sample_mean = sum / static_cast<double>( n );

        double variance = 0.0;
        for ( size_t i = 0; i < n; ++i )
        {
            const double d = data[ i ] - sample_mean;
            variance += d * d;
        }
        const double sample_stddev = std::sqrt( variance / static_cast<double>( n ) );

        EXPECT_NEAR( sample_mean, mean, 0.1 );
        EXPECT_NEAR( sample_stddev, stddev, 0.1 );
    }

    TEST_F( RandomCpuTests, FillNormal_RespectsNonZeroMeanAndStddev )
    {
        auto tensor = makeTensor();

        const float mean = 5.0f;
        const float stddev = 2.0f;
        fill_normal( tensor, mean, stddev );

        const float* data = tensor.data();
        const size_t n = tensor.size();

        double sum = 0.0;
        for ( size_t i = 0; i < n; ++i )
        {
            sum += data[ i ];
        }
        const double sample_mean = sum / static_cast<double>( n );

        double variance = 0.0;
        for ( size_t i = 0; i < n; ++i )
        {
            const double d = data[ i ] - sample_mean;
            variance += d * d;
        }
        const double sample_stddev = std::sqrt( variance / static_cast<double>( n ) );

        EXPECT_NEAR( sample_mean, mean, 0.2 );
        EXPECT_NEAR( sample_stddev, stddev, 0.2 );
    }

    TEST_F( RandomCpuTests, FillNormal_EmptyTensorIsNoOp )
    {
        TensorFp32 empty( Device::Cpu(), shape_t{ 0 } );

        EXPECT_NO_THROW( fill_normal( empty, 0.0f, 1.0f ) );
    }

    // ====================================================================
    // xavier (Glorot uniform: U(-limit, limit), limit = sqrt(6/(fan_in+fan_out)))
    // ====================================================================

    TEST_F( RandomCpuTests, Xavier_ValuesWithinGlorotBound )
    {
        auto tensor = makeTensor();

        const size_t fan_in = 784;   // MNIST input
        const size_t fan_out = 10;   // MNIST output classes
        xavier( tensor, fan_in, fan_out );

        const float limit = std::sqrt( 6.0f / static_cast<float>( fan_in + fan_out ) );

        const float* data = tensor.data();
        float actual_min = std::numeric_limits<float>::max();
        float actual_max = std::numeric_limits<float>::lowest();

        for ( size_t i = 0; i < tensor.size(); ++i )
        {
            EXPECT_GE( data[ i ], -limit );
            EXPECT_LE( data[ i ], limit );
            actual_min = std::min( actual_min, data[ i ] );
            actual_max = std::max( actual_max, data[ i ] );
        }

        EXPECT_LT( actual_min, -0.6f * limit );
        EXPECT_GT( actual_max, 0.6f * limit );
    }

    TEST_F( RandomCpuTests, Xavier_LargeFanInProducesSmallBound )
    {
        auto tensor = makeTensor();

        const size_t fan_in = 10000;
        const size_t fan_out = 1;
        xavier( tensor, fan_in, fan_out );

        const float limit = std::sqrt( 6.0f / static_cast<float>( fan_in + fan_out ) );
        EXPECT_LT( limit, 0.1f );

        const float* data = tensor.data();
        for ( size_t i = 0; i < tensor.size(); ++i )
        {
            EXPECT_GE( data[ i ], -limit );
            EXPECT_LE( data[ i ], limit );
        }
    }

    // ====================================================================
    // Reproducibility (seed contract)
    // ====================================================================

    TEST_F( RandomCpuTests, FillUniform_SameSeedProducesSameValues )
    {
        auto tensor1 = makeTensor();
        auto tensor2 = makeTensor();

        Mila::Core::RandomGenerator::getInstance().setSeed( 12345 );
        fill_uniform( tensor1, -1.0f, 1.0f );

        Mila::Core::RandomGenerator::getInstance().setSeed( 12345 );
        fill_uniform( tensor2, -1.0f, 1.0f );

        const float* a = tensor1.data();
        const float* b = tensor2.data();
        for ( size_t i = 0; i < tensor1.size(); ++i )
        {
            EXPECT_FLOAT_EQ( a[ i ], b[ i ] ) << "same seed should reproduce values at index " << i;
        }
    }
}
