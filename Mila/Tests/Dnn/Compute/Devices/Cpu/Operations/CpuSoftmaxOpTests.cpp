/**
 * @file CpuSoftmaxOpTests.cpp
 * @brief Test suite for the CPU Softmax operation.
 *
 * Tests exercise the CPU softmax operation using the modern ExecutionContext +
 * OperationRegistry APIs where available. If the SoftmaxOp factory is not
 * registered in the test build, op-level checks are skipped while the rest of
 * the tests still run where possible.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#ifdef USE_OMP
#include <omp.h>
#endif

import Mila;

namespace Dnn::Compute::Device::Cpu::Operations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CpuSoftmaxOp tests using modern ExecutionContext +
     * OperationRegistry pattern (when available).
     */
    class CpuSoftmaxOpTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Typed execution context for CPU tests
            exec_ctx_ = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

            // Attempt to create a default Softmax operation (CPU / FP32).
            // If registration is missing in this build, op_ remains null and
            // op-specific assertions are skipped. Use SoftmaxConfig at creation.
            try
            {
                SoftmaxConfig cfg;
                // Provide a name for clarity in diagnostics
                cfg.withName( "default_softmax" );
                op_ = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>(
                    "SoftmaxOp", exec_ctx_, cfg );
            }
            catch (const std::exception&)
            {
                op_.reset();
            }

            small_shape_ = { 2, 3, 4 };
            medium_shape_ = { 8, 16, 32 };
            large_shape_ = { 32, 64, 128 };
        }

        // Detect NaN/Inf in a host-accessible FP32 tensor
        template <typename TTensor>
        bool hasNaNorInfHost( const TTensor& tensor ) const
        {
            const float* data = static_cast<const float*>(tensor.rawData());

            for (size_t i = 0; i < tensor.size(); ++i)
            {
                if (std::isnan( data[i] ) || std::isinf( data[i] ))
                {
                    return true;
                }
            }

            return false;
        }

        // Helper method to compute softmax manually for verification
        static std::vector<float> computeSoftmaxRef( const std::vector<float>& input )
        {
            std::vector<float> out( input.size() );

            float maxv = input[0];
            for (float v : input) if (v > maxv) maxv = v;

            double sum = 0.0;
            for (size_t i = 0; i < input.size(); ++i)
            {
                out[i] = std::exp( input[i] - maxv );
                sum += out[i];
            }

            for (size_t i = 0; i < input.size(); ++i) out[i] = static_cast<float>( out[i] / sum );

            return out;
        }

        // Helper to verify probability distributions sum to 1 along axis
        bool checkProbabilityDistributionHost( const Tensor<TensorDataType::FP32, CpuMemoryResource>& tensor, int axis,
            float tol = 1e-5f ) const
        {
            int64_t ndim = static_cast<int64_t>( tensor.shape().size() );

            if (axis < 0) axis = static_cast<int>( ndim ) + axis;

            int64_t outer = 1;
            for (int i = 0; i < axis; ++i) outer *= static_cast<int64_t>( tensor.shape()[i] );

            int64_t dim = static_cast<int64_t>( tensor.shape()[axis] );

            int64_t inner = 1;
            for (int i = axis + 1; i < ndim; ++i) inner *= static_cast<int64_t>( tensor.shape()[i] );

            const float* data = static_cast<const float*>( tensor.rawData() );

            for (int64_t o = 0; o < outer; ++o)
            {
                for (int64_t in = 0; in < inner; ++in)
                {
                    double sum = 0.0;
                    for (int64_t d = 0; d < dim; ++d)
                    {
                        int64_t idx = (o * dim * inner) + (d * inner) + in;
                        sum += static_cast<double>( data[idx] );
                    }
                    if (std::abs( sum - 1.0 ) > tol) return false;
                }
            }

            return true;
        }

        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_ctx_;
        std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>> op_; // may be null if not registered

        shape_t small_shape_;
        shape_t medium_shape_;
        shape_t large_shape_;
    };

    TEST_F( CpuSoftmaxOpTests, OperationRegisteredName )
    {
        if (!op_)
        {
            GTEST_SKIP() << "SoftmaxOp not registered for CPU/FP32 in this build.";
        }

        EXPECT_EQ( op_->getName(), std::string( "Cpu::SoftmaxOp" ) );
    }

    TEST_F( CpuSoftmaxOpTests, BasicFunctionalityLastAxis )
    {
        // Create an op instance configured for axis=2 (last dimension)
        SoftmaxConfig cfg;
        cfg.withName( "basic_last_axis" ).withAxis( 2 );

        std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>> local_op;
        try
        {
            local_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>(
                "SoftmaxOp", exec_ctx_, cfg );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "SoftmaxOp not registered/configurable for CPU/FP32 in this build.";
        }

        using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        TensorT input( exec_ctx_->getDevice(), small_shape_ );
        TensorT output( exec_ctx_->getDevice(), small_shape_ );

        // Fill deterministic values
        float* in_ptr = static_cast<float*>(input.rawData());
        for (size_t i = 0; i < input.size(); ++i)
            in_ptr[i] = (static_cast<float>( i ) - 10.0f) / 10.0f;

        typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
        typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

        ASSERT_NO_THROW( local_op->forward( input, params, output, out_state ) );

        EXPECT_FALSE( hasNaNorInfHost( output ) );
        EXPECT_TRUE( checkProbabilityDistributionHost( output, 2 ) );

        // Verify a few distributions against reference implementation
        for (size_t b = 0; b < small_shape_[0]; ++b)
        {
            for (size_t t = 0; t < small_shape_[1]; ++t)
            {
                std::vector<float> in_vec;
                for (size_t c = 0; c < small_shape_[2]; ++c)
                {
                    size_t idx = (b * small_shape_[1] * small_shape_[2]) + (t * small_shape_[2]) + c;
                    in_vec.push_back( in_ptr[idx] );
                }

                auto expected = computeSoftmaxRef( in_vec );

                for (size_t c = 0; c < small_shape_[2]; ++c)
                {
                    size_t idx = (b * small_shape_[1] * small_shape_[2]) + (t * small_shape_[2]) + c;
                    float actual = static_cast<const float*>( output.rawData() )[idx];
                    EXPECT_NEAR( actual, expected[c], 1e-5f );
                }
            }
        }
    }

    TEST_F( CpuSoftmaxOpTests, AxisVariantsAndNegativeIndex )
    {
        // Create op instances per-axis so axis is part of the op configuration
        if (!exec_ctx_)
        {
            GTEST_SKIP() << "Execution context not available.";
        }

        using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        TensorT input( exec_ctx_->getDevice(), small_shape_ );
        TensorT output( exec_ctx_->getDevice(), small_shape_ );

        // deterministic fill
        float* in_ptr = static_cast<float*>(input.rawData());
        for (size_t i = 0; i < input.size(); ++i) in_ptr[i] = (static_cast<float>( i ) - 10.0f) / 10.0f;

        typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
        typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

        // Middle axis (1)
        {
            SoftmaxConfig cfg;
            cfg.withName( "axis1" ).withAxis( 1 );
            std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>> local_op;
            try
            {
                local_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>( "SoftmaxOp", exec_ctx_, cfg );
            }
            catch (const std::exception&)
            {
                GTEST_SKIP() << "SoftmaxOp (axis=1) not available";
            }

            ASSERT_NO_THROW( local_op->forward( input, params, output, out_state ) );
            EXPECT_FALSE( hasNaNorInfHost( output ) );
            EXPECT_TRUE( checkProbabilityDistributionHost( output, 1 ) );
        }

        // Negative axis (-1) => last
        {
            SoftmaxConfig cfg;
            cfg.withName( "axis_minus1" ).withAxis( -1 );
            std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>> local_op;
            try
            {
                local_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>( "SoftmaxOp", exec_ctx_, cfg );
            }
            catch (const std::exception&)
            {
                GTEST_SKIP() << "SoftmaxOp (axis=-1) not available";
            }

            ASSERT_NO_THROW( local_op->forward( input, params, output, out_state ) );
            EXPECT_FALSE( hasNaNorInfHost( output ) );
            EXPECT_TRUE( checkProbabilityDistributionHost( output, -1 ) );
        }

        // First axis (0)
        {
            SoftmaxConfig cfg;
            cfg.withName( "axis0" ).withAxis( 0 );
            std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>> local_op;
            try
            {
                local_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>( "SoftmaxOp", exec_ctx_, cfg );
            }
            catch (const std::exception&)
            {
                GTEST_SKIP() << "SoftmaxOp (axis=0) not available";
            }

            ASSERT_NO_THROW( local_op->forward( input, params, output, out_state ) );
            EXPECT_FALSE( hasNaNorInfHost( output ) );
            EXPECT_TRUE( checkProbabilityDistributionHost( output, 0 ) );
        }
    }

    TEST_F( CpuSoftmaxOpTests, EdgeCases )
    {
        SoftmaxConfig cfg;
        cfg.withName( "edge_cases" ).withAxis( 2 );

        std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>> local_op;
        try
        {
            local_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>( "SoftmaxOp", exec_ctx_, cfg );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "SoftmaxOp not available";
        }

        using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        TensorT input( exec_ctx_->getDevice(), small_shape_ );
        TensorT output( exec_ctx_->getDevice(), small_shape_ );

        typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
        typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

        // All zeros -> uniform distribution
        {
            float* in_ptr = static_cast<float*>(input.rawData());
            for (size_t i = 0; i < input.size(); ++i) in_ptr[i] = 0.0f;

            ASSERT_NO_THROW( local_op->forward( input, params, output, out_state ) );

            for (size_t b = 0; b < small_shape_[0]; ++b)
            {
                for (size_t t = 0; t < small_shape_[1]; ++t)
                {
                    float uniform = 1.0f / static_cast<float>( small_shape_[2] );
                    for (size_t c = 0; c < small_shape_[2]; ++c)
                    {
                        size_t idx = (b * small_shape_[1] * small_shape_[2]) + (t * small_shape_[2]) + c;
                        EXPECT_NEAR( static_cast<const float*>( output.rawData() )[idx], uniform, 1e-6f );
                    }
                }
            }
        }

        // Very large positive in one position -> approx one-hot
        {
            float* in_ptr = static_cast<float*>(input.rawData());
            for (size_t i = 0; i < input.size(); ++i) in_ptr[i] = -1000.0f;

            for (size_t b = 0; b < small_shape_[0]; ++b)
            {
                for (size_t t = 0; t < small_shape_[1]; ++t)
                {
                    size_t hot = (b + t) % small_shape_[2];
                    size_t idx = (b * small_shape_[1] * small_shape_[2]) + (t * small_shape_[2]) + hot;
                    in_ptr[idx] = 1000.0f;
                }
            }

            ASSERT_NO_THROW( local_op->forward( input, params, output, out_state ) );

            for (size_t b = 0; b < small_shape_[0]; ++b)
            {
                for (size_t t = 0; t < small_shape_[1]; ++t)
                {
                    size_t hot = (b + t) % small_shape_[2];
                    for (size_t c = 0; c < small_shape_[2]; ++c)
                    {
                        size_t idx = (b * small_shape_[1] * small_shape_[2]) + (t * small_shape_[2]) + c;
                        if (c == hot) EXPECT_NEAR( static_cast<const float*>( output.rawData() )[idx], 1.0f, 1e-5f );
                        else EXPECT_NEAR( static_cast<const float*>( output.rawData() )[idx], 0.0f, 1e-5f );
                    }
                }
            }
        }

        // Identical values -> uniform distribution
        {
            float* in_ptr = static_cast<float*>(input.rawData());
            for (size_t i = 0; i < input.size(); ++i) in_ptr[i] = 1.5f;

            ASSERT_NO_THROW( local_op->forward( input, params, output, out_state ) );

            for (size_t b = 0; b < small_shape_[0]; ++b)
            {
                for (size_t t = 0; t < small_shape_[1]; ++t)
                {
                    float uniform = 1.0f / static_cast<float>( small_shape_[2] );
                    for (size_t c = 0; c < small_shape_[2]; ++c)
                    {
                        size_t idx = (b * small_shape_[1] * small_shape_[2]) + (t * small_shape_[2]) + c;
                        EXPECT_NEAR( static_cast<const float*>( output.rawData() )[idx], uniform, 1e-5f );
                    }
                }
            }
        }
    }

    TEST_F( CpuSoftmaxOpTests, NumericalStabilityAndDeterminism )
    {
        SoftmaxConfig cfg;
        cfg.withName( "stability" ).withAxis( 2 );

        std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>> local_op;
        try
        {
            local_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>( "SoftmaxOp", exec_ctx_, cfg );
        }
        catch (const std::exception&)
        {
            GTEST_SKIP() << "SoftmaxOp not available";
        }

        using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        TensorT input( exec_ctx_->getDevice(), medium_shape_ );
        TensorT out1( exec_ctx_->getDevice(), medium_shape_ );
        TensorT out2( exec_ctx_->getDevice(), medium_shape_ );

        float* in_ptr = static_cast<float*>(input.rawData());
        for (size_t i = 0; i < input.size(); ++i)
        {
            int p = static_cast<int>( i % 8 );
            float v = 0.0f;
            switch (p)
            {
                case 0: v = 1.0f; break;
                case 1: v = -1.0f; break;
                case 2: v = 0.0001f; break;
                case 3: v = -0.0001f; break;
                case 4: v = 100.0f; break;
                case 5: v = -100.0f; break;
                case 6: v = 1000.0f; break;
                case 7: v = -1000.0f; break;
            }
            in_ptr[i] = v;
        }

        typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
        typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

        ASSERT_NO_THROW( local_op->forward( input, params, out1, out_state ) );
        EXPECT_FALSE( hasNaNorInfHost( out1 ) );

        local_op->forward( input, params, out2, out_state );

        for (size_t i = 0; i < out1.size(); ++i)
            EXPECT_EQ( static_cast<const float*>( out1.rawData() )[i], static_cast<const float*>( out2.rawData() )[i] );
    }

    TEST_F( CpuSoftmaxOpTests, InvalidAxis )
    {
        // Validate that constructing an op with an invalid axis triggers validation.
        // If the factory does not perform validation, this test may need to be adapted.
        if (!exec_ctx_)
        {
            GTEST_SKIP() << "Execution context not available.";
        }

        SoftmaxConfig cfg_pos;
        cfg_pos.withName( "invalid_pos" ).withAxis( static_cast<int>(small_shape_.size()) ); // out of bounds

        EXPECT_THROW( ( OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>( "SoftmaxOp", exec_ctx_, cfg_pos ) ),
            std::invalid_argument );

        SoftmaxConfig cfg_neg;
        cfg_neg.withName( "invalid_neg" ).withAxis( -static_cast<int>(small_shape_.size()) - 1 ); // out of bounds negative

        EXPECT_THROW( ( OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>( "SoftmaxOp", exec_ctx_, cfg_neg ) ),
            std::invalid_argument );
    }

    TEST_F( CpuSoftmaxOpTests, PerformanceAndOpenMPScaling )
    {
        if (std::getenv( "CI" ) != nullptr)
        {
            GTEST_SKIP() << "Skipping performance tests in CI environment";
        }

        if (!op_)
        {
            GTEST_SKIP() << "SoftmaxOp not registered; skipping perf/OpenMP op-level tests.";
        }

        // Use the op_ created in SetUp (default config) for perf runs
        using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        TensorT a( exec_ctx_->getDevice(), large_shape_ ), out( exec_ctx_->getDevice(), large_shape_ );
        float* ad = static_cast<float*>(a.rawData());

        for (size_t i = 0; i < a.size(); ++i)
            ad[i] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;

        typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
        typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

        const int iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();

        for (int it = 0; it < iterations; ++it)
            op_->forward( a, params, out, out_state );

        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>( end - start );
        size_t total = a.size() * iterations;
        double eps = static_cast<double>( total ) / (dur.count() * 1e-6);
        std::cout << "CpuSoftmaxOp: " << eps / 1e6 << " M elems/sec\n";

#ifdef USE_OMP
        int max_threads = omp_get_max_threads();
        std::vector<int> threads = { 1 };
        if (max_threads > 1)
        {
            threads.push_back( max_threads );
            if (max_threads > 3) threads.push_back( max_threads / 2 );
        }

        for (int num_threads : threads)
        {
            omp_set_num_threads( num_threads );
            auto s = std::chrono::high_resolution_clock::now();

            for (int it = 0; it < iterations; ++it)
                op_->forward( a, params, out, out_state );

            auto e = std::chrono::high_resolution_clock::now();
            auto d = std::chrono::duration_cast<std::chrono::microseconds>( e - s );
            double eps_t = static_cast<double>( a.size() * iterations ) / (d.count() * 1e-6);
            std::cout << "CpuSoftmaxOp with " << num_threads << " threads: " << eps_t / 1e6 << " M elems/sec\n";
        }
#else
        GTEST_SKIP() << "OpenMP not available, skipping OpenMP scaling test";
#endif
    }
} // namespace Dnn::Compute::Device::Cpu::Operations::Tests