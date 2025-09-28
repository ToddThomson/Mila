/**
 * @file CpuLinearOpTests.cpp
 * @brief Test suite for the CPU Fully Connected operation.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <iostream>
#include <cmath>
#ifdef USE_OMP
#include <omp.h>
#endif

import Mila;

namespace Compute::Cpu::Operations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CpuLinearOp tests
     */
    class CpuLinearOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Small shapes for quick tests
            small_batch_ = 2;
            small_seq_len_ = 3;
            small_in_features_ = 4;
            small_out_features_ = 5;
            small_input_shape_ = { small_batch_, small_seq_len_, small_in_features_ };
            small_output_shape_ = { small_batch_, small_seq_len_, small_out_features_ };
            small_weight_shape_ = { small_out_features_, small_in_features_ };
            small_bias_shape_ = { small_out_features_ };

            // Medium shapes for more thorough tests
            medium_batch_ = 8;
            medium_seq_len_ = 16;
            medium_in_features_ = 32;
            medium_out_features_ = 24;
            medium_input_shape_ = { medium_batch_, medium_seq_len_, medium_in_features_ };
            medium_output_shape_ = { medium_batch_, medium_seq_len_, medium_out_features_ };
            medium_weight_shape_ = { medium_out_features_, medium_in_features_ };
            medium_bias_shape_ = { medium_out_features_ };

            // Large shapes for stress tests
            large_batch_ = 32;
            large_seq_len_ = 64;
            large_in_features_ = 128;
            large_out_features_ = 96;
            large_input_shape_ = { large_batch_, large_seq_len_, large_in_features_ };
            large_output_shape_ = { large_batch_, large_seq_len_, large_out_features_ };
            large_weight_shape_ = { large_out_features_, large_in_features_ };
            large_bias_shape_ = { large_out_features_ };
        }

        std::shared_ptr<CpuLinearOp> createLinearOp( size_t in_features, size_t out_features, bool has_bias = true ) {
            LinearConfig config( in_features, out_features );
            if ( !has_bias ) {
                config.withBias( false );
            }
            return std::make_shared<CpuLinearOp>( cpu_context_, config );
        }

        bool hasNaNorInf( const Tensor<float, HostMemoryResource>& tensor ) {
            for ( size_t i = 0; i < tensor.size(); ++i ) {
                if ( std::isnan( tensor.data()[ i ] ) || std::isinf( tensor.data()[ i ] ) ) {
                    std::cout << "Found NaN or Inf at index " << i << ": " << tensor.data()[ i ] << std::endl;
                    return true;
                }
            }
            return false;
        }

        void referenceLinear(
            const Tensor<float, HostMemoryResource>& input,
            const Tensor<float, HostMemoryResource>& weights,
            const Tensor<float, HostMemoryResource>* bias,
            Tensor<float, HostMemoryResource>& output ) {

            int B = input.shape()[ 0 ];   // Batch size
            int T = input.shape()[ 1 ];   // Sequence length
            int C = input.shape()[ 2 ];   // Input features
            int OC = output.shape()[ 2 ]; // Output features

            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    int bt = b * T + t;
                    for ( int o = 0; o < OC; o++ ) {
                        float val = bias ? bias->data()[ o ] : 0.0f;
                        for ( int i = 0; i < C; i++ ) {
                            val += input.data()[ bt * C + i ] * weights.data()[ i + o * C ];
                        }
                        output.data()[ bt * OC + o ] = val;
                    }
                }
            }
        }

        void initializeTensors(
            Tensor<float, HostMemoryResource>& input,
            Tensor<float, HostMemoryResource>& weights,
            Tensor<float, HostMemoryResource>* bias = nullptr ) {

            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = (static_cast<float>( i % 17 ) - 8.0f) / 8.0f;
            }

            for ( size_t i = 0; i < weights.size(); ++i ) {
                weights.data()[ i ] = (static_cast<float>( i % 13 ) - 6.0f) / 12.0f;
            }

            if ( bias ) {
                for ( size_t i = 0; i < bias->size(); ++i ) {
                    bias->data()[ i ] = (static_cast<float>( i % 7 ) - 3.0f) / 6.0f;
                }
            }
        }

        void initializeWeightsAndBias(
            Tensor<float, HostMemoryResource>& weights,
            Tensor<float, HostMemoryResource>* bias = nullptr ) {

            for ( size_t i = 0; i < weights.size(); ++i ) {
                weights.data()[ i ] = (static_cast<float>( i % 13 ) - 6.0f) / 12.0f;
            }

            if ( bias ) {
                for ( size_t i = 0; i < bias->size(); ++i ) {
                    bias->data()[ i ] = (static_cast<float>( i % 7 ) - 3.0f) / 6.0f;
                }
            }
        }

        std::shared_ptr<DeviceContext> cpu_context_;

        size_t small_batch_, small_seq_len_, small_in_features_, small_out_features_;
        size_t medium_batch_, medium_seq_len_, medium_in_features_, medium_out_features_;
        size_t large_batch_, large_seq_len_, large_in_features_, large_out_features_;

        std::vector<size_t> small_input_shape_, small_output_shape_, small_weight_shape_, small_bias_shape_;
        std::vector<size_t> medium_input_shape_, medium_output_shape_, medium_weight_shape_, medium_bias_shape_;
        std::vector<size_t> large_input_shape_, large_output_shape_, large_weight_shape_, large_bias_shape_;
    };

    /**
     * @brief Test CpuLinearOp name property
     */
    TEST_F( CpuLinearOpTests, Name ) {
        auto op = createLinearOp( small_in_features_, small_out_features_ );
        EXPECT_EQ( op->getName(), "Cpu::LinearOp" );
    }

    /**
     * @brief Test basic functionality without bias
     */
    TEST_F( CpuLinearOpTests, BasicFunctionalityWithoutBias ) {
        auto op = createLinearOp( small_in_features_, small_out_features_, false );

        Tensor<float, HostMemoryResource> input( small_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( small_weight_shape_ );
        Tensor<float, HostMemoryResource> output( small_output_shape_ );
        Tensor<float, HostMemoryResource> expected_output( small_output_shape_ );

        initializeTensors( input, *weights );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        referenceLinear( input, *weights, nullptr, expected_output );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
        }
    }

    /**
     * @brief Test basic functionality with bias
     */
    TEST_F( CpuLinearOpTests, BasicFunctionalityWithBias ) {
        auto op = createLinearOp( small_in_features_, small_out_features_ );

        Tensor<float, HostMemoryResource> input( small_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( small_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( small_bias_shape_ );
        Tensor<float, HostMemoryResource> output( small_output_shape_ );
        Tensor<float, HostMemoryResource> expected_output( small_output_shape_ );

        initializeTensors( input, *weights, bias.get() );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        referenceLinear( input, *weights, bias.get(), expected_output );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
        }
    }

    /**
     * @brief Test with medium-sized tensors
     */
    TEST_F( CpuLinearOpTests, MediumSizeTensors ) {
        auto op = createLinearOp( medium_in_features_, medium_out_features_ );

        Tensor<float, HostMemoryResource> input( medium_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( medium_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( medium_bias_shape_ );
        Tensor<float, HostMemoryResource> output( medium_output_shape_ );
        Tensor<float, HostMemoryResource> expected_output( medium_output_shape_ );

        initializeTensors( input, *weights, bias.get() );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        referenceLinear( input, *weights, bias.get(), expected_output );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
        }
    }

    /**
     * @brief Test optimized path (divisible by LOOP_UNROLL=8)
     */
    TEST_F( CpuLinearOpTests, OptimizedPathDivisibleBy8 ) {
        std::vector<size_t> opt_input_shape = { 4, 2, small_in_features_ };   // 4*2 = 8, divisible by 8
        std::vector<size_t> opt_output_shape = { 4, 2, small_out_features_ };

        auto op = createLinearOp( small_in_features_, small_out_features_ );

        Tensor<float, HostMemoryResource> input( opt_input_shape );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( small_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( small_bias_shape_ );
        Tensor<float, HostMemoryResource> output( opt_output_shape );
        Tensor<float, HostMemoryResource> expected_output( opt_output_shape );

        initializeTensors( input, *weights, bias.get() );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        referenceLinear( input, *weights, bias.get(), expected_output );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
        }
    }

    /**
     * @brief Test naive path (not divisible by LOOP_UNROLL=8)
     */
    TEST_F( CpuLinearOpTests, NaivePathNotDivisibleBy8 ) {
        std::vector<size_t> naive_input_shape = { 3, 5, small_in_features_ };   // 3*5 = 15, not divisible by 8
        std::vector<size_t> naive_output_shape = { 3, 5, small_out_features_ };

        auto op = createLinearOp( small_in_features_, small_out_features_ );

        Tensor<float, HostMemoryResource> input( naive_input_shape );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( small_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( small_bias_shape_ );
        Tensor<float, HostMemoryResource> output( naive_output_shape );
        Tensor<float, HostMemoryResource> expected_output( naive_output_shape );

        initializeTensors( input, *weights, bias.get() );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        referenceLinear( input, *weights, bias.get(), expected_output );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
        }
    }

    /**
     * @brief Test edge cases with different input dimensions
     */
    TEST_F( CpuLinearOpTests, DifferentInputDimensions ) {
        std::vector<size_t> input_2d_shape = { 4, small_in_features_ };
        std::vector<size_t> output_2d_shape = { 4, small_out_features_ };

        auto op = createLinearOp( small_in_features_, small_out_features_ );

        Tensor<float, HostMemoryResource> input( input_2d_shape );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( small_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( small_bias_shape_ );
        Tensor<float, HostMemoryResource> output( output_2d_shape );

        initializeTensors( input, *weights, bias.get() );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        std::vector<size_t> input_4d_shape = { 2, 3, 4, small_in_features_ };
        std::vector<size_t> output_4d_shape = { 2, 3, 4, small_out_features_ };

        Tensor<float, HostMemoryResource> input_4d( input_4d_shape );
        Tensor<float, HostMemoryResource> output_4d( output_4d_shape );

        initializeTensors( input_4d, *weights, bias.get() );

        ASSERT_NO_THROW( op->forward( input_4d, parameters, output_4d, output_state ) );
    }

    /**
     * @brief Test error handling for invalid inputs
     */
    TEST_F( CpuLinearOpTests, ErrorHandling ) {
        auto op = createLinearOp( small_in_features_, small_out_features_ );

        std::vector<size_t> input_1d_shape = { small_in_features_ };
        Tensor<float, HostMemoryResource> input_1d( input_1d_shape );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( small_weight_shape_ );
        Tensor<float, HostMemoryResource> output( small_output_shape_ );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        EXPECT_THROW( op->forward( input_1d, parameters, output, output_state ), std::runtime_error );
    }

    /**
     * @brief Test with all-zero inputs
     */
    TEST_F( CpuLinearOpTests, ZeroInputs ) {
        auto op = createLinearOp( small_in_features_, small_out_features_ );

        Tensor<float, HostMemoryResource> input( small_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( small_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( small_bias_shape_ );
        Tensor<float, HostMemoryResource> output( small_output_shape_ );

        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 0.0f;
        }

        initializeWeightsAndBias( *weights, bias.get() );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        for ( size_t b = 0; b < small_batch_; ++b ) {
            for ( size_t t = 0; t < small_seq_len_; ++t ) {
                for ( size_t o = 0; o < small_out_features_; ++o ) {
                    size_t idx = (b * small_seq_len_ + t) * small_out_features_ + o;
                    EXPECT_FLOAT_EQ( output.data()[ idx ], bias->data()[ o ] );
                }
            }
        }
    }

    /**
     * @brief Test with all-zero weights
     */
    TEST_F( CpuLinearOpTests, ZeroWeights ) {
        auto op = createLinearOp( small_in_features_, small_out_features_ );

        Tensor<float, HostMemoryResource> input( small_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( small_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( small_bias_shape_ );
        Tensor<float, HostMemoryResource> output( small_output_shape_ );

        initializeTensors( input, *weights, bias.get() );

        for ( size_t i = 0; i < weights->size(); ++i ) {
            weights->data()[ i ] = 0.0f;
        }

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        for ( size_t b = 0; b < small_batch_; ++b ) {
            for ( size_t t = 0; t < small_seq_len_; ++t ) {
                for ( size_t o = 0; o < small_out_features_; ++o ) {
                    size_t idx = (b * small_seq_len_ + t) * small_out_features_ + o;
                    EXPECT_FLOAT_EQ( output.data()[ idx ], bias->data()[ o ] );
                }
            }
        }
    }

    /**
     * @brief Test numerical stability with extreme values
     */
    TEST_F( CpuLinearOpTests, NumericalStability ) {
        auto op = createLinearOp( medium_in_features_, medium_out_features_ );

        Tensor<float, HostMemoryResource> input( medium_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( medium_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( medium_bias_shape_ );
        Tensor<float, HostMemoryResource> output( medium_output_shape_ );

        for ( size_t i = 0; i < input.size(); ++i ) {
            float val;
            int pattern = i % 8;
            switch ( pattern ) {
                case 0: val = 1e6f; break;
                case 1: val = -1e6f; break;
                case 2: val = 1e-6f; break;
                case 3: val = -1e-6f; break;
                case 4: val = 0.0f; break;
                case 5: val = 1.0f; break;
                case 6: val = -1.0f; break;
                case 7: val = 100.0f; break;
                default: val = 0.0f; break;
            }
            input.data()[ i ] = val;
        }

        for ( size_t i = 0; i < weights->size(); ++i ) {
            weights->data()[ i ] = (i % 2 == 0) ? 0.01f : -0.01f;
        }

        for ( size_t i = 0; i < bias->size(); ++i ) {
            bias->data()[ i ] = (i % 2 == 0) ? 0.1f : -0.1f;
        }

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        EXPECT_FALSE( hasNaNorInf( output ) );
    }

    /**
     * @brief Test deterministic behavior
     */
    TEST_F( CpuLinearOpTests, DeterministicBehavior ) {
        auto op = createLinearOp( medium_in_features_, medium_out_features_ );

        Tensor<float, HostMemoryResource> input( medium_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( medium_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( medium_bias_shape_ );
        Tensor<float, HostMemoryResource> output1( medium_output_shape_ );
        Tensor<float, HostMemoryResource> output2( medium_output_shape_ );

        initializeTensors( input, *weights, bias.get() );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state1;
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state2;

        op->forward( input, parameters, output1, output_state1 );
        op->forward( input, parameters, output2, output_state2 );

        for ( size_t i = 0; i < output1.size(); ++i ) {
            EXPECT_EQ( output1.data()[ i ], output2.data()[ i ] );
        }
    }

    /**
     * @brief Test device context validation
     */
    TEST_F( CpuLinearOpTests, DeviceContextValidation ) {
        try {
            LinearConfig cuda_config( small_in_features_, small_out_features_ );
            EXPECT_THROW( CpuLinearOp( std::make_shared<DeviceContext>( "CUDA:0" ), cuda_config ), std::runtime_error );
        }
        catch ( const std::exception& e ) {
            GTEST_SKIP() << "CUDA device not available, skipping device mismatch test: " << e.what();
        }
    }

    /**
     * @brief Test constructor variants
     */
    TEST_F( CpuLinearOpTests, Constructors ) {
        LinearConfig config( small_in_features_, small_out_features_ );

        ASSERT_NO_THROW( (CpuLinearOp( config )) );
        ASSERT_NO_THROW( CpuLinearOp( cpu_context_, config ) );
    }

    /**
     * @brief Test configuration options
     */
    TEST_F( CpuLinearOpTests, ConfigurationOptions ) {
        LinearConfig config_with_bias( small_in_features_, small_out_features_ );
        EXPECT_TRUE( config_with_bias.hasBias() );
        EXPECT_EQ( config_with_bias.getInputFeatures(), small_in_features_ );
        EXPECT_EQ( config_with_bias.getOutputFeatures(), small_out_features_ );

        LinearConfig config_no_bias( small_in_features_, small_out_features_ );
        config_no_bias.withBias( false );
        EXPECT_FALSE( config_no_bias.hasBias() );

        ASSERT_NO_THROW( CpuLinearOp( cpu_context_, config_with_bias ) );
        ASSERT_NO_THROW( CpuLinearOp( cpu_context_, config_no_bias ) );
    }

    /**
     * @brief Test single input feature
     */
    TEST_F( CpuLinearOpTests, SingleInputFeature ) {
        auto op = createLinearOp( 1, small_out_features_ );

        std::vector<size_t> input_shape = { small_batch_, small_seq_len_, 1 };
        std::vector<size_t> weight_shape = { small_out_features_, 1 };

        Tensor<float, HostMemoryResource> input( input_shape );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( weight_shape );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( small_bias_shape_ );
        Tensor<float, HostMemoryResource> output( small_output_shape_ );

        initializeTensors( input, *weights, bias.get() );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );
    }

    /**
     * @brief Test single output feature
     */
    TEST_F( CpuLinearOpTests, SingleOutputFeature ) {
        auto op = createLinearOp( small_in_features_, 1 );

        std::vector<size_t> output_shape = { small_batch_, small_seq_len_, 1 };
        std::vector<size_t> weight_shape = { 1, small_in_features_ };
        std::vector<size_t> bias_shape = { 1 };

        Tensor<float, HostMemoryResource> input( small_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( weight_shape );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( bias_shape );
        Tensor<float, HostMemoryResource> output( output_shape );

        initializeTensors( input, *weights, bias.get() );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );
    }

    /**
     * @brief Test performance with large tensors
     */
    TEST_F( CpuLinearOpTests, Performance ) {
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping performance test in CI environment";
        }

        auto op = createLinearOp( large_in_features_, large_out_features_ );

        Tensor<float, HostMemoryResource> input( large_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( large_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( large_bias_shape_ );
        Tensor<float, HostMemoryResource> output( large_output_shape_ );

        initializeTensors( input, *weights, bias.get() );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        const int iterations = 10;
        auto start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            op->forward( input, parameters, output, output_state );
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

        size_t flops_per_iter = large_batch_ * large_seq_len_ * large_out_features_ * (2 * large_in_features_);
        size_t total_flops = flops_per_iter * iterations;

        double flops_per_second = static_cast<double>( total_flops ) / (duration.count() * 1e-6);
        double gflops = flops_per_second / 1e9;
        double avg_time_per_iter = duration.count() / iterations;

        RecordProperty( "Performance_GFLOPS", std::to_string( gflops ) );
        RecordProperty( "Average_Time_us", std::to_string( avg_time_per_iter ) );

        std::cout << "Performance: " << gflops << " GFLOPS, "
            << "Average Time: " << avg_time_per_iter << " us" << std::endl;

        EXPECT_TRUE( true );
    }

    /**
     * @brief Test OpenMP scaling
     */
    TEST_F( CpuLinearOpTests, OpenMPScaling ) {
    #ifdef USE_OMP
        if ( std::getenv( "CI" ) != nullptr ) {
            GTEST_SKIP() << "Skipping OpenMP scaling test in CI environment";
        }

        auto op = createLinearOp( large_in_features_, large_out_features_ );

        Tensor<float, HostMemoryResource> input( large_input_shape_ );
        auto weights = std::make_shared<Tensor<float, HostMemoryResource>>( large_weight_shape_ );
        auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( large_bias_shape_ );
        Tensor<float, HostMemoryResource> output( large_output_shape_ );

        initializeTensors( input, *weights, bias.get() );

        std::vector<std::shared_ptr<ITensorData>> parameters = { weights, bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

        int max_threads = omp_get_max_threads();
        std::cout << "Max OpenMP threads: " << max_threads << std::endl;

        std::vector<int> thread_counts = { 1 };
        if ( max_threads > 1 ) {
            thread_counts.push_back( max_threads );
            if ( max_threads > 3 ) {
                thread_counts.push_back( max_threads / 2 );
            }
        }

        const int iterations = 10;
        size_t flops_per_iter = large_batch_ * large_seq_len_ * large_out_features_ * (2 * large_in_features_);

        for ( int num_threads : thread_counts ) {
            omp_set_num_threads( num_threads );

            auto start_time = std::chrono::high_resolution_clock::now();

            for ( int i = 0; i < iterations; ++i ) {
                op->forward( input, parameters, output, output_state );
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

            double flops_per_second = static_cast<double>( flops_per_iter * iterations ) /
                (duration.count() * 1e-6);

            std::cout << "CPU Linear with " << num_threads << " threads: "
                << flops_per_second / 1e9 << " GFLOPS" << std::endl;
        }
    #else
        // The skipped test is reported as 
        std::cout << "OpenMP not available, skipping scaling test";
		SUCCEED();
    #endif
    }
}