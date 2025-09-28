/**
 * @file CpuLayerNormTests.cpp
 * @brief Test suite for the CPU LayerNorm operation.
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

namespace Compute::Cpu::Operations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Test fixture for CpuLayerNormOp tests
     */
    class CpuLayerNormOpTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cpu_context_ = std::make_shared<DeviceContext>( "CPU" );

            // Small shapes for quick tests
            small_batch_ = 2;
            small_seq_len_ = 3;
            small_feature_dim_ = 4;
            small_shape_ = { small_batch_, small_seq_len_, small_feature_dim_ };

            // Medium shapes for more thorough tests
            medium_batch_ = 8;
            medium_seq_len_ = 16;
            medium_feature_dim_ = 32;
            medium_shape_ = { medium_batch_, medium_seq_len_, medium_feature_dim_ };

            // Large shape for stress tests
            large_batch_ = 32;
            large_seq_len_ = 64;
            large_feature_dim_ = 128;
            large_shape_ = { large_batch_, large_seq_len_, large_feature_dim_ };
        }

        std::shared_ptr<CpuLayerNormOp> createLayerNormOp( size_t feature_dim, float epsilon = 1e-5f ) {
            LayerNormConfig config( feature_dim );
            config.withEpsilon( epsilon );
            
            return std::make_shared<CpuLayerNormOp>( cpu_context_, config );
        }

        std::vector<std::shared_ptr<ITensorData>> createParameters( size_t feature_dim ) {
            std::vector<std::shared_ptr<ITensorData>> params;

            auto weight = std::make_shared<Tensor<float, HostMemoryResource>>( std::vector<size_t>{feature_dim} );
            for ( size_t i = 0; i < feature_dim; ++i ) {
                weight->data()[ i ] = 1.0f;
            }
            params.push_back( weight );

            auto bias = std::make_shared<Tensor<float, HostMemoryResource>>( std::vector<size_t>{feature_dim} );
            for ( size_t i = 0; i < feature_dim; ++i ) {
                bias->data()[ i ] = 0.0f;
            }
            params.push_back( bias );

            return params;
        }

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> createOutputState( size_t batch_size, size_t sequence_length ) {
            std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> output_state;

            auto mean = std::make_shared<Tensor<float, HostMemoryResource>>( std::vector<size_t>{batch_size, sequence_length} );
            output_state.push_back( mean );

            auto rstd = std::make_shared<Tensor<float, HostMemoryResource>>( std::vector<size_t>{batch_size, sequence_length} );
            output_state.push_back( rstd );

            return output_state;
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

        void layerNormReference(
            const Tensor<float, HostMemoryResource>& input,
            const std::vector<std::shared_ptr<ITensorData>>& params,
            float epsilon,
            Tensor<float, HostMemoryResource>& output,
            std::vector<float>& means,
            std::vector<float>& rstds ) {

            int B = input.shape()[ 0 ];
            int T = input.shape()[ 1 ];
            int C = input.shape()[ 2 ];

            const float* weight = static_cast<const float*>(params[ 0 ]->rawData());
            const float* bias = static_cast<const float*>(params[ 1 ]->rawData());

            means.resize( B * T );
            rstds.resize( B * T );

            for ( int b = 0; b < B; b++ ) {
                for ( int t = 0; t < T; t++ ) {
                    float mean = 0.0f;
                    int input_offset = b * T * C + t * C;
                    for ( int i = 0; i < C; i++ ) {
                        mean += input.data()[ input_offset + i ];
                    }
                    mean /= C;

                    float variance = 0.0f;
                    for ( int i = 0; i < C; i++ ) {
                        float diff = input.data()[ input_offset + i ] - mean;
                        variance += diff * diff;
                    }
                    variance /= C;

                    float rstd = 1.0f / sqrtf( variance + epsilon );

                    means[ b * T + t ] = mean;
                    rstds[ b * T + t ] = rstd;

                    for ( int i = 0; i < C; i++ ) {
                        float normalized = (input.data()[ input_offset + i ] - mean) * rstd;
                        output.data()[ input_offset + i ] = normalized * weight[ i ] + bias[ i ];
                    }
                }
            }
        }

        std::shared_ptr<DeviceContext> cpu_context_;

        size_t small_batch_, small_seq_len_, small_feature_dim_;
        size_t medium_batch_, medium_seq_len_, medium_feature_dim_;
        size_t large_batch_, large_seq_len_, large_feature_dim_;

        std::vector<size_t> small_shape_;
        std::vector<size_t> medium_shape_;
        std::vector<size_t> large_shape_;
    };

    /**
     * @brief Test CpuLayerNormOp name property
     */
    TEST_F( CpuLayerNormOpTests, Name ) {
        auto op = createLayerNormOp( small_feature_dim_ );
        EXPECT_EQ( op->getName(), "Cpu::LayerNormOp" );
    }

    /**
     * @brief Test basic functionality of CpuLayerNormOp
     */
    TEST_F( CpuLayerNormOpTests, BasicFunctionality ) {
        auto op = createLayerNormOp( small_feature_dim_ );

        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        auto parameters = createParameters( small_feature_dim_ );
        auto output_state = createOutputState( small_batch_, small_seq_len_ );

        Tensor<float, HostMemoryResource> expected_output( small_shape_ );
        std::vector<float> expected_means;
        std::vector<float> expected_rstds;

        float epsilon = 1e-5f;
        layerNormReference( input, parameters, epsilon, expected_output, expected_means, expected_rstds );

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
        }

        float* mean_data = output_state[ 0 ]->data();
        float* rstd_data = output_state[ 1 ]->data();

        for ( size_t i = 0; i < small_batch_ * small_seq_len_; ++i ) {
            EXPECT_NEAR( mean_data[ i ], expected_means[ i ], 1e-5f );
            EXPECT_NEAR( rstd_data[ i ], expected_rstds[ i ], 1e-5f );
        }
    }

    /**
     * @brief Test different epsilon values for numerical stability
     */
    TEST_F( CpuLayerNormOpTests, EpsilonValues ) {
        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        auto parameters = createParameters( small_feature_dim_ );
        auto output_state = createOutputState( small_batch_, small_seq_len_ );

        std::vector<float> epsilon_values = { 1e-3f, 1e-5f, 1e-7f, 1e-9f };

        for ( float epsilon : epsilon_values ) {
            auto op = createLayerNormOp( small_feature_dim_, epsilon );

            ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

            Tensor<float, HostMemoryResource> expected_output( small_shape_ );
            std::vector<float> expected_means;
            std::vector<float> expected_rstds;
            
            layerNormReference( input, parameters, epsilon, expected_output, expected_means, expected_rstds );

            auto out = output.toString( true );
			auto ref_out = expected_output.toString( true );

            for ( size_t i = 0; i < output.size(); ++i ) {
                EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
            }

            EXPECT_FALSE( hasNaNorInf( output ) );
        }
    }

    /**
     * @brief Test edge cases with zero, very small, and very large values
     */
    TEST_F( CpuLayerNormOpTests, EdgeCases ) {
        auto op = createLayerNormOp( small_feature_dim_ );

        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        auto parameters = createParameters( small_feature_dim_ );
        auto output_state = createOutputState( small_batch_, small_seq_len_ );

        // Test 1: All zeros
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 0.0f;
        }

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], 0.0f, 1e-5f );
        }

        // Test 2: All identical values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 1.0f;
        }

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], 0.0f, 1e-5f );
        }

        // Test 3: Very large values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 1e6f + static_cast<float>( i );
        }

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );
        EXPECT_FALSE( hasNaNorInf( output ) );

        // Test 4: Very small values
        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = 1e-6f * static_cast<float>( i + 1 );
        }

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );
        EXPECT_FALSE( hasNaNorInf( output ) );
    }

    /**
     * @brief Test with custom weights and biases
     */
    TEST_F( CpuLayerNormOpTests, CustomWeightsBiases ) {
        auto op = createLayerNormOp( small_feature_dim_ );

        Tensor<float, HostMemoryResource> input( small_shape_ );
        Tensor<float, HostMemoryResource> output( small_shape_ );

        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i ) - 10.0f) / 10.0f;
        }

        auto custom_params = createParameters( small_feature_dim_ );
        for ( size_t i = 0; i < small_feature_dim_; ++i ) {
            static_cast<Tensor<float, HostMemoryResource>*>( custom_params[ 0 ].get() )->data()[ i ] = 2.0f;
            static_cast<Tensor<float, HostMemoryResource>*>( custom_params[ 1 ].get() )->data()[ i ] = 1.0f;
        }

        auto output_state = createOutputState( small_batch_, small_seq_len_ );

        ASSERT_NO_THROW( op->forward( input, custom_params, output, output_state ) );

        Tensor<float, HostMemoryResource> expected_output( small_shape_ );
        std::vector<float> expected_means;
        std::vector<float> expected_rstds;
        layerNormReference( input, custom_params, 1e-5f, expected_output, expected_means, expected_rstds );

        for ( size_t i = 0; i < output.size(); ++i ) {
            EXPECT_NEAR( output.data()[ i ], expected_output.data()[ i ], 1e-5f );
        }
    }

    /**
     * @brief Test numerical stability with varied inputs
     */
    TEST_F( CpuLayerNormOpTests, NumericalStability ) {
        auto op = createLayerNormOp( medium_feature_dim_ );

        Tensor<float, HostMemoryResource> input( medium_shape_ );
        Tensor<float, HostMemoryResource> output( medium_shape_ );

        auto parameters = createParameters( medium_feature_dim_ );
        auto output_state = createOutputState( medium_batch_, medium_seq_len_ );

        for ( size_t i = 0; i < input.size(); ++i ) {
            float val;
            int pattern = i % 8;
            switch ( pattern ) {
                case 0: val = 1.0f; break;
                case 1: val = -1.0f; break;
                case 2: val = 0.0001f; break;
                case 3: val = -0.0001f; break;
                case 4: val = 10.0f; break;
                case 5: val = -10.0f; break;
                case 6: val = 100.0f; break;
                case 7: val = -100.0f; break;
                default: val = 0.0f; break;
            }
            input.data()[ i ] = val;
        }

        ASSERT_NO_THROW( op->forward( input, parameters, output, output_state ) );

        EXPECT_FALSE( hasNaNorInf( output ) );
    }

    /**
     * @brief Test deterministic behavior
     */
    TEST_F( CpuLayerNormOpTests, DeterministicBehavior ) {
        auto op = createLayerNormOp( medium_feature_dim_ );

        Tensor<float, HostMemoryResource> input( medium_shape_ );
        Tensor<float, HostMemoryResource> output1( medium_shape_ );
        Tensor<float, HostMemoryResource> output2( medium_shape_ );

        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( i % 17 ) - 8.5f) * 0.1f;
        }

        auto parameters = createParameters( medium_feature_dim_ );
        auto output_state1 = createOutputState( medium_batch_, medium_seq_len_ );
        auto output_state2 = createOutputState( medium_batch_, medium_seq_len_ );

        op->forward( input, parameters, output1, output_state1 );
        op->forward( input, parameters, output2, output_state2 );

        for ( size_t i = 0; i < output1.size(); ++i ) {
            EXPECT_EQ( output1.data()[ i ], output2.data()[ i ] );
        }
    }

    /**
     * @brief Test device context validation
     */
    TEST_F( CpuLayerNormOpTests, DeviceContextValidation ) {
        try {
            LayerNormConfig config( small_feature_dim_ );
            EXPECT_THROW( CpuLayerNormOp( std::make_shared<DeviceContext>( "CUDA:0" ), config ), std::runtime_error );
        }
        catch ( const std::exception& e ) {
            std::cout << "Skipping device context validation test: " << e.what() << std::endl;
            SUCCEED();
            return;
        }
    }

    /**
     * @brief Test constructor variants
     */
    TEST_F( CpuLayerNormOpTests, Constructors ) {
        LayerNormConfig config( small_feature_dim_ );

        ASSERT_NO_THROW( ( CpuLayerNormOp( config ) ) );
        ASSERT_NO_THROW( CpuLayerNormOp( cpu_context_, config ) );
    }

    /**
     * @brief Test performance with large tensors
     */
    TEST_F( CpuLayerNormOpTests, Performance ) {
        if ( std::getenv( "CI" ) != nullptr ) {
            std::cout << "Skipping performance test in CI environment" << std::endl;
            SUCCEED();
            return;
        }

        auto op = createLayerNormOp( large_feature_dim_ );

        Tensor<float, HostMemoryResource> input( large_shape_ );
        Tensor<float, HostMemoryResource> output( large_shape_ );

        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
        }

        auto parameters = createParameters( large_feature_dim_ );
        auto output_state = createOutputState( large_batch_, large_seq_len_ );

        const int iterations = 10;
        auto start_time = std::chrono::high_resolution_clock::now();

        for ( int i = 0; i < iterations; ++i ) {
            op->forward( input, parameters, output, output_state );
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

        size_t total_elements = input.size() * iterations;
        double elements_per_second = static_cast<double>( total_elements ) / (duration.count() * 1e-6);
        std::cout << "CPU LayerNorm Performance: " << elements_per_second / 1e6 << " million elements/sec" << std::endl;
        std::cout << "Average time per iteration: " << duration.count() / iterations << " microseconds" << std::endl;

        EXPECT_TRUE( true );
    }

    /**
     * @brief Test OpenMP scaling
     */
    TEST_F( CpuLayerNormOpTests, OpenMPScaling ) {
    #ifndef USE_OMP
        std::cout << "OpenMP not available, skipping scaling test" << std::endl;
        SUCCEED();
        return;
    #else
        if ( std::getenv( "CI" ) != nullptr ) {
            std::cout << "Skipping OpenMP scaling test in CI environment" << std::endl;
            SUCCEED();
            return;
        }

        auto op = createLayerNormOp( large_feature_dim_ );

        Tensor<float, HostMemoryResource> input( large_shape_ );
        Tensor<float, HostMemoryResource> output( large_shape_ );

        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = (static_cast<float>( rand() ) / RAND_MAX * 2.0f - 1.0f) * 10.0f;
        }

        auto parameters = createParameters( large_feature_dim_ );
        auto output_state = createOutputState( large_batch_, large_seq_len_ );

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

        for ( int num_threads : thread_counts ) {
            omp_set_num_threads( num_threads );

            auto start_time = std::chrono::high_resolution_clock::now();

            for ( int i = 0; i < iterations; ++i ) {
                op->forward( input, parameters, output, output_state );
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

            double elements_per_second = static_cast<double>( input.size() * iterations ) /
                (duration.count() * 1e-6);

            std::cout << "CPU LayerNorm with " << num_threads << " threads: "
                << elements_per_second / 1e6 << " million elements/sec" << std::endl;
        }
    #endif
    }
}