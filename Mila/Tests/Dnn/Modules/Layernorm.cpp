#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda_fp16.h>  // For half type

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<typename TPrecision, Compute::DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == Compute::DeviceType::Cuda,
        Compute::CudaMemoryResource,
        Compute::HostMemoryResource>;

    template<typename TPrecision, Compute::DeviceType TDevice>
    struct LayerNormTestData {
        std::vector<size_t> shape;
        std::shared_ptr<LayerNorm<TPrecision, TDevice>> ln_module;
        bool has_bias;
        int64_t axis;
        bool is_training;

        // Make the test data structure self-initializing
        static LayerNormTestData Create(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t channels,
            int64_t axis = -1,
            bool has_bias = true,
            bool is_training = false )
        {
            LayerNormTestData data;
            data.shape = { batch_size, sequence_length, channels };
            data.has_bias = has_bias;
            data.axis = axis;
            data.is_training = is_training;

            std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.ln_module = std::make_shared<LayerNorm<TPrecision, TDevice>>(
                name, device_str, data.shape, axis, has_bias, is_training );

            return data;
        }

        // Overload for creating with device context
        static LayerNormTestData CreateWithContext(
            const std::string& name,
            size_t batch_size,
            size_t sequence_length,
            size_t channels,
            std::shared_ptr<DeviceContext> context,
            int64_t axis = -1,
            bool has_bias = true,
            bool is_training = false )
        {
            LayerNormTestData data;
            data.shape = { batch_size, sequence_length, channels };
            data.has_bias = has_bias;
            data.axis = axis;
            data.is_training = is_training;

            data.ln_module = std::make_shared<LayerNorm<TPrecision, TDevice>>(
                name, context, data.shape, axis, has_bias, is_training );

            return data;
        }
    };

    class LayerNormTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Initialize test parameters only
            batch_size_ = 128;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;
            has_bias_ = true;
            axis_ = -1; // Normalize across the last dimension (channels)
            // Modules will be created on demand
        }

        // Factory methods to lazily create test data as needed
        LayerNormTestData<float, Compute::DeviceType::Cpu>& CpuFloatData() {
            if ( !cpu_float_data_.ln_module ) {
                cpu_float_data_ = LayerNormTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_ln_float", cpu_batch_size_, sequence_length_, channels_, axis_, has_bias_ );
            }
            return cpu_float_data_;
        }

        LayerNormTestData<float, Compute::DeviceType::Cpu>& CpuNoBiasFloatData() {
            if ( !cpu_no_bias_float_data_.ln_module ) {
                cpu_no_bias_float_data_ = LayerNormTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_ln_no_bias_float", cpu_batch_size_, sequence_length_, channels_, axis_, false );
            }
            return cpu_no_bias_float_data_;
        }

        LayerNormTestData<float, Compute::DeviceType::Cuda>& CudaFloatData() {
            if ( !cuda_float_data_.ln_module ) {
                cuda_float_data_ = LayerNormTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_ln_float", batch_size_, sequence_length_, channels_, axis_, has_bias_ );
            }
            return cuda_float_data_;
        }

        LayerNormTestData<float, Compute::DeviceType::Cuda>& CudaNoBiasFloatData() {
            if ( !cuda_no_bias_float_data_.ln_module ) {
                cuda_no_bias_float_data_ = LayerNormTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_ln_no_bias_float", batch_size_, sequence_length_, channels_, axis_, false );
            }
            return cuda_no_bias_float_data_;
        }

        LayerNormTestData<float, Compute::DeviceType::Cpu>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.ln_module ) {
                training_cpu_float_data_ = LayerNormTestData<float, Compute::DeviceType::Cpu>::Create(
                    "cpu_ln_float_training", cpu_batch_size_, sequence_length_, channels_, axis_, has_bias_, true );
            }
            return training_cpu_float_data_;
        }

        LayerNormTestData<float, Compute::DeviceType::Cuda>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.ln_module ) {
                training_cuda_float_data_ = LayerNormTestData<float, Compute::DeviceType::Cuda>::Create(
                    "cuda_ln_float_training", batch_size_, sequence_length_, channels_, axis_, has_bias_, true );
            }
            return training_cuda_float_data_;
        }

        LayerNormTestData<float, Compute::DeviceType::Cpu>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.ln_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = LayerNormTestData<float, Compute::DeviceType::Cpu>::CreateWithContext(
                    "cpu_context_ln_float", cpu_batch_size_, sequence_length_, channels_, cpu_context, axis_, has_bias_ );
            }
            return context_cpu_float_data_;
        }

        LayerNormTestData<half, Compute::DeviceType::Cuda>& CudaHalfData() {
            if ( !cuda_half_data_.ln_module ) {
                cuda_half_data_ = LayerNormTestData<half, Compute::DeviceType::Cuda>::Create(
                    "cuda_ln_half", batch_size_, sequence_length_, channels_, axis_, has_bias_ );
            }
            return cuda_half_data_;
        }

        LayerNormTestData<half, Compute::DeviceType::Cuda>& CudaNoBiasHalfData() {
            if ( !cuda_no_bias_half_data_.ln_module ) {
                cuda_no_bias_half_data_ = LayerNormTestData<half, Compute::DeviceType::Cuda>::Create(
                    "cuda_ln_no_bias_half", batch_size_, sequence_length_, channels_, axis_, false );
            }
            return cuda_no_bias_half_data_;
        }

        // Test parameters
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        bool has_bias_{ true };
        int64_t axis_{ -1 };

        // Test data objects - initialized on demand
        LayerNormTestData<float, Compute::DeviceType::Cpu> cpu_float_data_;
        LayerNormTestData<float, Compute::DeviceType::Cpu> cpu_no_bias_float_data_;
        LayerNormTestData<float, Compute::DeviceType::Cpu> context_cpu_float_data_;
        LayerNormTestData<float, Compute::DeviceType::Cpu> training_cpu_float_data_;

        LayerNormTestData<float, Compute::DeviceType::Cuda> cuda_float_data_;
        LayerNormTestData<float, Compute::DeviceType::Cuda> cuda_no_bias_float_data_;
        LayerNormTestData<float, Compute::DeviceType::Cuda> training_cuda_float_data_;

        LayerNormTestData<half, Compute::DeviceType::Cuda> cuda_half_data_;
        LayerNormTestData<half, Compute::DeviceType::Cuda> cuda_no_bias_half_data_;
    };

    // Common test function templates
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestGetName( const LayerNormTestData<TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.ln_module->getName(), expected_name );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestParameterCount( const LayerNormTestData<TPrecision, TDevice>& data ) {
        // Determine expected parameter count based on normalization dimension and bias
        size_t channels = data.shape[ 2 ]; // Last dimension is channels
        size_t expected_count = channels; // Weight parameters

        if ( data.has_bias ) {
            expected_count += channels; // Bias parameters
        }

        EXPECT_EQ( data.ln_module->parameterCount(), expected_count );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestForward( const LayerNormTestData<TPrecision, TDevice>& data ) {
        using MR = MemoryResourceType<TPrecision, TDevice>;

        Tensor<TPrecision, MR> input( data.shape );
        Tensor<TPrecision, MR> output( data.shape );

        if constexpr ( TDevice == Compute::DeviceType::Cuda ) {
            Tensor<TPrecision, Compute::HostMemoryResource> host_input( data.shape );

            // Fill host input with predictable values
            for ( size_t i = 0; i < host_input.size(); ++i ) {
                host_input.data()[ i ] = static_cast<TPrecision>( (i % 10) * 0.1f );
            }

            input.copyFrom( host_input );
        }
        else {
            for ( size_t i = 0; i < input.size(); ++i ) {
                input.data()[ i ] = static_cast<TPrecision>( (i % 10) * 0.1f );
            }
        }

        data.ln_module->forward( input, output );

        EXPECT_EQ( output.size(), input.size() );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestSimpleNormalization() {
        using MR = MemoryResourceType<TPrecision, TDevice>;

        // Create a small, predictable input
        std::vector<size_t> shape = { 1, 2, 3 };
        Tensor<TPrecision, MR> input( shape );
        Tensor<TPrecision, MR> output( shape );

        // Initialize with known values
        input.data()[ 0 ] = 1.0f;
        input.data()[ 1 ] = 2.0f;
        input.data()[ 2 ] = 3.0f;
        input.data()[ 3 ] = 4.0f;
        input.data()[ 4 ] = 5.0f;
        input.data()[ 5 ] = 6.0f;

        // Create LayerNorm module with all weights=1 and all biases=0
        std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";
        auto ln = std::make_shared<LayerNorm<TPrecision, TDevice>>(
            "test_simple_ln", device_str, shape, -1, true );

        // Get weight and bias tensors and set them to known values
        auto weight = ln->getWeight();
        auto bias = ln->getBias();

        for ( size_t i = 0; i < weight->size(); ++i ) {
            weight->data()[ i ] = 1.0f;
        }

        for ( size_t i = 0; i < bias->size(); ++i ) {
            bias->data()[ i ] = 0.0f;
        }

        // Perform forward pass
        ln->forward( input, output );

        // Expected output for a normalized sequence with mean=0 and var=1
        const float tolerance = 1e-5f;

        // Row 1: [1,2,3] normalized should be approximately [-1.22474, 0, 1.22474]
        EXPECT_NEAR( output.data()[ 0 ], -1.22474f, tolerance );
        EXPECT_NEAR( output.data()[ 1 ], 0.0f, tolerance );
        EXPECT_NEAR( output.data()[ 2 ], 1.22474f, tolerance );

        // Row 2: [4,5,6] normalized should be approximately [-1.22474, 0, 1.22474]
        EXPECT_NEAR( output.data()[ 3 ], -1.22474f, tolerance );
        EXPECT_NEAR( output.data()[ 4 ], 0.0f, tolerance );
        EXPECT_NEAR( output.data()[ 5 ], 1.22474f, tolerance );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const LayerNormTestData<TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.ln_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestTrainingMode( const LayerNormTestData<TPrecision, TDevice>& data, bool expected_mode ) {
        EXPECT_EQ( data.ln_module->isTraining(), expected_mode );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestDeviceType( const LayerNormTestData<TPrecision, TDevice>& data ) {
        auto device_context = data.ln_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TPrecision>
    void TestCpuCudaEquivalence(
        const LayerNormTestData<TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const LayerNormTestData<TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_shape = { 2, 4, 8 }; // Small shape for quick verification

        // Create test modules with smaller dimensions
        std::string cpu_device = "CPU";
        std::string cuda_device = "CUDA:0";

        auto cpu_test_module = std::make_shared<LayerNorm<TPrecision, Compute::DeviceType::Cpu>>(
            "cpu_test", cpu_device, test_shape, -1, cpu_data.has_bias );

        auto cuda_test_module = std::make_shared<LayerNorm<TPrecision, Compute::DeviceType::Cuda>>(
            "cuda_test", cuda_device, test_shape, -1, cuda_data.has_bias );

        // Create random input data
        Tensor<TPrecision, Compute::HostMemoryResource> host_input( test_shape );

        // Fill with predictable values between -1 and 1
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TPrecision>( -1.0 + 2.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        // Copy parameters from CPU module to CUDA module for fair comparison
        auto cpu_weight = cpu_test_module->getWeight();
        auto cuda_weight = cuda_test_module->getWeight();

        Tensor<TPrecision, Compute::CudaMemoryResource> device_weight( cpu_weight->shape() );
        device_weight.copyFrom( *cpu_weight );
        cuda_weight->copyFrom( device_weight );

        if ( cpu_data.has_bias && cuda_data.has_bias ) {
            auto cpu_bias = cpu_test_module->getBias();
            auto cuda_bias = cuda_test_module->getBias();

            Tensor<TPrecision, Compute::CudaMemoryResource> device_bias( cpu_bias->shape() );
            device_bias.copyFrom( *cpu_bias );
            cuda_bias->copyFrom( device_bias );
        }

        // Create CPU output
        Tensor<TPrecision, Compute::HostMemoryResource> cpu_output( test_shape );
        cpu_test_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TPrecision, Compute::CudaMemoryResource> device_input( test_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TPrecision, Compute::CudaMemoryResource> cuda_output( test_shape );
        cuda_test_module->forward( device_input, cuda_output );

        // Copy CUDA output back to host for comparison
        Tensor<TPrecision, Compute::HostMemoryResource> cuda_output_host( test_shape );
        cuda_output_host.copyFrom( cuda_output );

        // Compare outputs with tolerance for floating point differences
        const float epsilon = 1e-4f;
        bool all_equal = true;

        for ( size_t i = 0; i < cpu_output.size(); ++i ) {
            float diff = std::abs( static_cast<float>( cpu_output.data()[ i ] ) - static_cast<float>( cuda_output_host.data()[ i ] ) );
            if ( diff > epsilon ) {
                std::cout << "Difference at index " << i << ": CPU=" << cpu_output.data()[ i ]
                    << ", CUDA=" << cuda_output_host.data()[ i ] << ", diff=" << diff << std::endl;
                    all_equal = false;
                    // Only show first few differences to avoid flooding the output
                    if ( i > 10 ) break;
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }

    // Test edge cases with minimal and large shapes
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestEdgeCases() {
        using MR = MemoryResourceType<TPrecision, TDevice>;
        std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";

        try {
            // Test with minimal shape
            std::vector<size_t> minimal_shape = { 1, 1, 4 };
            auto minimal_module = std::make_shared<LayerNorm<TPrecision, TDevice>>(
                "minimal", device_str, minimal_shape );

            Tensor<TPrecision, MR> minimal_input( minimal_shape );
            Tensor<TPrecision, MR> minimal_output( minimal_shape );

            // Fill with values
            for ( size_t i = 0; i < minimal_input.size(); ++i ) {
                minimal_input.data()[ i ] = static_cast<TPrecision>( i + 1 );
            }

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );

            // Test with single value per channel (edge case for normalization)
            std::vector<size_t> single_shape = { 1, 1, 1 };
            auto single_module = std::make_shared<LayerNorm<TPrecision, TDevice>>(
                "single", device_str, single_shape );

            Tensor<TPrecision, MR> single_input( single_shape );
            Tensor<TPrecision, MR> single_output( single_shape );

            single_input.data()[ 0 ] = 42.0f;  // Any value should normalize to 0 since stddev is 0

            EXPECT_NO_THROW( single_module->forward( single_input, single_output ) );

            // Test with larger dimensions
            std::vector<size_t> large_shape = { 2, 2, 1024 };
            auto large_module = std::make_shared<LayerNorm<TPrecision, TDevice>>(
                "large", device_str, large_shape );

            Tensor<TPrecision, MR> large_input( large_shape );
            Tensor<TPrecision, MR> large_output( large_shape );

            EXPECT_NO_THROW( large_module->forward( large_input, large_output ) );
            EXPECT_EQ( large_output.size(), 4096 );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestTrainingModeBehavior(
        const LayerNormTestData<TPrecision, TDevice>& training_data,
        const LayerNormTestData<TPrecision, TDevice>& inference_data ) {

        using MR = MemoryResourceType<TPrecision, TDevice>;

        // Verify training status
        EXPECT_TRUE( training_data.ln_module->isTraining() );
        EXPECT_FALSE( inference_data.ln_module->isTraining() );

        // Test mode switching
        training_data.ln_module->setTraining( false );
        EXPECT_FALSE( training_data.ln_module->isTraining() );

        inference_data.ln_module->setTraining( true );
        EXPECT_TRUE( inference_data.ln_module->isTraining() );

        // Test that both can process data regardless of training mode
        std::vector<size_t> test_shape = { 2, 3, 4 };

        // Create tensor with initial values - using the pattern seen in TestNumericalStability
        Tensor<TPrecision, MR> input( test_shape, 0.1f );  // Initialize all elements to 0.1f
        Tensor<TPrecision, MR> train_output( test_shape );
        Tensor<TPrecision, MR> infer_output( test_shape );

        // Both should be able to process data without errors
        EXPECT_NO_THROW( training_data.ln_module->forward( input, train_output ) );
        EXPECT_NO_THROW( inference_data.ln_module->forward( input, infer_output ) );

        // Reset for other tests
        training_data.ln_module->setTraining( true );
        inference_data.ln_module->setTraining( false );
    }

    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestNumericalStability() {
        using MR = MemoryResourceType<TPrecision, TDevice>;
        std::string device_str = TDevice == Compute::DeviceType::Cuda ? "CUDA:0" : "CPU";

        std::vector<size_t> shape = { 2, 4, 8 };
        auto stability_module = std::make_shared<LayerNorm<TPrecision, TDevice>>(
            "stability", device_str, shape );

        // Create tensors with initial values
        Tensor<TPrecision, MR> large_input( shape, 1.0e+4f );  // Initialize all elements with 1.0e+4f
        Tensor<TPrecision, MR> small_input( shape, 1.0e-6f );  // Initialize all elements with 1.0e-6f
        Tensor<TPrecision, MR> large_output( shape );
        Tensor<TPrecision, MR> small_output( shape );

        // Run forward passes
        stability_module->forward( large_input, large_output );
        stability_module->forward( small_input, small_output );

        if constexpr ( TDevice == Compute::DeviceType::Cuda ) {
            // For CUDA, copy results back to host for verification
            Tensor<TPrecision, Compute::HostMemoryResource> large_host_output( shape );
            Tensor<TPrecision, Compute::HostMemoryResource> small_host_output( shape );
            large_host_output.copyFrom( large_output );
            small_host_output.copyFrom( small_output );

            // Verify no NaNs or Infs in large output
            bool has_nan_or_inf = false;
            for ( size_t i = 0; i < large_host_output.size(); ++i ) {
                if ( std::isnan( large_host_output.data()[ i ] ) || std::isinf( large_host_output.data()[ i ] ) ) {
                    has_nan_or_inf = true;
                    break;
                }
            }

            EXPECT_FALSE( has_nan_or_inf ) << "Output contains NaN or Inf values with large inputs";

            // Verify no NaNs or Infs in small output
            has_nan_or_inf = false;
            for ( size_t i = 0; i < small_host_output.size(); ++i ) {
                if ( std::isnan( small_host_output.data()[ i ] ) || std::isinf( small_host_output.data()[ i ] ) ) {
                    has_nan_or_inf = true;
                    break;
                }
            }

            EXPECT_FALSE( has_nan_or_inf ) << "Output contains NaN or Inf values with small inputs";
        }
        else {
            // For CPU, verify directly
            bool has_nan_or_inf = false;
            for ( size_t i = 0; i < large_output.size(); ++i ) {
                if ( std::isnan( large_output.data()[ i ] ) || std::isinf( large_output.data()[ i ] ) ) {
                    has_nan_or_inf = true;
                    break;
                }
            }

            EXPECT_FALSE( has_nan_or_inf ) << "Output contains NaN or Inf values with large inputs";

            // Verify small output
            has_nan_or_inf = false;
            for ( size_t i = 0; i < small_output.size(); ++i ) {
                if ( std::isnan( small_output.data()[ i ] ) || std::isinf( small_output.data()[ i ] ) ) {
                    has_nan_or_inf = true;
                    break;
                }
            }

            EXPECT_FALSE( has_nan_or_inf ) << "Output contains NaN or Inf values with small inputs";
        }
    }


    // Test deterministic behavior (for CUDA implementation)
    template<typename TPrecision>
    void TestDeterministicBehavior() {
        std::vector<size_t> shape = { 2, 4, 8 };
        auto deterministic_module = std::make_shared<LayerNorm<TPrecision, Compute::DeviceType::Cuda>>(
            "deterministic", "CUDA:0", shape );

        Tensor<TPrecision, Compute::HostMemoryResource> host_input( shape );
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TPrecision>( i * 0.01f );
        }

        Tensor<TPrecision, Compute::CudaMemoryResource> device_input( shape );
        device_input.copyFrom( host_input );

        Tensor<TPrecision, Compute::CudaMemoryResource> output1( shape );
        Tensor<TPrecision, Compute::CudaMemoryResource> output2( shape );

        // Run forward pass twice
        deterministic_module->forward( device_input, output1 );
        deterministic_module->forward( device_input, output2 );

        // Copy results back to host for comparison
        Tensor<TPrecision, Compute::HostMemoryResource> host_output1( shape );
        Tensor<TPrecision, Compute::HostMemoryResource> host_output2( shape );
        host_output1.copyFrom( output1 );
        host_output2.copyFrom( output2 );

        // Verify outputs are identical
        bool outputs_match = true;
        for ( size_t i = 0; i < host_output1.size(); ++i ) {
            if ( host_output1.data()[ i ] != host_output2.data()[ i ] ) {
                outputs_match = false;
                std::cout << "Mismatch at index " << i << ": "
                    << host_output1.data()[ i ] << " vs "
                    << host_output2.data()[ i ] << std::endl;
                break;
            }
        }

        EXPECT_TRUE( outputs_match ) << "Multiple runs with the same input produced different results";
    }

    // Mock test for save/load functionality
    template<typename TPrecision, Compute::DeviceType TDevice>
    void TestSaveLoadMockup( const LayerNormTestData<TPrecision, TDevice>& data ) {
        // This is a mock test as we don't have actual save/load implemented
        auto weight = data.ln_module->getWeight();
        EXPECT_NE( weight, nullptr );
        EXPECT_EQ( weight->size(), data.shape[ 2 ] );

        if ( data.has_bias ) {
            auto bias = data.ln_module->getBias();
            EXPECT_NE( bias, nullptr );
            EXPECT_EQ( bias->size(), data.shape[ 2 ] );
        }
    }

    // CPU Tests with float precision
    TEST_F( LayerNormTests, Cpu_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cpu>( CpuFloatData(), "cpu_ln_float" );
    }

    TEST_F( LayerNormTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( LayerNormTests, Cpu_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( LayerNormTests, Cpu_Float_TestSimpleNormalization ) {
        TestSimpleNormalization<float, Compute::DeviceType::Cpu>();
    }

    TEST_F( LayerNormTests, Cpu_Float_TestPrint ) {
        TestPrint<float, Compute::DeviceType::Cpu>(
            CpuFloatData(), "LayerNorm: cpu_ln_float" );
    }

    TEST_F( LayerNormTests, Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( LayerNormTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( CpuFloatData(), false );
    }

    // CPU Tests with no bias
    TEST_F( LayerNormTests, Cpu_Float_NoBias_TestName ) {
        TestGetName<float, Compute::DeviceType::Cpu>( CpuNoBiasFloatData(), "cpu_ln_no_bias_float" );
    }

    TEST_F( LayerNormTests, Cpu_Float_NoBias_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cpu>( CpuNoBiasFloatData() );
    }

    TEST_F( LayerNormTests, Cpu_Float_NoBias_TestForward ) {
        TestForward<float, Compute::DeviceType::Cpu>( CpuNoBiasFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( LayerNormTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cpu>( TrainingCpuFloatData(), true );
    }

    // Context Construction Tests
    TEST_F( LayerNormTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    TEST_F( LayerNormTests, Context_Cpu_Float_Forward ) {
        TestForward<float, Compute::DeviceType::Cpu>( ContextCpuFloatData() );
    }

    // CUDA Tests with float precision
    TEST_F( LayerNormTests, Cuda_Float_TestName ) {
        TestGetName<float, Compute::DeviceType::Cuda>( CudaFloatData(), "cuda_ln_float" );
    }

    TEST_F( LayerNormTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( LayerNormTests, Cuda_Float_TestForward ) {
        TestForward<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( LayerNormTests, Cuda_Float_TestPrint ) {
        TestPrint<float, Compute::DeviceType::Cuda>(
            CudaFloatData(), "LayerNorm: cuda_ln_float" );
    }

    TEST_F( LayerNormTests, Cuda_Float_DeviceType ) {
        TestDeviceType<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    TEST_F( LayerNormTests, Cuda_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( CudaFloatData(), false );
    }

    // CUDA Tests with no bias
    TEST_F( LayerNormTests, Cuda_Float_NoBias_TestName ) {
        TestGetName<float, Compute::DeviceType::Cuda>( CudaNoBiasFloatData(), "cuda_ln_no_bias_float" );
    }

    TEST_F( LayerNormTests, Cuda_Float_NoBias_ParameterCount ) {
        TestParameterCount<float, Compute::DeviceType::Cuda>( CudaNoBiasFloatData() );
    }

    TEST_F( LayerNormTests, Cuda_Float_NoBias_TestForward ) {
        TestForward<float, Compute::DeviceType::Cuda>( CudaNoBiasFloatData() );
    }

    // CUDA Tests with half precision
    TEST_F( LayerNormTests, Cuda_Half_TestName ) {
        TestGetName<half, Compute::DeviceType::Cuda>( CudaHalfData(), "cuda_ln_half" );
    }

    TEST_F( LayerNormTests, Cuda_Half_ParameterCount ) {
        TestParameterCount<half, Compute::DeviceType::Cuda>( CudaHalfData() );
    }

    TEST_F( LayerNormTests, Cuda_Half_TestForward ) {
        TestForward<half, Compute::DeviceType::Cuda>( CudaHalfData() );
    }

    TEST_F( LayerNormTests, Cuda_Half_TestPrint ) {
        TestPrint<half, Compute::DeviceType::Cuda>( CudaHalfData(), "LayerNorm: cuda_ln_half" );
    }

    TEST_F( LayerNormTests, Cuda_Half_TrainingMode ) {
        TestTrainingMode<half, Compute::DeviceType::Cuda>( CudaHalfData(), false );
    }

    // CUDA Training Mode Tests
    TEST_F( LayerNormTests, Cuda_Training_Float_TrainingMode ) {
        TestTrainingMode<float, Compute::DeviceType::Cuda>( TrainingCudaFloatData(), true );
    }

    // Training behavior tests
    TEST_F( LayerNormTests, Cpu_Float_TrainingModeBehavior ) {
        TestTrainingModeBehavior<float, Compute::DeviceType::Cpu>(
            TrainingCpuFloatData(), CpuFloatData() );
    }

    TEST_F( LayerNormTests, Cuda_Float_TrainingModeBehavior ) {
        TestTrainingModeBehavior<float, Compute::DeviceType::Cuda>(
            TrainingCudaFloatData(), CudaFloatData() );
    }

    // Edge Case Tests
    TEST_F( LayerNormTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cpu>();
    }

    TEST_F( LayerNormTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<float, Compute::DeviceType::Cuda>();
    }

    // Numerical stability tests
    TEST_F( LayerNormTests, Cpu_Float_NumericalStability ) {
        TestNumericalStability<float, Compute::DeviceType::Cpu>();
    }

    TEST_F( LayerNormTests, Cuda_Float_NumericalStability ) {
        TestNumericalStability<float, Compute::DeviceType::Cuda>();
    }

    // Deterministic behavior test (CUDA only)
    TEST_F( LayerNormTests, Cuda_Float_Deterministic ) {
        TestDeterministicBehavior<float>();
    }

    // Mock save/load tests
    TEST_F( LayerNormTests, Cpu_Float_SaveLoad ) {
        TestSaveLoadMockup<float, Compute::DeviceType::Cpu>( CpuFloatData() );
    }

    TEST_F( LayerNormTests, Cuda_Float_SaveLoad ) {
        TestSaveLoadMockup<float, Compute::DeviceType::Cuda>( CudaFloatData() );
    }

    // CPU-CUDA equivalence test
    TEST_F( LayerNormTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
    }

    TEST_F( LayerNormTests, CpuCuda_NoBias_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float>( CpuNoBiasFloatData(), CudaNoBiasFloatData() );
    }
}
