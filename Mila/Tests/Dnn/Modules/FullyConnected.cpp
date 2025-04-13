#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
    struct FullyConnectedTestData {
        std::vector<size_t> input_shape;
        std::vector<size_t> output_shape;
        std::shared_ptr<FullyConnected<TInput, TPrecision, TDevice>> fc_module;
        size_t channels;
        size_t output_channels;
        bool has_bias;
    };

    class FullyConnectedTests : public ::testing::Test {
    protected:
        void SetUp() override {
            try {
                std::cout << "Starting SetUp..." << std::endl;

                try {
                    batch_size_ = 64;
                    cpu_batch_size_ = 4;
                    sequence_length_ = 128;
                    channels_ = 256;
                    output_features_ = 4;
                    output_channels_ = output_features_ * channels_;
                    has_bias_ = true;

                    std::cout << "Initialized variables: output_channels_=" << output_channels_ << std::endl;
                }
                catch ( const std::exception& e ) {
                    std::cerr << "Exception during variable initialization: " << e.what() << std::endl;
                    throw;
                }

                try {
                    // CPU test data (float precision)
                    std::cout << "Creating cpu_float_data_..." << std::endl;
                    cpu_float_data_.input_shape = { cpu_batch_size_, sequence_length_, channels_ };
                    cpu_float_data_.output_shape = { cpu_batch_size_, sequence_length_, output_channels_ };
                    cpu_float_data_.channels = channels_;
                    cpu_float_data_.output_channels = output_channels_;
                    cpu_float_data_.has_bias = has_bias_;
                    std::cout << "Creating CPU module..." << std::endl;
                    cpu_float_data_.fc_module = std::make_shared<FullyConnected<float, float, Compute::DeviceType::Cpu>>(
                        "fc_cpu_float", channels_, output_channels_ );
                    std::cout << "CPU module created successfully" << std::endl;
                }
                catch ( const std::bad_alloc& e ) {
                    std::cerr << "Bad allocation during CPU module setup: " << e.what() << std::endl;
                    throw;
                }
                catch ( const std::exception& e ) {
                    std::cerr << "Exception during CPU module setup: " << e.what() << std::endl;
                    throw;
                }

                try {
                    // CPU test data without bias (float precision)
                    std::cout << "Creating cpu_no_bias_float_data_..." << std::endl;
                    cpu_no_bias_float_data_.input_shape = { cpu_batch_size_, sequence_length_, channels_ };
                    cpu_no_bias_float_data_.output_shape = { cpu_batch_size_, sequence_length_, output_channels_ };
                    cpu_no_bias_float_data_.channels = channels_;
                    cpu_no_bias_float_data_.output_channels = output_channels_;
                    cpu_no_bias_float_data_.has_bias = false;
                    std::cout << "Creating CPU no-bias module..." << std::endl;
                    cpu_no_bias_float_data_.fc_module = std::make_shared<FullyConnected<float, float, Compute::DeviceType::Cpu>>(
                        "fc_cpu_no_bias_float", channels_, output_channels_, false );
                    std::cout << "CPU no-bias module created successfully" << std::endl;
                }
                catch ( const std::bad_alloc& e ) {
                    std::cerr << "Bad allocation during CPU no-bias module setup: " << e.what() << std::endl;
                    throw;
                }
                catch ( const std::exception& e ) {
                    std::cerr << "Exception during CPU no-bias module setup: " << e.what() << std::endl;
                    throw;
                }

                try {
                    // CUDA test data (float precision)
                    std::cout << "Creating cuda_float_data_..." << std::endl;
                    cuda_float_data_.input_shape = { batch_size_, sequence_length_, channels_ };
                    cuda_float_data_.output_shape = { batch_size_, sequence_length_, output_channels_ };
                    cuda_float_data_.channels = channels_;
                    cuda_float_data_.output_channels = output_channels_;
                    cuda_float_data_.has_bias = has_bias_;
                    std::cout << "Creating CUDA module..." << std::endl;
                    cuda_float_data_.fc_module = std::make_shared<FullyConnected<float, float, Compute::DeviceType::Cuda>>(
                        "fc_cuda_float", channels_, output_channels_ );
                    std::cout << "CUDA module created successfully" << std::endl;
                }
                catch ( const std::bad_alloc& e ) {
                    std::cerr << "Bad allocation during CUDA module setup: " << e.what() << std::endl;
                    throw;
                }
                catch ( const std::exception& e ) {
                    std::cerr << "Exception during CUDA module setup: " << e.what() << std::endl;
                    throw;
                }

                try {
                    // CUDA test data without bias (float precision)
                    std::cout << "Creating cuda_no_bias_float_data_..." << std::endl;
                    cuda_no_bias_float_data_.input_shape = { batch_size_, sequence_length_, channels_ };
                    cuda_no_bias_float_data_.output_shape = { batch_size_, sequence_length_, output_channels_ };
                    cuda_no_bias_float_data_.channels = channels_;
                    cuda_no_bias_float_data_.output_channels = output_channels_;
                    cuda_no_bias_float_data_.has_bias = false;
                    std::cout << "Creating CUDA no-bias module..." << std::endl;
                    cuda_no_bias_float_data_.fc_module = std::make_shared<FullyConnected<float, float, Compute::DeviceType::Cuda>>(
                        "fc_cuda_no_bias_float", channels_, output_channels_, false );
                    std::cout << "CUDA no-bias module created successfully" << std::endl;
                }
                catch ( const std::bad_alloc& e ) {
                    std::cerr << "Bad allocation during CUDA no-bias module setup: " << e.what() << std::endl;
                    throw;
                }
                catch ( const std::exception& e ) {
                    std::cerr << "Exception during CUDA no-bias module setup: " << e.what() << std::endl;
                    throw;
                }

                try {
                    // Setup training mode modules
                    std::cout << "Creating training_cpu_float_data_..." << std::endl;
                    training_cpu_float_data_.input_shape = { cpu_batch_size_, sequence_length_, channels_ };
                    training_cpu_float_data_.output_shape = { cpu_batch_size_, sequence_length_, output_channels_ };
                    training_cpu_float_data_.channels = channels_;
                    training_cpu_float_data_.output_channels = output_channels_;
                    training_cpu_float_data_.has_bias = has_bias_;
                    std::cout << "Creating CPU training module..." << std::endl;
                    training_cpu_float_data_.fc_module = std::make_shared<FullyConnected<float, float, Compute::DeviceType::Cpu>>(
                        "fc_cpu_float_training", channels_, output_channels_, true, true );
                    std::cout << "CPU training module created successfully" << std::endl;
                }
                catch ( const std::bad_alloc& e ) {
                    std::cerr << "Bad allocation during CPU training module setup: " << e.what() << std::endl;
                    throw;
                }
                catch ( const std::exception& e ) {
                    std::cerr << "Exception during CPU training module setup: " << e.what() << std::endl;
                    throw;
                }

                try {
                    // Setup CUDA training mode modules
                    std::cout << "Creating training_cuda_float_data_..." << std::endl;
                    training_cuda_float_data_.input_shape = { batch_size_, sequence_length_, channels_ };
                    training_cuda_float_data_.output_shape = { batch_size_, sequence_length_, output_channels_ };
                    training_cuda_float_data_.channels = channels_;
                    training_cuda_float_data_.output_channels = output_channels_;
                    training_cuda_float_data_.has_bias = has_bias_;
                    std::cout << "Creating CUDA training module..." << std::endl;
                    training_cuda_float_data_.fc_module = std::make_shared<FullyConnected<float, float, Compute::DeviceType::Cuda>>(
                        "fc_cuda_float_training", channels_, output_channels_, true, true );
                    std::cout << "CUDA training module created successfully" << std::endl;
                }
                catch ( const std::bad_alloc& e ) {
                    std::cerr << "Bad allocation during CUDA training module setup: " << e.what() << std::endl;
                    throw;
                }
                catch ( const std::exception& e ) {
                    std::cerr << "Exception during CUDA training module setup: " << e.what() << std::endl;
                    throw;
                }

                std::cout << "SetUp completed successfully" << std::endl;
            }
            catch ( const std::bad_alloc& e ) {
                std::cerr << "*** Bad allocation in SetUp: " << e.what() << std::endl;

                // Calculate memory requirements for one of the tensors to help diagnose
                size_t weight_size = output_channels_ * channels_ * sizeof( float );
                std::cerr << "Weight tensor size would be: " << (weight_size / (1024 * 1024))
                    << " MB (" << weight_size << " bytes)" << std::endl;

                throw; // Re-throw to let GTest know the setup failed
            }
            catch ( const std::exception& e ) {
                std::cerr << "*** Exception in SetUp: " << e.what() << std::endl;
                throw; // Re-throw to let GTest know the setup failed
            }
        }


        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        size_t output_channels_{ 0 };
        size_t output_features_{ 0 };
        bool has_bias_{ true };

        // Structured test data
        FullyConnectedTestData<float, float, Compute::DeviceType::Cpu> cpu_float_data_;
        FullyConnectedTestData<float, float, Compute::DeviceType::Cuda> cuda_float_data_;
        FullyConnectedTestData<float, float, Compute::DeviceType::Cpu> cpu_no_bias_float_data_;
        FullyConnectedTestData<float, float, Compute::DeviceType::Cuda> cuda_no_bias_float_data_;
        FullyConnectedTestData<float, float, Compute::DeviceType::Cpu> training_cpu_float_data_;
        FullyConnectedTestData<float, float, Compute::DeviceType::Cuda> training_cuda_float_data_;
    };

    // Common test function templates
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestGetName( const FullyConnectedTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.fc_module->getName(), expected_name );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestParameterCount( const FullyConnectedTestData<TInput, TPrecision, TDevice>& data ) {
        auto num_parameters = (data.output_channels * data.channels); // weights
        if ( data.has_bias ) {
            num_parameters += data.output_channels; // bias
        }
        EXPECT_EQ( data.fc_module->parameterCount(), num_parameters );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestInitializeParameterTensors( const FullyConnectedTestData<TInput, TPrecision, TDevice>& data ) {
        auto parameters = data.fc_module->getParameterTensors();
        size_t expected_size = data.has_bias ? 2 : 1; // weights and bias or just weights
        EXPECT_EQ( parameters.size(), expected_size );

        // Validate weight tensor exists
        EXPECT_NE( parameters.find( "weight" ), parameters.end() );

        // Validate bias tensor exists if has_bias is true
        if ( data.has_bias ) {
            EXPECT_NE( parameters.find( "bias" ), parameters.end() );
        }
        else {
            EXPECT_EQ( parameters.find( "bias" ), parameters.end() );
        }
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestForward( const FullyConnectedTestData<TInput, TPrecision, TDevice>& data ) {
        Tensor<TInput, TMemResource> input( data.input_shape );
        Tensor<TPrecision, TMemResource> output( data.output_shape );
        data.fc_module->forward( input, output );
        EXPECT_EQ( output.size(), data.output_shape[ 0 ] * data.output_shape[ 1 ] * data.output_shape[ 2 ] );
    }

    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice>
    void TestPrint( const FullyConnectedTestData<TInput, TPrecision, TDevice>& data, const std::string& expected_substring ) {
        std::string output = data.fc_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    // Function to test equivalence of CPU and CUDA outputs
    template<typename TInput, typename TPrecision>
    void TestCpuCudaEquivalence(
        const FullyConnectedTestData<TInput, TPrecision, Compute::DeviceType::Cpu>& cpu_data,
        const FullyConnectedTestData<TInput, TPrecision, Compute::DeviceType::Cuda>& cuda_data ) {

        // Create a small test shape to make comparison faster
        std::vector<size_t> test_input_shape = { 2, 4, cpu_data.channels };
        std::vector<size_t> test_output_shape = { 2, 4, cpu_data.output_channels };

        // Create random input data
        Tensor<TInput, Compute::HostMemoryResource> host_input( test_input_shape );

        // Fill with predictable values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( -1.0 + 2.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        // Initialize the weights and biases with the same values for both CPU and CUDA modules
        // Copy parameters from CPU module to CUDA module for fair comparison
        auto cpu_params = cpu_data.fc_module->getParameterTensors();
        auto cuda_params = cuda_data.fc_module->getParameterTensors();

        // Copy weights
        Tensor<TPrecision, Compute::DeviceMemoryResource> cuda_weights( cpu_params[ "weight" ]->shape() );
        cuda_weights.copyFrom( *cpu_params[ "weight" ] );
        cuda_params[ "weight" ]->copyFrom( cuda_weights );

        // Copy bias if it exists
        if ( cpu_data.has_bias && cuda_data.has_bias ) {
            Tensor<TPrecision, Compute::DeviceMemoryResource> cuda_bias( cpu_params[ "bias" ]->shape() );
            cuda_bias.copyFrom( *cpu_params[ "bias" ] );
            cuda_params[ "bias" ]->copyFrom( cuda_bias );
        }

        // Create CPU output
        Tensor<TPrecision, Compute::HostMemoryResource> cpu_output( test_output_shape );
        cpu_data.fc_module->forward( host_input, cpu_output );

        // Create device input by copying host data
        Tensor<TInput, Compute::DeviceMemoryResource> device_input( test_input_shape );
        device_input.copyFrom( host_input );

        // Create device output
        Tensor<TPrecision, Compute::DeviceMemoryResource> cuda_output( test_output_shape );
        cuda_data.fc_module->forward( device_input, cuda_output );

        // Copy CUDA output back to host for comparison
        Tensor<TPrecision, Compute::HostMemoryResource> cuda_output_host( test_output_shape );
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
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }

    // Test with different dimensions (edge cases)
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestEdgeCases() {
        try {
            // Test with minimal sizes
            std::vector<size_t> minimal_input_shape = { 1, 1, 8 };
            std::vector<size_t> minimal_output_shape = { 1, 1, 16 };

            auto minimal_module = std::make_shared<FullyConnected<TInput, TPrecision, TDevice>>(
                "minimal_fc", 8, 16 );

            Tensor<TInput, TMemResource> minimal_input( minimal_input_shape );
			//std::cout << "Creating minimal output tensor..." << std::endl;
            Tensor<TPrecision, TMemResource> minimal_output( minimal_output_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 16 );
        }
		catch ( const std::exception& e ) {
			std::cerr << "Exception during minimal test: " << e.what() << std::endl;
			throw;
		}

        try {
			std::cout << "Testing with odd dimensions..." << std::endl;
            // Test with odd dimensions
			// TODO: Add fc variant ops for odd dimensions
			// For now the inside/last dimension must be a multiple of 4
            std::vector<size_t> odd_input_shape = { 3, 5, 8 };
            std::vector<size_t> odd_output_shape = { 3, 5, 12 };

            // Using dimensions that are multiples of 4 to avoid alignment issues
			// due to float4 alignment in CUDA kernels
            //
            // For CUDA tensors: dimensions should be multiples of 4 for best performance
            //std::vector<size_t> odd_input_shape = { 4, 8, 8 };  // Aligned dimensions instead of {3, 5, 7}
            //std::vector<size_t> odd_output_shape = { 4, 8, 12 }; // Aligned dimensions instead of {3, 5, 11}

            auto odd_module = std::make_shared<FullyConnected<TInput, TPrecision, TDevice>>(
                "odd_fc", 8, 12 );

            Tensor<TInput, TMemResource> odd_input( odd_input_shape );
            std::cout << "Creating minimal output tensor..." << std::endl;
            // Ensure scope for CUDA tensor to control its lifecycle
            {
                Tensor<TPrecision, TMemResource> odd_output( odd_output_shape );

                // Debug information
                if constexpr ( std::is_same_v<TMemResource, Compute::DeviceMemoryResource> ) {
                    std::cout << "Odd output pointer: " << std::hex << reinterpret_cast<uintptr_t>(odd_output.raw_data())
                        << std::dec << " aligned: " << (reinterpret_cast<uintptr_t>(odd_output.raw_data()) % 128 == 0) << std::endl;
                }

                EXPECT_NO_THROW( odd_module->forward( odd_input, odd_output ) );
                 EXPECT_EQ( odd_output.size(), 3 * 5 * 12 );
            }

            // Force synchronization after CUDA operations
            if constexpr ( std::is_same_v<TMemResource, Compute::DeviceMemoryResource> ) {
                cudaDeviceSynchronize();
            }

            std::cout << "Odd dimensions test passed." << std::endl;
        }
		catch ( const std::exception& e ) {
			std::cerr << "Exception during odd dimensions test: " << e.what() << std::endl;
			throw;
		}

        try {
            // Test with large dimensions for LLM with shape 64x1024x4096
            std::cout << "Starting large dimensions test (64x1024x4096)..." << std::endl;

            // Check if the device is Host, skip the test if true
            if constexpr ( TDevice == Compute::DeviceType::Cpu ) {
                std::cout << "Skipping large dimensions test on Host device." << std::endl;
                return;
            }

            // Using dimensions requested for language model testing
            std::vector<size_t> large_input_shape = { 64, 1024, 4096 };
            std::vector<size_t> large_output_shape = { 64, 1024, 4096 };

            // Memory calculation:
            // Input: 64 * 1024 * 4096 * 4 bytes = 1,073,741,824 bytes (~1024MB = 1GB)
            // Output: 64 * 1024 * 4096 * 4 bytes = 1,073,741,824 bytes (~1024MB = 1GB)
            // Weights: 4096 * 4096 * 4 bytes = 67,108,864 bytes (~64MB)
            // Bias: 4096 * 4 bytes = 16,384 bytes (~16KB)
            // Total: ~2.1GB of VRAM

            std::cout << "Memory estimate: ~2.1GB of GPU memory" << std::endl;

            auto large_module = std::make_shared<FullyConnected<TInput, TPrecision, TDevice>>(
                "large_fc", 4096, 4096 );

            std::cout << "Creating input tensor..." << std::endl;
            Tensor<TInput, TMemResource> large_input( large_input_shape );

            std::cout << "Creating output tensor..." << std::endl;
            Tensor<TPrecision, TMemResource> large_output( large_output_shape );

            std::cout << "Running forward pass with 64x1024x4096 tensors..." << std::endl;
            EXPECT_NO_THROW( large_module->forward( large_input, large_output ) );
            EXPECT_EQ( large_output.size(), 64 * 1024 * 4096 );

            std::cout << "Large dimensions test completed successfully" << std::endl;
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during large dimensions test: " << e.what() << std::endl;
            throw;
        }

    }

    // Test training mode behavior
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestTrainingModeBehavior(
        const FullyConnectedTestData<TInput, TPrecision, TDevice>& training_data,
        const FullyConnectedTestData<TInput, TPrecision, TDevice>& inference_data ) {

        // Verify training status
        EXPECT_TRUE( training_data.fc_module->isTraining() );
        EXPECT_FALSE( inference_data.fc_module->isTraining() );

        // Test mode switching
        training_data.fc_module->setTraining( false );
        EXPECT_FALSE( training_data.fc_module->isTraining() );

        inference_data.fc_module->setTraining( true );
        EXPECT_TRUE( inference_data.fc_module->isTraining() );

        // Reset for other tests
        training_data.fc_module->setTraining( true );
        inference_data.fc_module->setTraining( false );
    }

    // Test numerical stability with different input scales
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestNumericalStability( const FullyConnectedTestData<TInput, TPrecision, TDevice>& data ) {
        std::vector<size_t> test_input_shape = { 2, 4, data.channels };
        std::vector<size_t> test_output_shape = { 2, 4, data.output_channels };

        // Create input tensors
        Tensor<TInput, TMemResource> large_input( test_input_shape );
        Tensor<TInput, TMemResource> small_input( test_input_shape );

        // Create output tensors
        Tensor<TPrecision, TMemResource> large_output( test_output_shape );
        Tensor<TPrecision, TMemResource> small_output( test_output_shape );

        // Fill with large values
        if constexpr ( TMemResource::is_host_accessible ) {
            // Direct fill for host-accessible memory
            for ( size_t i = 0; i < large_input.size(); ++i ) {
                large_input.data()[ i ] = static_cast<TInput>( 1000.0f );
            }

            for ( size_t i = 0; i < small_input.size(); ++i ) {
                small_input.data()[ i ] = static_cast<TInput>( 0.001f );
            }
        }
        else {
            // For device memory, create on host and copy
            Tensor<TInput, Compute::HostMemoryResource> host_large_input( test_input_shape );
            Tensor<TInput, Compute::HostMemoryResource> host_small_input( test_input_shape );

            for ( size_t i = 0; i < host_large_input.size(); ++i ) {
                host_large_input.data()[ i ] = static_cast<TInput>( 1000.0f );
            }

            for ( size_t i = 0; i < host_small_input.size(); ++i ) {
                host_small_input.data()[ i ] = static_cast<TInput>( 0.001f );
            }

            large_input.copyFrom( host_large_input );
            small_input.copyFrom( host_small_input );
        }

        // Perform forward passes
        data.fc_module->forward( large_input, large_output );
        data.fc_module->forward( small_input, small_output );

        // Check for NaN or Inf - needs to be done on host-accessible memory
        bool has_nan_or_inf = false;

        if constexpr ( TMemResource::is_host_accessible ) {
            // For host memory, check directly
            for ( size_t i = 0; i < large_output.size(); ++i ) {
                if ( std::isnan( large_output.data()[ i ] ) || std::isinf( large_output.data()[ i ] ) ) {
                    has_nan_or_inf = true;
                    break;
                }
            }
        }
        else {
            // For device memory, copy to host first
            Tensor<TPrecision, Compute::HostMemoryResource> host_large_output( test_output_shape );
            host_large_output.copyFrom( large_output );

            for ( size_t i = 0; i < host_large_output.size(); ++i ) {
                if ( std::isnan( host_large_output.data()[ i ] ) || std::isinf( host_large_output.data()[ i ] ) ) {
                    has_nan_or_inf = true;
                    break;
                }
            }
        }

        EXPECT_FALSE( has_nan_or_inf ) << "Output contains NaN or Inf values with large inputs";

        has_nan_or_inf = false;

        if constexpr ( TMemResource::is_host_accessible ) {
            // For host memory, check directly
            for ( size_t i = 0; i < small_output.size(); ++i ) {
                if ( std::isnan( small_output.data()[ i ] ) || std::isinf( small_output.data()[ i ] ) ) {
                    has_nan_or_inf = true;
                    break;
                }
            }
        }
        else {
            // For device memory, copy to host first
            Tensor<TPrecision, Compute::HostMemoryResource> host_small_output( test_output_shape );
            host_small_output.copyFrom( small_output );

            for ( size_t i = 0; i < host_small_output.size(); ++i ) {
                if ( std::isnan( host_small_output.data()[ i ] ) || std::isinf( host_small_output.data()[ i ] ) ) {
                    has_nan_or_inf = true;
                    break;
                }
            }
        }

        EXPECT_FALSE( has_nan_or_inf ) << "Output contains NaN or Inf values with small inputs";
    }


    // Test deterministic behavior (for CUDA implementation)
    template<typename TInput, typename TPrecision>
    void TestDeterministicBehavior(
        const FullyConnectedTestData<TInput, TPrecision, Compute::DeviceType::Cuda>& data ) {

        std::vector<size_t> test_input_shape = { 2, 4, data.channels };
        std::vector<size_t> test_output_shape = { 2, 4, data.output_channels };

        // Create input tensor
        Tensor<TInput, Compute::HostMemoryResource> host_input( test_input_shape );

        // Fill with predictable values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TInput>( (i % 10) * 0.1f );
        }

        Tensor<TInput, Compute::DeviceMemoryResource> device_input( test_input_shape );
        device_input.copyFrom( host_input );

        // Create output tensors
        Tensor<TPrecision, Compute::DeviceMemoryResource> output1( test_output_shape );
        Tensor<TPrecision, Compute::DeviceMemoryResource> output2( test_output_shape );

        // Run forward pass twice
        data.fc_module->forward( device_input, output1 );
        data.fc_module->forward( device_input, output2 );

        // Copy outputs back to host for comparison
        Tensor<TPrecision, Compute::HostMemoryResource> host_output1( test_output_shape );
        Tensor<TPrecision, Compute::HostMemoryResource> host_output2( test_output_shape );

        host_output1.copyFrom( output1 );
        host_output2.copyFrom( output2 );

        // Verify outputs are identical
        bool outputs_match = true;

        for ( size_t i = 0; i < host_output1.size(); ++i ) {
            if ( host_output1.data()[ i ] != host_output2.data()[ i ] ) {
                outputs_match = false;
                break;
            }
        }

        EXPECT_TRUE( outputs_match ) << "Multiple runs with the same input produced different results";
    }

    // Mock test for save/load functionality
    template<typename TInput, typename TPrecision, Compute::DeviceType TDevice, typename TMemResource>
    void TestSaveLoad( const FullyConnectedTestData<TInput, TPrecision, TDevice>& data ) {
		// TODO: Implement save/load functionality
    }

    // CPU Tests with float precision
    TEST_F( FullyConnectedTests, Cpu_Float_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_, "fc_cpu_float" );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_InitializeParameterTensors ) {
        TestInitializeParameterTensors<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_TestPrint ) {
        TestPrint<float, float, Compute::DeviceType::Cpu>(
            cpu_float_data_, "FullyConnected: fc_cpu_float" );
    }

    // CUDA Tests with float precision
    TEST_F( FullyConnectedTests, Cuda_Float_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_, "fc_cuda_float" );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_InitializeParameterTensors ) {
        TestInitializeParameterTensors<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_TestPrint ) {
        TestPrint<float, float, Compute::DeviceType::Cuda>(
            cuda_float_data_, "FullyConnected: fc_cuda_float" );
    }

    // Test CPU and CUDA equivalence
    TEST_F( FullyConnectedTests, CpuCuda_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float, float>( cpu_float_data_, cuda_float_data_ );
    }

    // NEW TEST CASES

    // Tests for no-bias configuration
    TEST_F( FullyConnectedTests, Cpu_Float_NoBias_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_no_bias_float_data_, "fc_cpu_no_bias_float" );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_NoBias_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_no_bias_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_NoBias_InitializeParameterTensors ) {
        TestInitializeParameterTensors<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_no_bias_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cpu_Float_NoBias_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_no_bias_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_NoBias_TestName ) {
        TestGetName<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_no_bias_float_data_, "fc_cuda_no_bias_float" );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_NoBias_ParameterCount ) {
        TestParameterCount<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_no_bias_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_NoBias_InitializeParameterTensors ) {
        TestInitializeParameterTensors<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_no_bias_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_NoBias_TestForward ) {
        TestForward<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_no_bias_float_data_ );
    }

    TEST_F( FullyConnectedTests, CpuCuda_NoBias_Forward_Output_Equivalence ) {
        TestCpuCudaEquivalence<float, float>( cpu_no_bias_float_data_, cuda_no_bias_float_data_ );
    }

    // Edge case tests
    TEST_F( FullyConnectedTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>();
    }

    TEST_F( FullyConnectedTests, Cuda_Float_EdgeCases ) {
        TestEdgeCases<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>();
    }

    // Training mode tests
    TEST_F( FullyConnectedTests, Cpu_Float_TrainingModeBehavior ) {
        TestTrainingModeBehavior<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            training_cpu_float_data_, cpu_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_TrainingModeBehavior ) {
        TestTrainingModeBehavior<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            training_cuda_float_data_, cuda_float_data_ );
    }

    // Numerical stability tests
    TEST_F( FullyConnectedTests, Cpu_Float_NumericalStability ) {
        TestNumericalStability<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_NumericalStability ) {
        TestNumericalStability<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_ );
    }

    // Deterministic behavior test (CUDA only)
    TEST_F( FullyConnectedTests, Cuda_Float_Deterministic ) {
        TestDeterministicBehavior<float, float>( cuda_float_data_ );
    }

    // Mock save/load tests
    TEST_F( FullyConnectedTests, Cpu_Float_SaveLoad ) {
        TestSaveLoad<float, float, Compute::DeviceType::Cpu, Compute::HostMemoryResource>(
            cpu_float_data_ );
    }

    TEST_F( FullyConnectedTests, Cuda_Float_SaveLoad ) {
        TestSaveLoad<float, float, Compute::DeviceType::Cuda, Compute::DeviceMemoryResource>(
            cuda_float_data_ );
    }
}
