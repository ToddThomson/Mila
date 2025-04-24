#include <iostream>
#include <sstream>
#include <memory>
#include <vector>
#include <string>
#include <format>
#include <random>

import Mila;

// Helper function to compare tensors with tolerance
template <typename T>
bool compareTensors( const Mila::Dnn::Tensor<T, Mila::Dnn::Compute::HostMemoryResource>& a,
    const Mila::Dnn::Tensor<T, Mila::Dnn::Compute::HostMemoryResource>& b,
    float epsilon = 1e-5f ) {
    if ( a.size() != b.size() ) {
        std::cout << "Size mismatch: " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }

    size_t mismatch_count = 0;
    for ( size_t i = 0; i < a.size(); ++i ) {
        float diff = std::abs( a.data()[ i ] - b.data()[ i ] );
        if ( diff > epsilon ) {
            if ( mismatch_count < 10 ) { // Only show first 10 mismatches
                std::cout << "Mismatch at index " << i << ": "
                    << a.data()[ i ] << " vs " << b.data()[ i ]
                    << " (diff = " << diff << ")" << std::endl;
            }
            mismatch_count++;
        }
    }

    if ( mismatch_count > 0 ) {
        std::cout << "Total mismatches: " << mismatch_count << " out of " << a.size() << std::endl;
        return false;
    }
    return true;
}

int main() {
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    Mila::initialize();

    std::cout << "Mila version: " << Mila::getAPIVersion().ToString() << std::endl;

    auto devices = Compute::list_devices();
    std::cout << "Available compute devices: ";
    for ( const auto& device : devices ) {
        std::cout << device << " ";
    }
    std::cout << std::endl;

    // Check if CUDA is available
    bool cuda_available = false;
    for ( const auto& device : devices ) {
        if ( device.find( "CUDA" ) != std::string::npos ) {
            cuda_available = true;
            break;
        }
    }

    if ( !cuda_available ) {
        std::cout << "No CUDA device available, skipping equivalency test." << std::endl;
        return 0;
    }

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Testing CudaFullyConnectedOp and CpuFullyConnectedOp equivalency..." << std::endl;

    try {
        // Create device contexts
        auto cuda_context = std::make_shared<DeviceContext>( "CUDA:0" );
        auto cpu_context = std::make_shared<DeviceContext>( "CPU" );

        // Create operations from registry
        auto cuda_op_base = OperationRegistry::instance().createOperation<float, float, DeviceType::Cuda>(
            "Cuda::FullyConnectedOp", cuda_context );
        auto cpu_op_base = OperationRegistry::instance().createOperation<float, float, DeviceType::Cpu>(
            "Cpu::FullyConnectedOp", cpu_context );

        if ( !cuda_op_base || !cpu_op_base ) {
            std::cerr << "Failed to create operations from registry." << std::endl;
            return 1;
        }

        // Cast to the correct operation types
        auto cuda_fc_op = std::dynamic_pointer_cast<CudaFullyConnectedOp<float>>(cuda_op_base);
        auto cpu_fc_op = std::dynamic_pointer_cast<UnaryOperation<float, float, DeviceType::Cpu>>(cpu_op_base);

        if ( !cuda_fc_op || !cpu_fc_op ) {
            std::cerr << "Failed to cast operations to the correct types." << std::endl;
            return 1;
        }

        // Define tensor dimensions
        size_t batch_size = 64;
        size_t seq_len = 128;
        size_t in_features = 256;
        size_t out_features = 512;

        std::vector<size_t> input_shape = { batch_size, seq_len, in_features };
        std::vector<size_t> output_shape = { batch_size, seq_len, out_features };
        std::vector<size_t> weight_shape = { out_features, in_features };
        std::vector<size_t> bias_shape = { out_features };

        // Create host tensors
        Tensor<float, HostMemoryResource> host_input( input_shape );
        Tensor<float, HostMemoryResource> host_weights( weight_shape );
        Tensor<float, HostMemoryResource> host_bias( bias_shape );
        Tensor<float, HostMemoryResource> cpu_output( output_shape );
        Tensor<float, HostMemoryResource> cuda_output_host( output_shape );

        // Initialize with fixed seed for reproducibility
        std::mt19937 rng( 42 );
        std::uniform_real_distribution<float> dist( -1.0f, 1.0f );

        // Initialize input with test values
        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = 1.0f; // dist( rng );
        }

        // Initialize weights
        for ( size_t i = 0; i < host_weights.size(); ++i ) {
            host_weights.data()[ i ] = 1.0f; // dist( rng ) * 0.1f; // Scale weights smaller
        }

        // Initialize bias
        for ( size_t i = 0; i < host_bias.size(); ++i ) {
            host_bias.data()[ i ] = 0.0f; // dist( rng ) * 0.1f;
        }

        // Create CUDA tensors
        Tensor<float, DeviceMemoryResource> cuda_input( input_shape );
        auto cuda_weights = std::make_shared<Tensor<float, DeviceMemoryResource>>( weight_shape );
        auto cuda_bias = std::make_shared<Tensor<float, DeviceMemoryResource>>( bias_shape );
        Tensor<float, DeviceMemoryResource> cuda_output( output_shape );

        // Copy data to CUDA
        cuda_input.copyFrom( host_input );
        cuda_weights->copyFrom( host_weights );
        cuda_bias->copyFrom( host_bias );

        // Create CPU tensor parameters
        auto cpu_weights = std::make_shared<Tensor<float, HostMemoryResource>>( host_weights );
        auto cpu_bias = std::make_shared<Tensor<float, HostMemoryResource>>( host_bias );

        // Operation parameters and cache
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> cuda_params = { cuda_weights, cuda_bias };
        std::vector<std::shared_ptr<Tensor<float, DeviceMemoryResource>>> cuda_output_cache;

        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_params = { cpu_weights, cpu_bias };
        std::vector<std::shared_ptr<Tensor<float, HostMemoryResource>>> cpu_output_cache;

        OperationAttributes props;

        // Execute operations
        std::cout << "Running CUDA FullyConnectedOp..." << std::endl;
        cuda_fc_op->forward( cuda_input, cuda_params, props, cuda_output, cuda_output_cache );

        std::cout << "Running CPU FullyConnectedOp..." << std::endl;
        cpu_fc_op->forward( host_input, cpu_params, props, cpu_output, cpu_output_cache );

        // Copy CUDA result back to host
        cuda_output_host.copyFrom( cuda_output );

        // Compare results
        std::cout << "Comparing results..." << std::endl;
        bool equivalent = compareTensors( cuda_output_host, cpu_output, 1e-4f );

        if ( equivalent ) {
            std::cout << "✓ PASS: CUDA and CPU implementations produce equivalent results!" << std::endl;
        }
        else {
            std::cout << "✗ FAIL: CUDA and CPU implementations produce different results." << std::endl;
        }

        // Print a sample of the outputs
        std::cout << "\nSample output values (first 5):" << std::endl;
        std::cout << "Index\tCUDA\t\tCPU\t\tDiff" << std::endl;
        std::cout << "-----\t----------\t----------\t----------" << std::endl;
        for ( size_t i = 0; i < std::min( size_t( 5 ), cuda_output_host.size() ); ++i ) {
            float cuda_val = cuda_output_host.data()[ i ];
            float cpu_val = cpu_output.data()[ i ];
            float diff = std::abs( cuda_val - cpu_val );
            std::cout << std::format( "{}\t{:.6f}\t{:.6f}\t{:.6f}", i, cuda_val, cpu_val, diff ) << std::endl;
        }


    }
    catch ( const std::exception& e ) {
        std::cerr << "Error during testing: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "---------------------------------------------------" << std::endl;

    return 0;
}