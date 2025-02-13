#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <format>


import Mila;

int main() {

    using namespace Mila::Dnn;

    Mila::Initialize();

    std::cout << "Mila version: " << Mila::GetAPIVersion().ToString() << std::endl;

    auto devices = Compute::list_devices();
    std::cout << "Available compute devices: ";
    for ( const auto& device : devices ) {
        std::cout << device << " ";
    }
    std::cout << std::endl;

    std::cout << "The current Compute Device is: " << Mila::getDevice()->getName() << std::endl;

    size_t cuda_batch_size = 4;
    size_t cpu_batch_size = 2;
    size_t sequence_length = 4;
    size_t channels = 3;

    std::vector<size_t> cpu_input_shape = std::vector<size_t>{ cpu_batch_size, sequence_length, channels };
    //std::vector<size_t> cuda_input_shape = { cuda_batch_size, sequence_length, channels };

    auto cpu_softmax = Modules::Softmax<float>(
        "Cpu_softmax", cpu_input_shape );

    Tensor<float, Compute::CpuMemoryResource> input( cpu_input_shape );
    random<float, Compute::CpuMemoryResource>( input, 0.0f, 5.0f );

	input.print();

    //auto cuda_input = input.to<Compute::DeviceMemoryResource>();

    auto output = cpu_softmax.forward( input );
    
    //auto output2 = cuda_layernorm->forward( std::make_shared<DeviceTensor<float>>( cuda_input ) );

    std::cout << "Cpu output: " << std::endl;
    output.print();

	auto B = output.shape()[ 0 ];
	auto T = output.shape()[ 1 ];
	auto V = output.shape()[ 2 ];
	// Check if all values in the output sum to a value close to 1
	float sum = 0.0f;
    for ( size_t i = 0; i < B; ++i ) {
		for ( size_t j = 0; j < T; ++j ) {
			float sum = 0.0f;
			for ( size_t v = 0; v < V; ++v ) {
				sum += output[ i, j, v ];
			}
			std::cout << std::format( "Sum({},{}): ", i, j ) << sum << std::endl;
		}
    }

    //std::cout << "Cuda output: " << std::endl;
    //auto from_cuda_output2 = output2->to<Compute::CpuMemoryResource>();
    //from_cuda_output2.print();

	return 0;
}