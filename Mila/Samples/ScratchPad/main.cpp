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

    size_t batch_size = 2;
    size_t sequence_length = 2;
    size_t channels = 4;
	size_t num_heads = 4;

    std::vector<size_t> input_shape = std::vector<size_t>{ batch_size, sequence_length, channels };
    std::vector<size_t> output_shape = std::vector<size_t>{ batch_size, sequence_length, channels };

    auto transformer_block = Blocks::TransformerBlock<float>( input_shape, num_heads );

    Tensor<float, Compute::CpuMemoryResource> X( input_shape );
	X.set_name( "X" );

    random<float, Compute::CpuMemoryResource>( X, 0.0f, 5.0f );

    Tensor<float, Compute::CpuMemoryResource> Y( output_shape );
	Y.set_name( "Y" );

	X.print();

    transformer_block.forward( X, Y );

    std::cout << "Cpu output: " << std::endl;
    Y.print();

	return 0;
}