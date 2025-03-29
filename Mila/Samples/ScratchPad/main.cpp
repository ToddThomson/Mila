#include <iostream>
#include <sstream>
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

 //   size_t batch_size = 2;
 //   size_t sequence_length = 2;
 //   size_t channels = 4;
	//size_t num_heads = 4;

 //   std::vector<size_t> input_shape = std::vector<size_t>{ batch_size, sequence_length, channels };
 //   std::vector<size_t> output_shape = std::vector<size_t>{ batch_size, sequence_length, channels };

 //   auto transformer_block = TransformerBlock<float>( "tf", input_shape, num_heads);
	//std::cout << transformer_block << std::endl;

 //   Tensor<float> X( input_shape );
	//X.setName( "tf.X" );

 //   random<float>( X, 0.0f, 5.0f );

 //   Tensor<float> Y( output_shape );
	//Y.setName( "tf.Y" );

	//std::cout << ">>> TransformerBlock input: " << std::endl;
	//std::cout << X;

 //   //transformer_block.forward( X, Y );

 //   std::cout << ">>> TransformerBlock output: " << std::endl;
 //   std::cout << Y;

	return 0;
}