#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <memory>

import Mila;

int main() {

    using namespace Mila::Dnn;

	Mila::initialize();

    std::cout << "Mila version: " << Mila::getAPIVersion().ToString() << std::endl;

    auto devices = Compute::list_devices();
    std::cout << "Available compute devices: ";
    for ( const auto& device : devices ) {
        std::cout << device << " ";
    }
    std::cout << std::endl;

    //std::cout << "The current Compute Device is: " << Mila::getDevice()->getDeviceName() << std::endl;

 //   auto layernorm_model = Model<float>();

 //   // Hyperparameters: batch, time / sequence length, number of channels
 //   size_t B = 2;
 //   size_t T = 3;
 //   size_t C = 4;

 //   // Create the required layernorm tensors
	//auto input = Tensor<float>( std::vector<size_t>{ B, T, C } );
 //   random<float>( input, -1.0f, 1.0f );
 //   std::cout << input.toString();


	//auto ln_normalized_shape = std::vector<size_t>{ C };

 //   // auto weights = Tensor<float>( std::vector<size_t>{ C }, 1.0f );
 //   //auto bias = Tensor<float>( std::vector<size_t>{ C });
 //  
	//// Register the tensors with the model
	////auto X = layernorm_model.tensor( "X", input );
	////auto W = layernorm_model.tensor( "W", weights );
	////auto b = layernorm_model.tensor( "B", bias );

 //   //auto Y =  layernorm_model.layernorm( "ln1", X, ln_normalized_shape );

 //   //model.print();

 //   // now let's calculate everything ourselves
	////layernorm_model.build();

 //   // forward pass
 //   //layernorm_model.forward();
	////Y->print();

 //   //model.backward( c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C );

    return 0;
}
