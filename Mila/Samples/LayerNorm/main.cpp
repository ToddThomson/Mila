#include <iostream>
#include <vector>

import Mila;
import App.Model.LayerNorm;

int main() {

    using namespace Mila::Dnn;
    using namespace App::Model::LayerNorm;

	Mila::Initialize();

    std::cout << "Mila version: " << Mila::GetAPIVersion().ToString() << std::endl;

    // Hyperparameters: batch, time / sequence length, number of channels
    size_t B = 2;
    size_t T = 3;
    size_t C = 4;

	auto input_shape = std::vector<size_t>{ B, T, C };

    // TJT: B,T,C should come from input tensor shape?

    auto model = LayerNormModel<float>( "LayerNorm Model", B, T, C);
	
    auto layernorm = std::make_shared<Modules::LayerNorm<float,Compute::CpuMemoryResource>>("ln1", input_shape);

	int index = model.add( layernorm );
    
    xavier<float,Compute::CpuMemoryResource>( *layernorm->getWeight(), C, C );

    model.print();
 
    // Input tensor 
    Tensor<float,Compute::CpuMemoryResource> X( input_shape );
    random<float,Compute::CpuMemoryResource>( X, -1.0f, 1.0f );
	
    X.print();

    /*Tensor<float> grad_x = Tensor<float>( { B * T * C } );
    Tensor<float> grad_w = Tensor<float>( { C } );
    Tensor<float> grad_b = Tensor<float>( { C } );
    Tensor<float> grad_y = Tensor<float>( { B * T * C } );*/

    // now let's calculate everything ourselves

    // forward pass
    auto Y = model.forward( X );
	Y.print();

    //model.backward( c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C );

    return 0;
}
