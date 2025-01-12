// poor man's tensor checker
#include <iostream>

import Mila;

import App.Model.LayerNorm;
import App.Ops.LayerNorm;

int main() {

    using namespace Mila::Dnn;
    using namespace App::Model::LayerNorm;
    using namespace App::Ops;

    // Hyperparameters: batch, time / sequence length, number of channels
    size_t B = 2;
    size_t T = 3;
    size_t C = 4;

    // TJT: B,T,C should come from input tensor shape?

    auto model = LayerNormModel<float>( "LayerNorm Model", B, T, C);
	//auto layer = LayerNormOp<float>( "ln1", B, T, C );

	//model.add( layer );
    model.add( LayerNormOp<float>( "ln1", B, T, C) );

    auto layer = std::dynamic_pointer_cast<LayerNormOp<float>>( model[ 0 ] );
    
    xavier( layer->Weight(), C, C );

    model.print();
 
    // Input tensor 
    Tensor<float> x = Tensor<float>( { B * T * C } );
    Tensor<float> y = Tensor<float>( { B * T * C } );

    random( x, -1.0f, 1.0f );

    /*Tensor<float> grad_x = Tensor<float>( { B * T * C } );
    Tensor<float> grad_w = Tensor<float>( { C } );
    Tensor<float> grad_b = Tensor<float>( { C } );
    Tensor<float> grad_y = Tensor<float>( { B * T * C } );*/

    // now let's calculate everything ourselves

    // forward pass
    auto y_result = model.forward( x );

    //model.backward( c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C );

    return 0;
}
