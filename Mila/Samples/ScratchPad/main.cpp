#include <iostream>
#include <thrust/host_vector.h>

import Mila;

int main() {

    using namespace Mila::Dnn;

    //std::cout << "Mila version: " << Mila::GetAPIVersion().ToString() << std::endl;

    auto devices = Compute::list_devices();
	std::cout << "Available Compute Devices: ";
	for ( const auto& device : devices ) {
		std::cout << device << " ";
	}
	std::cout << std::endl;

    Tensor<float> tensor( {  1000, 1000 } );
    random( tensor, 0.0f, 5.0f );
    tensor.fill( 5.0f );
    tensor[ 1,2 ] = 3.0f;

    tensor.print();


    return 0;
}