﻿#include <iostream>
#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>

import Mila;

int main() {

    using namespace Mila::Dnn;

	Mila::Initialize();

    std::cout << "Mila version: " << Mila::GetAPIVersion().ToString() << std::endl;

    auto devices = Compute::list_devices();
	std::cout << "Available Compute Devices: ";
	for ( const auto& device : devices ) {
		std::cout << device << " ";
	}
	std::cout << std::endl;

    // Create a host tensor by default
    Tensor<float> tensor( { 1000, 1000 } );
    random( tensor, 0.0f, 5.0f );
    tensor.print();

    tensor.fill( 5.0f );
    //tensor[ 1,2 ] = 3.0f;

    tensor.print();


    return 0;
}