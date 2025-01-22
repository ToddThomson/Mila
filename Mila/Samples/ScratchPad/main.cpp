#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <iostream>

import Mila;

int main() {

    using namespace Mila::Dnn;

    std::cout << "Mila version: " << Mila::GetAPIVersion().ToString() << std::endl;

    auto devices = Compute::list_devices();
	std::cout << "Available Compute Devices: ";
	for ( const auto& device : devices ) {
		std::cout << device << " ";
	}
	std::cout << std::endl;

    Tensor<float> tensor( {  1000, 1000 }, "CUDA:0" );
    random( tensor, 0.0f, 5.0f );
    //tensor.fill( 5.0f );
    //tensor[ 1,2,1 ] = 3.0f;

    tensor.print();

    //std::vector<size_t> shape = { 2, 3 };
    Tensor<float> t2( { 2,3 } );

    auto view = t2.vectorSpan();
	std::cout << "Tensor mapped to a vector" << std::endl;
    std::cout << "Size: " << view.size() << std::endl;
    std::cout << "Rank: " << view.rank() << std::endl;
    std::cout << "Rows: " << view.extent( 0 ) << std::endl;

	auto view2 = t2.matrixSpan( { 3,2 } );
	std::cout << "Tensor mapped to 3x2 matrix" << std::endl;
    std::cout << "Size: " << view2.size() << std::endl;
    std::cout << "Rank: " << view2.rank() << std::endl;
    std::cout << "Rows: " << view2.extent( 0 ) << std::endl;
    std::cout << "Cols: " << view2.extent( 1 ) << std::endl;

    auto view3 = t2.matrixSpan( { 2,3 } );
    std::cout << "Tensor mapped to 2x3 matrix" << std::endl;
    std::cout << "Size: " << view3.size() << std::endl;
    std::cout << "Rank: " << view3.rank() << std::endl;
    std::cout << "Rows: " << view3.extent( 0 ) << std::endl;
    std::cout << "Cols: " << view3.extent( 1 ) << std::endl;

	/*float value = tensor[1,1];
	std::cout << "Value at [1,1]: " << value << std::endl;*/
    
    tensor.print();

    return 0;
}