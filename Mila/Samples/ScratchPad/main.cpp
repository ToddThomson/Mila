#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <iostream>

import Mila;

int main() {

    using namespace Mila::Dnn;

    std::cout << "Mila version: " << Mila::GetAPIVersion().ToString() << std::endl;

    // Construct a session
    auto dnn = Session();

    Tensor<float> tensor( { 2,3 } );
    tensor.fill( 5.0f );
    tensor[ 1,2 ] = 3.0f;

    std::vector<size_t> shape = { 2, 3 };
    Tensor<float> t2( shape );

    auto view = t2.as_vector( 20 );
	std::cout << "Tensor mapped to a 4 element vector" << std::endl;
    std::cout << "Size: " << view.size() << std::endl;
    std::cout << "Rank: " << view.rank() << std::endl;
    std::cout << "Rows: " << view.extent( 0 ) << std::endl;

	auto view2 = t2.as_matrix( { 3,2 } );
	std::cout << "Tensor mapped to 3x2 matrix" << std::endl;
    std::cout << "Size: " << view2.size() << std::endl;
    std::cout << "Rank: " << view2.rank() << std::endl;
    std::cout << "Rows: " << view2.extent( 0 ) << std::endl;
    std::cout << "Cols: " << view2.extent( 1 ) << std::endl;

    auto view3 = t2.as_matrix( { 2,3 } );
    std::cout << "Tensor mapped to 2x3 matrix" << std::endl;
    std::cout << "Size: " << view3.size() << std::endl;
    std::cout << "Rank: " << view3.rank() << std::endl;
    std::cout << "Rows: " << view3.extent( 0 ) << std::endl;
    std::cout << "Cols: " << view3.extent( 1 ) << std::endl;

	float value = tensor[1,1];
	std::cout << "Value at [1,1]: " << value << std::endl;
    
    tensor.print();

    return 0;
}