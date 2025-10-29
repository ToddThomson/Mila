#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <format>

//import Mila;
//
//int Softmax() {
//
//    using namespace Mila::Dnn;
//
//    Mila::Initialize();
//
//    std::cout << "Mila version: " << Mila::GetAPIVersion().ToString() << std::endl;
//
//    auto devices = Compute::list_devices();
//    std::cout << "Available compute devices: ";
//    for ( const auto& device : devices ) {
//        std::cout << device << " ";
//    }
//    std::cout << std::endl;
//
//    std::cout << "The current Compute Device is: " << Mila::getDevice()->getDeviceName() << std::endl;
//
//    size_t cuda_batch_size = 4;
//    size_t cpu_batch_size = 2;
//    size_t sequence_length = 4;
//    size_t channels = 3;
//
//    shape_t cpu_input_shape = shape_t{ cpu_batch_size, sequence_length, channels };
//    //shape_t cuda_input_shape = { cuda_batch_size, sequence_length, channels };
//
//    auto cpu_softmax = Softmax<float>(
//        "Cpu_softmax", cpu_input_shape );
//
//    Tensor<float, Compute::HostMemoryResource> input( cpu_input_shape );
//    random<float, Compute::HostMemoryResource>( input, 0.0f, 5.0f );
//
//	input.print();
//
//    //auto cuda_input = input.to<Compute::CudaDeviceMemoryResource>();
//
//    auto output = cpu_softmax.forward( input );
//    
//    //auto output2 = cuda_layernorm->forward( std::make_shared<DeviceTensor<float>>( cuda_input ) );
//
//    std::cout << "Cpu output: " << std::endl;
//    output.print();
//
//	auto B = output.shape()[ 0 ];
//	auto T = output.shape()[ 1 ];
//	auto V = output.shape()[ 2 ];
//	// Check if all values in the output sum to a value close to 1
//	float sum = 0.0f;
//    for ( size_t i = 0; i < B; ++i ) {
//		for ( size_t j = 0; j < T; ++j ) {
//			float sum = 0.0f;
//			for ( size_t v = 0; v < V; ++v ) {
//				sum += output[ i, j, v ];
//			}
//			std::cout << std::format( "Sum({},{}): ", i, j ) << sum << std::endl;
//		}
//    }
//
//    //std::cout << "Cuda output: " << std::endl;
//    //auto from_cuda_output2 = output2->to<Compute::HostMemoryResource>();
//    //from_cuda_output2.print();
//
//	return 0;
//}