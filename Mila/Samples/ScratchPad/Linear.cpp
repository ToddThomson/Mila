#include <iostream>
#include <memory>
#include <vector>

//import Mila;
//
//namespace Scratchpad::Linear
//{
//    using namespace Mila::Dnn;
//
//    void SampleLinear() {
//        size_t cuda_batch_size = 4;
//        size_t cpu_batch_size = 4;
//        size_t sequence_length = 1024;
//        size_t channels = 768;
//
//        shape_t cpu_input_shape = { cpu_batch_size, sequence_length, channels };
//        shape_t cuda_input_shape = { cuda_batch_size, sequence_length, channels };
//
//		auto A = std::shared_ptr<Tensor<float, Compute::HostMemoryResource>>( cpu_input_shape );
//
//        //size_t output_channels = 3 * channels;
//
//        //auto cpu_linear = Linear<float>( "cpu_linear_1", cpu_input_shape, output_channels );
//
//        //auto cuda_linear = Linear<float, float, Compute::CudaDeviceMemoryResource>(
//        //    "cuda_linear_2", cuda_input_shape, output_channels );
//
//        //Tensor<float, Compute::HostMemoryResource> input( cpu_input_shape );
//        //random<float, Compute::HostMemoryResource>( input, 0.0f, 5.0f );
//
//        //auto cuda_input = input.to<Compute::CudaDeviceMemoryResource>();
//
//        ////auto output = cpu_linear.forward( HostTensor<float>( input ) );
//        ////auto output2 = cuda_linear.forward( DeviceTensor<float>( cuda_input ) );
//
//        //std::cout << "Cpu output: " << std::endl;
//        ////output.print();
//
//        //std::cout << "Cuda output: " << std::endl;
//        ////auto from_cuda_output2 = output2.to<Compute::HostMemoryResource>();
//        ////from_cuda_output2.print();
//    }
//}