#include <iostream>
#include <memory>
#include <vector>

//import Mila;

//namespace Scratchpad::LayerNorm
//{
//    using namespace Mila::Dnn;
//
//    void SampleLayerNorm() {
//        std::cout << "Mila version: " << Mila::GetAPIVersion().ToString() << std::endl;
//
//        auto devices = Compute::list_devices();
//        std::cout << "Available compute devices: ";
//        for ( const auto& device : devices ) {
//            std::cout << device << " ";
//        }
//        std::cout << std::endl;
//
//        std::cout << "The current Compute Device is: " << Mila::getDevice()->getDeviceName() << std::endl;
//
//        std::unique_ptr<LayerNorm<float, float, Compute::HostMemoryResource> cpu_layernorm{ nullptr };
//        //std::unique_ptr<Modules::LayerNorm<float, Compute::CudaDeviceMemoryResource>> cuda_layernorm{ nullptr };
//
//        size_t cuda_batch_size = 4;
//        size_t cpu_batch_size = 4;
//        size_t sequence_length = 1024;
//        size_t channels = 768;
//
//        std::vector<size_t> cpu_input_shape = { cpu_batch_size, sequence_length, channels };
//        std::vector<size_t> cuda_input_shape = { cuda_batch_size, sequence_length, channels };
//
//        cpu_layernorm = std::make_unique<LayerNorm<float, float,Compute::HostMemoryResource>>(
//            "Cpu_ln_1", cpu_input_shape );
//
//        //cuda_layernorm = std::make_unique<Modules::LayerNorm<float, Compute::CudaDeviceMemoryResource>>(
//        //    "Cuda_ln_1", cuda_input_shape );
//
//        Tensor<float, Compute::HostMemoryResource> input( cpu_input_shape );
//        random<float, Compute::HostMemoryResource>( input, 0.0f, 5.0f );
//
//        auto cuda_input = input.to<Compute::CudaDeviceMemoryResource>();
//
//        auto output = cpu_layernorm->forward( input );
//        //auto output2 = cuda_layernorm->forward( std::make_shared<DeviceTensor<float>>( cuda_input ) );
//
//        std::cout << "Cpu output: " << std::endl;
//        output.print();
//
//        std::cout << "Cuda output: " << std::endl;
//        //auto from_cuda_output2 = output2->to<Compute::HostMemoryResource>();
//        //from_cuda_output2.print();
//    }
//}