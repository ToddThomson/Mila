#include <iostream>
#include <memory>
#include <vector>

import Mila;

using namespace Mila::Dnn;

namespace Scratchpad::LayerNorm
{
    void SampleLayerNorm() {
        std::cout << "Mila version: " << Mila::GetAPIVersion().ToString() << std::endl;

        auto devices = Compute::list_devices();
        std::cout << "Available compute devices: ";
        for ( const auto& device : devices ) {
            std::cout << device << " ";
        }
        std::cout << std::endl;

        std::cout << "The current Compute Device is: " << Mila::getDevice()->getName() << std::endl;

        std::unique_ptr<Modules::LayerNorm<float, Compute::CpuMemoryResource>> cpu_layernorm{ nullptr };
        //std::unique_ptr<Modules::LayerNorm<float, Compute::DeviceMemoryResource>> cuda_layernorm{ nullptr };

        size_t cuda_batch_size = 4;
        size_t cpu_batch_size = 4;
        size_t sequence_length = 1024;
        size_t channels = 768;

        std::vector<size_t> cpu_input_shape = { cpu_batch_size, sequence_length, channels };
        std::vector<size_t> cuda_input_shape = { cuda_batch_size, sequence_length, channels };

        cpu_layernorm = std::make_unique<Modules::LayerNorm<float, Compute::CpuMemoryResource>>(
            "Cpu_ln_1", cpu_input_shape );

        //cuda_layernorm = std::make_unique<Modules::LayerNorm<float, Compute::DeviceMemoryResource>>(
        //    "Cuda_ln_1", cuda_input_shape );

        Tensor<float, Compute::CpuMemoryResource> input( cpu_input_shape );
        random<float, Compute::CpuMemoryResource>( input, 0.0f, 5.0f );

        auto cuda_input = input.to<Compute::DeviceMemoryResource>();

        auto output = cpu_layernorm->forward( input );
        //auto output2 = cuda_layernorm->forward( std::make_shared<DeviceTensor<float>>( cuda_input ) );

        std::cout << "Cpu output: " << std::endl;
        output.print();

        std::cout << "Cuda output: " << std::endl;
        //auto from_cuda_output2 = output2->to<Compute::CpuMemoryResource>();
        //from_cuda_output2.print();
    }
}