#include <iostream>
#include <memory>
#include <vector>

import Mila;

using namespace Mila::Dnn;

namespace Scratchpad::Linear
{
    void SampleLinear() {
        std::unique_ptr<Modules::Linear<float, Compute::CpuMemoryResource>> cpu_linear;
        std::unique_ptr<Modules::Linear<float, Compute::DeviceMemoryResource>> cuda_linear;

        size_t cuda_batch_size = 4;
        size_t cpu_batch_size = 4;
        size_t sequence_length = 1024;
        size_t channels = 768;

        std::vector<size_t> cpu_input_shape = { cpu_batch_size, sequence_length, channels };
        std::vector<size_t> cuda_input_shape = { cuda_batch_size, sequence_length, channels };

        size_t output_channels = 3 * channels;

        cpu_linear = std::make_unique<Modules::Linear<float, Compute::CpuMemoryResource>>(
            "cpu_linear_1", cpu_input_shape, output_channels, true );

        cuda_linear = std::make_unique<Modules::Linear<float, Compute::DeviceMemoryResource>>(
            "cuda_linear_2", cuda_input_shape, output_channels, true );

        Tensor<float, Compute::CpuMemoryResource> input( cpu_input_shape );
        random<float, Compute::CpuMemoryResource>( input, 0.0f, 5.0f );

        auto cuda_input = input.to<Compute::DeviceMemoryResource>();

        auto output = cpu_linear->forward( std::make_shared<HostTensor<float>>( input ) );
        auto output2 = cuda_linear->forward( std::make_shared<DeviceTensor<float>>( cuda_input ) );

        std::cout << "Cpu output: " << std::endl;
        output->print();

        std::cout << "Cuda output: " << std::endl;
        auto from_cuda_output2 = output2->to<Compute::CpuMemoryResource>();
        from_cuda_output2.print();
    }
}