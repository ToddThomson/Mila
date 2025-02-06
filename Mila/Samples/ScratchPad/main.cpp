#include <iostream>
#include <memory>

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

    std::cout << "The current Compute Device is: " << Mila::getDevice()->getName() << std::endl;

    std::unique_ptr<Modules::MatMul<float, Compute::CpuMemoryResource>> cpu_matmul;
    std::unique_ptr<Modules::MatMul<float, Compute::DeviceMemoryResource>> cuda_matmul;
    
    size_t cuda_batch_size = 4;
	size_t cpu_batch_size = 4;
    size_t sequence_length = 1024;
    size_t channels = 768;

	std::vector<size_t> cpu_input_shape = { cpu_batch_size, sequence_length, channels };
    std::vector<size_t> cuda_input_shape = { cuda_batch_size, sequence_length, channels };

    size_t output_channels = 3 * channels;

    cpu_matmul = std::make_unique<Modules::MatMul<float, Compute::CpuMemoryResource>>( 
        "CpuMatMul_1", cpu_input_shape, output_channels, true );

    cuda_matmul = std::make_unique<Modules::MatMul<float, Compute::DeviceMemoryResource>>(
        "CudaMatMul_2", cuda_input_shape, output_channels, true );

    Tensor<float, Compute::CpuMemoryResource> input( cpu_input_shape );
    random<float,Compute::CpuMemoryResource>( input, 0.0f, 5.0f );

    input.print();

	auto cuda_input = input.to<Compute::DeviceMemoryResource>();

    auto output = cpu_matmul->forward( std::make_shared<HostTensor<float>>( input ) );
    auto output2 = cuda_matmul->forward( std::make_shared<DeviceTensor<float>>( cuda_input ) );

	std::cout << "Cpu output: " << std::endl;
	output->print();

	std::cout << "Cuda output: " << std::endl;
	auto from_cuda_output2 = output2->to<Compute::CpuMemoryResource>();
	from_cuda_output2.print();

    return 0;
}