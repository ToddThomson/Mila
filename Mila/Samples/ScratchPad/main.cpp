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

    std::cout << "The current ComputeDevice is: " << Mila::getDevice()->getName() << std::endl;

    std::unique_ptr<Modules::MatMul<float, Compute::CpuMemoryResource>> cpu_matmul;
    std::unique_ptr<Modules::MatMul<float, Compute::DeviceMemoryResource>> cuda_matmul;
    
    size_t cuda_batch_size_ = 128;
	size_t cpu_batch_size_ = 2;
    size_t sequence_length_ = 1024;
    size_t channels_ = 768;
    size_t output_channels_ = 3 * channels_;

    cpu_matmul = std::make_unique<Modules::MatMul<float, Compute::CpuMemoryResource>>( "CpuMatMul_1", cpu_batch_size_, sequence_length_, channels_, output_channels_, true );

    cuda_matmul = std::make_unique<Modules::MatMul<float, Compute::DeviceMemoryResource>>( "CudaMatMul_2", cuda_batch_size_, sequence_length_, channels_, output_channels_, true );

    Tensor<float, Compute::CpuMemoryResource> input( { cpu_batch_size_, sequence_length_, channels_ } );
    random( input, 0.0f, 5.0f );

    input.print();

    auto output = cpu_matmul->forward( std::make_shared<HostTensor<float>>( input ) );

    //auto output = cuda_matmul->forward( std::make_shared<Tensor<float,Compute::CpuMemoryResource>>( input ) );

	output->print();

    return 0;
}