module;
#include <memory>
//#include <format>

export module Compute.CudaComputeResource;

import Compute.ComputeResource;
import Compute.CudaDeviceMemoryResource;

namespace Mila::Dnn::Compute
{
	// TODO: Remove this class if not needed
    /*export 
    class CudaComputeResource : public ComputeResource {
    public:
        CudaComputeResource() : memory_resource_( std::make_shared<CudaDeviceMemoryResource>() ) {}

        std::shared_ptr<CudaDeviceMemoryResource> getMemoryResource() const {
            return memory_resource_;
        }



        

    private:
        std::shared_ptr<CudaDeviceMemoryResource> memory_resource_;
    };*/
}