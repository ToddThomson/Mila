module;
#include <memory>

export module Compute.CudaComputeResource;

import Compute.ComputeResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn::Compute
{
    export 
    class CudaComputeResource : public ComputeResource {
    public:
        CudaComputeResource() : memory_resource_( std::make_shared<DeviceMemoryResource>() ) {}

        std::shared_ptr<DeviceMemoryResource> getMemoryResource() const {
            return memory_resource_;
        }

    private:
        std::shared_ptr<DeviceMemoryResource> memory_resource_;
    };
}