module;
#include <memory>

export module Compute.CpuComputeResource;

import Compute.ComputeResource;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
    export class HostComputeResource : public ComputeResource {
    public:
		
        using MemoryResource = HostMemoryResource;

        HostComputeResource() : memory_resource_( std::make_shared<HostMemoryResource>() ) {}

        std::shared_ptr<HostMemoryResource> getMemoryResource() const {
            return memory_resource_;
        }

    private:
        std::shared_ptr<HostMemoryResource> memory_resource_;
    };
}