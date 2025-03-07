module;
#include <memory>

export module Compute.CpuComputeResource;

import Compute.ComputeResource;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
    export class CpuComputeResource : public ComputeResource {
    public:
		
        using MemoryResource = CpuMemoryResource;

        CpuComputeResource() : memory_resource_( std::make_shared<CpuMemoryResource>() ) {}

        std::shared_ptr<CpuMemoryResource> getMemoryResource() const {
            return memory_resource_;
        }

    private:
        std::shared_ptr<CpuMemoryResource> memory_resource_;
    };
}