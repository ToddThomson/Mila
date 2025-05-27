module;
#include <memory>
//#include <format>

export module Compute.CudaComputeResource;

import Compute.ComputeResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn::Compute
{
    export 
    class CudaComputeResource : public ComputeResource {
    public:
        CudaComputeResource() : memory_resource_( std::make_shared<CudaMemoryResource>() ) {}

        std::shared_ptr<CudaMemoryResource> getMemoryResource() const {
            return memory_resource_;
        }

    private:
        std::shared_ptr<CudaMemoryResource> memory_resource_;
    };
}