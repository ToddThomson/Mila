module;
#include <memory>
#include <cstring>
#include <algorithm> // for std::fill_n

export module Compute.CpuComputeResource;

import Compute.ComputeResource;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
    // TODO: Remove
    /*export class HostComputeResource : public ComputeResource {
    public:
		
        using MemoryResource = CpuMemoryResource;

        HostComputeResource() : memory_resource_( std::make_shared<CpuMemoryResource>() ) {}

        std::shared_ptr<CpuMemoryResource> getMemoryResource() const {
            return memory_resource_;
        }

        void fillImpl( void* data, size_t element_count, const void* value_ptr, size_t element_size ) override {
            char* byte_data = static_cast<char*>(data);
            for ( size_t i = 0; i < element_count; ++i ) {
                std::memcpy( byte_data + i * element_size, value_ptr, element_size );
            }
        }

        void memcpy( void* dst, const void* src, size_t size_bytes ) override {
            std::memcpy( dst, src, size_bytes );
        }

    private:
        std::shared_ptr<CpuMemoryResource> memory_resource_;
    };*/
}