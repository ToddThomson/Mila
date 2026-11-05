/**
 * @file MetalBufferLocation.ixx
 * @brief Where an address a tensor holds lies in the Metal buffers: the form Metal binds.
 */

module;

// Only the name: each module that includes metal-cpp costs every importer of Mila (MetalBackend.md section 5).
namespace MTL
{
    class Buffer;
}

export module Compute.MetalBufferLocation;

import std;

namespace Mila::Dnn::Compute
{
    export struct MetalBufferLocation
    {
        MTL::Buffer* buffer{ nullptr };
        std::size_t offset{ 0 };
    };
}
