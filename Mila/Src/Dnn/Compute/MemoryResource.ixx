module;
#include <memory_resource>

export module Compute.MemoryResource;

namespace Mila::Dnn::Compute
{
	export using MemoryResource = std::pmr::memory_resource;
}
