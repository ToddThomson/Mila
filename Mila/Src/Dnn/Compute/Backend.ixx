module;
#include <memory_resource>
#include <type_traits>

export module Compute.Backend;

namespace Mila::Dnn::Compute
{
    export
    template<typename T>
    concept IsCpuComputeResource = requires {
        { T::is_cpu_compute() } -> std::convertible_to<bool>;
    };

    export
    template<typename T>
    concept IsCudaComputeResource = requires {
        { T::is_cuda_compute() } -> std::convertible_to<bool>;
    };

	
}