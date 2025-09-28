module;
#include <memory_resource>
#include <type_traits>

export module Compute.ResourceHelpers;

namespace Mila::Dnn::Compute
{
	// TODO: Remove these concepts if not needed

    /*export
    template<typename T>
    concept IsCpuComputeResource = requires {
        { T::is_cpu_compute() } -> std::convertible_to<bool>;
    };

    export
    template<typename T>
    concept IsCudaComputeResource = requires {
        { T::is_cuda_compute() } -> std::convertible_to<bool>;
    };*/
}