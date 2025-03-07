export module Compute.MemoryResourceProperties;

namespace Mila::Dnn::Compute
{
    export struct CpuAccessible {
        static constexpr bool is_cpu_accessible = true;
    };

    export struct CudaAccessible {
        static constexpr bool is_cuda_accessible = true;
    };
}