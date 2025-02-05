export module Compute.ComputeResource;

namespace Mila::Dnn::Compute
{
	export struct CpuResource {
		static constexpr bool is_host_accessible = true;
	};

	export struct CudaResource {
		static constexpr bool is_device_accessible = true;
	};
}