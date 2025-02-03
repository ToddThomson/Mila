export module Compute.MemoryResourceProperties;

namespace Mila::Dnn::Compute
{
	export struct HostAccessible {
		static constexpr bool is_host_accessible = true;
	};

	export struct DeviceAccessible {
		static constexpr bool is_device_accessible = true;
	};
}