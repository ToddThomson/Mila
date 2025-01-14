module;
#include <vector>
#include <string>

export module Compute.DeviceHelpers;

import Compute.DeviceRegistry;

namespace Mila::Dnn::Compute
{
	export std::vector<std::string> list_devices() {
		auto& registry = DeviceRegistry::instance();
		return registry.list_devices();
	}
}