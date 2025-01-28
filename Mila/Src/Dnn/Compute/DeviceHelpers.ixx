module;
#include <vector>
#include <string>

export module Compute.DeviceHelpers;

import Compute.DeviceContext;
import Compute.DeviceRegistry;

namespace Mila::Dnn::Compute
{
	export std::vector<std::string> list_devices() {
		// This call ensures that the DeviceContext is initialized
		auto device = Compute::DeviceContext::instance().getDevice();
		
		auto& registry = DeviceRegistry::instance();
		
		return registry.list_devices();
	}
}