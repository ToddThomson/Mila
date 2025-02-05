module;
#include <string>

export module Compute.ComputeDevice;

import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
	export class ComputeDevice {
	public:
		virtual ~ComputeDevice() = default;

		virtual constexpr DeviceType getDeviceType() const = 0;

		virtual std::string getName() const = 0;
	};
}