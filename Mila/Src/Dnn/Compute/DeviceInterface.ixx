module;
#include <iostream>
#include <set>
#include <string>

export module Compute.DeviceInterface;

import Compute.OperationType;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
	export class DeviceInterface {
	public:
		virtual ~DeviceInterface() = default;

		virtual constexpr DeviceType getDeviceType() const = 0;

		virtual std::set<OperationType> supportedOps() const = 0;

		virtual std::string getName() const = 0;
	};
}