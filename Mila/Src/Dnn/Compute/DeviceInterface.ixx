module;
#include <vector>
#include <iostream>
#include <set>
#include <string>

export module Compute.DeviceInterface;

//import Dnn.Tensor;
import Compute.Operations;

namespace Mila::Dnn::Compute
{
	export class DeviceInterface {
	public:
		virtual ~DeviceInterface() = default;

		virtual std::set<Operation> supportedOps() const = 0;

		virtual std::string name() const = 0;

	};
}

