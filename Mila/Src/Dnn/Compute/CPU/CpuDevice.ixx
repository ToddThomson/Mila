module;
#include <iostream>
#include <set>
#include <string>

export module Compute.CpuDevice;

import Compute.DeviceRegistry;
import Compute.DeviceInterface;
import Compute.Operations;

namespace Mila::Dnn::Compute::Cpu
{
	export class CpuDevice : public DeviceInterface {
	public:
		std::set<Operation> supportedOps() const override {
			return { Operation::LayerNorm };
		}

		std::string name() const override {
			return "CPU";
		}

		static void RegisterDevice() {
			DeviceRegistry::instance().registerDevice( "CPU", []() -> std::unique_ptr<DeviceInterface> {
				return std::make_unique<CpuDevice>();
				} );
		}

	private:
		static bool registered_;
	};

	//export bool CpuDevice::registered_ = (CpuDevice::RegisterDevice(),true);
}