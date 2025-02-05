module;
#include <iostream>
#include <set>
#include <string>

export module Compute.CpuDevice;

import Compute.DeviceRegistry;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.OperationType;

namespace Mila::Dnn::Compute
{
	export class CpuDevice : public ComputeDevice {
	public:
		static void registerDevice() {
			DeviceRegistry::instance().registerDevice( "CPU", []() -> std::shared_ptr<ComputeDevice> {
				return std::make_shared<CpuDevice>();
				} );
		}

		constexpr DeviceType getDeviceType() const override {
			return DeviceType::Cpu;
		}

		std::string getName() const override {
			return "CPU";
		}

	private:
		static bool registered_;
	};

	export bool CpuDevice::registered_ = (CpuDevice::registerDevice(),true);
}