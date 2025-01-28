module;
#include <iostream>
#include <set>
#include <string>

export module Compute.CpuDevice;

import Compute.DeviceRegistry;
import Compute.DeviceInterface;
import Compute.DeviceType;
import Compute.OperationType;

namespace Mila::Dnn::Compute
{
	export class CpuDevice : public DeviceInterface {
	public:
		std::set<OperationType> supportedOps() const override {
			return { OperationType::kLayerNormOp, OperationType::kMatMulOp };
		}

		static void registerDevice() {
			DeviceRegistry::instance().registerDevice( "CPU", []() -> std::shared_ptr<DeviceInterface> {
				return std::make_shared<CpuDevice>();
				} );
		}

		constexpr DeviceType getDeviceType() const override {
			return DeviceType::kCpu;
		}

		std::string getName() const override {
			return "CPU";
		}

	private:
		static bool registered_;
	};

	export bool CpuDevice::registered_ = (CpuDevice::registerDevice(),true);
}