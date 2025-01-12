module;
#include <string>
#include <stdexcept>
#include <set>
#include <memory>

export module Dnn.Session;

import Compute.DeviceRegistry;
import Compute.DeviceInterface;
import Compute.CudaDevice;
import Compute.Operations;

namespace Mila::Dnn
{
	using namespace Compute;

	export class Session {
	public:
		explicit Session( const std::string& deviceName = "CPU", int deviceId = 0 ) {
			set_device( deviceName, deviceId );
		}

		void set_device( const std::string& deviceName, int deviceId = 0 ) {
			if (deviceName.rfind( "CUDA", 0 ) == 0) {
				device_ = std::make_unique<Cuda::CudaDevice>( deviceId );
			}
			else {
				device_ = Compute::DeviceRegistry::instance().createDevice( deviceName );
			}
		}

		std::set<Operation> supportedOps() const {
			return device_->supportedOps();
		}

		DeviceInterface& getDevice() {
			if (!device_) {
				throw std::runtime_error( "Device not configured." );
			}

			return *device_;
		}

	private:

		std::unique_ptr<Compute::DeviceInterface> device_;
	};
}
