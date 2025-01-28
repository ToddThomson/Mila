module;

export module Compute.Devices;

//import Compute.CudaDevice;
//import Compute.CpuDevice;

namespace Mila::Dnn::Compute
{
	export class Devices {
	public:
		static Devices& instance() {
			static Devices instance;

			// Lazy initialization of devices
			if ( !is_initialized_ ) {
				is_initialized_ = true;

				//CpuDevice::registerDevice();
				//CudaDevice::registerDevices();
			}

			return instance;
		}

		const bool isInitialized() const {
			return is_initialized_;
		}

		Devices( const Devices& ) = delete;
		Devices operator=( const Devices& ) = delete;

	private:
		Devices() = default;
		static inline bool is_initialized_ = false;
	};
}