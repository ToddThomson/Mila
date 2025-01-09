module;
#include <string>
#include <stdexcept>
#include <set>
#include <memory>

export module Dnn.Session;

import Compute.BackendRegistry;
import Compute.BackendInterface;
import Compute.CudaBackend;
import Compute.Operations;

namespace Mila::Dnn
{
	using namespace Compute;

	export class Session {
	public:
		explicit Session( const std::string& backendName = "CPU", int deviceId = 0 ) {
			setBackend( backendName, deviceId );
		}

		void setBackend( const std::string& backendName, int deviceId = 0 ) {
			if (backendName.rfind( "CUDA", 0 ) == 0) {
				backend_ = std::make_unique<Cuda::CudaBackend>( deviceId );
			}
			else {
				backend_ = Compute::BackendRegistry::instance().createBackend( backendName );
			}
		}

		std::set<Operation> supportedOperations() const {
			return backend_->supportedOperations();
		}

		BackendInterface& getBackend() {
			if (!backend_) {
				throw std::runtime_error( "Backend not configured." );
			}

			return *backend_;
		}

	private:

		std::unique_ptr<Compute::BackendInterface> backend_;
	};
}
