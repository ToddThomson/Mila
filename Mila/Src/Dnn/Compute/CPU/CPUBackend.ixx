module;
#include <iostream>
#include <set>
#include <string>

export module Compute.CpuBackend;

import Compute.BackendRegistry;
import Compute.BackendInterface;
import Compute.Operations;

namespace Mila::Dnn::Compute::Cpu
{
	export class CpuBackend : public BackendInterface {
	public:
		std::set<Operation> supportedOperations() const override {
			return { Operation::MatrixMultiply, Operation::LayerNorm };
		}

		std::string name() const override {
			return "CPU";
		}
	};

	struct CpuBackendRegistration {
		CpuBackendRegistration() {
			BackendRegistry::instance().registerBackend( "CPU", []() {
				return std::make_unique<CpuBackend>();
				} );
		}
	};
}