module;
#include <vector>
#include <iostream>
#include <set>
#include <string>

export module Compute.BackendInterface;

import Compute.Operations;

namespace Mila::Dnn::Compute
{
	export class BackendInterface {
	public:
		virtual ~BackendInterface() = default;

		virtual int reduce( const std::vector<int>& data ) {
			throw std::runtime_error( "operation not supported: reduce " );
		}

		virtual std::set<Operation> supportedOperations() const = 0;

		virtual std::string name() const = 0;
	};
}

