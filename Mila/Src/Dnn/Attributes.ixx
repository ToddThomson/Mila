module;
#include <type_traits>
#include <unordered_map>


export module Dnn.ModuleAttributes;

namespace Mila::Dnn
{
	export class IOperationTensors {
	public:
		virtual ~IOperationTensors() = default;

		virtual std::unordered_map<int64_t, void*> get_inputs_tensors() = 0;
		virtual std::unordered_map<int64_t, void*> get_output_tensors() = 0;
		virtual std::unordered_map<int64_t, void*> get_options_tensors() = 0;
	};

	export template<class Derived>
	class OperationTensorsBase : public IOperationTensors {
	public:
		std::unordered_map<int64_t, void*> get_inputs_tensors() override {
			return static_cast<Derived*>(this)->get_inputs();
		}
	};
}
