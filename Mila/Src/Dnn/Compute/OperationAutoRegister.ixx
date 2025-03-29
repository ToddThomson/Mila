module;
#include <memory>
#include <string>

export module Compute.AutoRegisterOp;

import Compute.OpRegistry;
import Compute.OperationRegistry;
import Compute.DeviceType;
import Compute.OperationBase;

namespace Mila::Dnn::Compute
{
	export template <typename T>
		class AutoRegisterOp {
		protected:
			static inline bool registered = [] {
				OpRegistry::instance().registerClass(
					T::className(),
					[]() -> void* {
						return new T();
					}
				);
				return true;
				}();
	};

	// Specialized AutoRegisterOp for OperationBase derivatives
	export template <typename T, typename TInput, typename TPrecision, Compute::DeviceType TDeviceType>
		class OperationAutoRegisterOp {
		protected:
			// Forward declare the expected className function from T
			static std::string GetClassName() {
				return T::className();
			}
			static inline bool registered = [] {
				Compute::OperationRegistry::instance().registerOperation<TInput, TPrecision, TDeviceType>(
					GetClassName(),
					[]() -> std::unique_ptr<Compute::OperationBase<TInput, TPrecision, TDeviceType>> {
						return std::make_unique<T>();
					}
				);
				return true;
				}();
	};
}