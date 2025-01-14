module;
#include <vector>
#include <map>
#include <mutex>
#include <functional>

export module Compute.DeviceRegistry;

import Compute.DeviceInterface;

export namespace Mila::Dnn::Compute
{
    export class DeviceRegistry {
    public:
        using DeviceFactory = std::function<std::shared_ptr<DeviceInterface>()>;

        static DeviceRegistry& instance() {
            static DeviceRegistry registry;
            return registry;
        }

        void registerDevice( const std::string name, DeviceFactory factory ) {
            std::lock_guard<std::mutex> lock( mutex_ );
            devices_[name] = std::move( factory );
        }

		std::shared_ptr<DeviceInterface> createDevice( const std::string name ) const {
			std::lock_guard<std::mutex> lock( mutex_ );
			auto it = devices_.find( name );
			if (it == devices_.end()) {
                return nullptr; // throw std::runtime_error( "Invalid device name." );
			}
			return it->second();
		}



		std::vector<std::string> list_devices() const {
			std::lock_guard<std::mutex> lock( mutex_ );
			std::vector<std::string> names;
			
            for (const auto& [name, _] : devices_) {
				names.push_back( name );
			}
			return names;
        }

    private:
        DeviceRegistry() = default;
        std::map<std::string, DeviceFactory> devices_;
        mutable std::mutex mutex_;
    };
}
