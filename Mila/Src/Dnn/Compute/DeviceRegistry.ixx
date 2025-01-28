module;
#include <vector>
#include <map>
#include <mutex>
#include <functional>
#include <string>

export module Compute.DeviceRegistry;

import Compute.DeviceInterface;
//import Compute.Devices;

export namespace Mila::Dnn::Compute
{
    export class DeviceRegistry {
    public:
        using DeviceFactory = std::function<std::shared_ptr<DeviceInterface>()>;

        static DeviceRegistry& instance() {
            static DeviceRegistry registry;

            if ( !is_initialized_ ) {
                is_initialized_ = true;

                //Devices::instance().isInitialized();
            }
            return registry;
        }

        void registerDevice( const std::string& name, DeviceFactory factory ) {
            std::lock_guard<std::mutex> lock( mutex_ );
            devices_[name] = std::move( factory );
        }

        std::shared_ptr<DeviceInterface> createDevice( const std::string& name ) const {
            std::lock_guard<std::mutex> lock( mutex_ );
            auto it = devices_.find( name );
            if (it == devices_.end()) {
                // TODO: Review throw std::runtime_error( "Invalid device type." );
                return nullptr; 
            }
            return it->second();
        }

        std::vector<std::string> list_devices() const {
			if ( !is_initialized_ ) {
				throw std::runtime_error( "Device registry is not initialized." );
			}

            std::lock_guard<std::mutex> lock( mutex_ );
            std::vector<std::string> types;
            
            for (const auto& [type, _] : devices_) {
                types.push_back( type );
            }
            return types;
        }

    private:
        DeviceRegistry() = default;
        std::map<std::string, DeviceFactory> devices_;
        mutable std::mutex mutex_;
        static inline bool is_initialized_ = false;
    };
}
