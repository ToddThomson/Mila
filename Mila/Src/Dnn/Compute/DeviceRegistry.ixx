module;
#include <vector>
#include <map>
#include <mutex>
#include <functional>
#include <string>
#include <type_traits>
#include <stdexcept>

export module Compute.DeviceRegistry;

import Compute.ComputeDevice;

export namespace Mila::Dnn::Compute
{
    export class DeviceRegistry {
    public:
        using DeviceFactory = std::function<std::shared_ptr<ComputeDevice>()>;

        static DeviceRegistry& instance() {
            static DeviceRegistry registry;

            if ( !is_initialized_ ) {
                is_initialized_ = true;
            }
            return registry;
        }

        void registerDevice( const std::string& name, DeviceFactory factory ) {
            if ( name.empty() ) {
                throw std::invalid_argument( "Device name cannot be empty." );
            }
            if ( !factory ) {
                throw std::invalid_argument( "Device factory cannot be null." );
            }

            std::lock_guard<std::mutex> lock( mutex_ );
            devices_[name] = std::move( factory );
        }

        std::shared_ptr<ComputeDevice> createDevice( const std::string& device_name ) const {
            std::lock_guard<std::mutex> lock( mutex_ );
            auto it = devices_.find( device_name );
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
