module;
#include <vector>
#include <map>
#include <mutex>
#include <functional>

export module Compute.BackendRegistry;

import Compute.BackendInterface;

export namespace Mila::Dnn::Compute
{
    export class BackendRegistry {
    public:
        using BackendFactory = std::function<std::unique_ptr<BackendInterface>()>;

        static BackendRegistry& instance() {
            static BackendRegistry registry;
            return registry;
        }

        void registerBackend( const std::string name, BackendFactory factory ) {
            std::lock_guard<std::mutex> lock( mutex_ );
            backends_[name] = std::move( factory );
        }

		std::unique_ptr<BackendInterface> createBackend( const std::string name ) const {
			std::lock_guard<std::mutex> lock( mutex_ );
			auto it = backends_.find( name );
			if (it == backends_.end()) {
				return nullptr;
			}
			return it->second();
		}

		std::vector<std::string> listBackends() const {
			std::lock_guard<std::mutex> lock( mutex_ );
			std::vector<std::string> names;
			
            for (const auto& [name, _] : backends_) {
				names.push_back( name );
			}
			return names;
        }

    private:
        BackendRegistry() = default;
        std::map<std::string, BackendFactory> backends_;
        mutable std::mutex mutex_;
    };
}
