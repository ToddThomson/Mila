module;
#include <unordered_map>
#include <string>
#include <any>

export module Compute.OperationProperties;

namespace Mila::Dnn::Compute
{
	/**
	* @brief Common operation properties that are frequently used
	*/
	export
	struct OperationProperties {
		// Common properties with default values
		int64_t axis = -1;      // Used by Softmax, LayerNorm, etc.
		float epsilon = 1e-5f;  // Used by LayerNorm, etc.

		// When needed, uncommon properties can be stored in the props_map
		std::unordered_map<std::string, std::any> props_map;

		// Helper methods to get values from the props_map with type safety
		template<typename T>
		T get( const std::string& key, const T& default_value ) const {
			auto it = props_map.find( key );
			if ( it != props_map.end() ) {
				try {
					return std::any_cast<T>(it->second);
				}
				catch ( const std::bad_any_cast& ) {
					// Return default if type doesn't match
					return default_value;
				}
			}
			return default_value;
		}

		// Helper method to set a value in the props_map
		template<typename T>
		void set( const std::string& key, const T& value ) {
			props_map[ key ] = value;
		}
	};
}