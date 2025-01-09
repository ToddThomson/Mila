module;
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <iostream>
#include <stdexcept>

export module Dnn.Model;

import Dnn.Module;
import Dnn.Tensor;

namespace Mila::Dnn
{
    export template<typename T>
    class Model : public Module<T> {
    public:

        Model() = default;

        template <typename ModuleType, typename = std::enable_if_t<std::is_base_of_v<Module<T>, ModuleType>>>
        void add( ModuleType&& module ) {
			std::string name = module.name();
            if ( module_map_.count( name ) > 0 ) {
                throw std::invalid_argument( "Module with name '" + name + "'" + " already exists." );
            }
            auto module_ptr = std::make_shared<ModuleType>( module );
            modules_.emplace_back( module_ptr );
            module_map_[ name ] = module_ptr;
            module_names_.emplace_back( name );
            //modules_.emplace_back( std::make_shared<std::remove_reference_t<ModuleType>>>( std::forward<ModuleType>( module ) ) );
        }

        Tensor<T> forward( const Tensor<T>& input ) override {
            Tensor<T> output = input;
            for ( const auto& module : modules_ ) {
                output  = module->forward( output );
            }

            return output;
        }

        // Module access functions

        // Access modules by index
        std::shared_ptr<Module<T>> operator[]( size_t index ) const {
            if ( index > modules_.size() ) {
                throw std::out_of_range( "Index out of range" );
            }
            return modules_[ index ];
        }
        // Access modules by name
        std::shared_ptr<Module<T>> operator[]( const std::string& name ) const {
            auto it = module_map_.find( name );
            if ( it == module_map_.end() ) {
                throw std::out_of_range( "No module found with name '" + name + "'." );
            }
            return it->second;
        }

		size_t parameters() override {
			size_t total_parameters = 0;
			for ( const auto& module : modules_ ) {
				total_parameters += module->parameters();
			}
			return total_parameters;
		}

		void print() override {
			std::cout << "Modules: " << std::endl;
			for ( const auto& module : modules_ ) {
				module->print();
			}
			std::cout << "Total parameters: " << parameters() << std::endl;
		}

    private:
        std::vector<std::shared_ptr<Module<T>>> modules_;
        std::unordered_map<std::string, std::shared_ptr<Module<T>>> module_map_;
        std::vector<std::string> module_names_;
    };
}