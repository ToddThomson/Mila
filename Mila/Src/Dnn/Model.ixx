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
    /**
     * @brief A class representing a neural network model.
     * 
     * @tparam T The data type used for the model's parameters and computations.
     */
    export template<typename T>
    class Model : public Module<T> {
    public:

        /**
         * @brief Default constructor for the Model class.
         */
        Model() = default;

        /**
         * @brief Adds a module to the model.
         *
         * @tparam ModuleType The type of the module to add.
         * @param module The module to add.
         * @throws std::invalid_argument if a module with the same name already exists.
         */
        template <typename ModuleType> requires std::derived_from<ModuleType, Module<T>>
        size_t add( std::shared_ptr<ModuleType> module ) {
            std::string name = module->name();
            if ( std::find(module_names_.begin(), module_names_.end(), name) != module_names_.end() ) {
                throw std::invalid_argument( "Module with name '" + name + "'" + " already exists." );
            }
            //auto module_ptr = std::make_shared<ModuleType>( std::move(module) );
            //module->setParent( this );
            modules_.emplace_back( std::move(module) );
            module_names_.emplace_back( name );

            return modules_.size() - 1;
        }

        /**
         * @brief Performs a forward pass through the model.
         * 
         * @param input The input tensor.
         * @return Tensor<T> The output tensor.
         */
        std::shared_ptr<Tensor<T>> forward( const std::shared_ptr<Tensor<T>>& input ) override {
            std::shared_ptr<Tensor<T>> out = input;
            for ( const auto& module : modules_ ) {
                out = module->forward( out );
            }

            return out;
        }

		// ---------------------------------------------------------------------
        // Module access functions

        /**
         * @brief Accesses a module by its index.
         * 
         * @param index The index of the module.
         * @return std::shared_ptr<Module<T>> A shared pointer to the module.
         * @throws std::out_of_range if the index is out of range.
         */
        std::shared_ptr<Module<T>> operator[]( size_t index ) const {
            if ( index >= modules_.size() ) {
                throw std::out_of_range( "Index out of range" );
            }
            return modules_[ index ];
        }

        /**
         * @brief Accesses a module by its name.
         * 
         * @param name The name of the module.
         * @return std::shared_ptr<Module<T>> A shared pointer to the module.
         * @throws std::out_of_range if no module with the given name is found.
         */
        std::shared_ptr<Module<T>> operator[]( const std::string& name ) const {
            auto it = std::find(module_names_.begin(), module_names_.end(), name);
            if ( it == module_names_.end() ) {
                throw std::out_of_range( "No module found with name '" + name + "'." );
            }
            size_t index = std::distance(module_names_.begin(), it);
            return modules_[ index ];
        }

        /**
         * @brief Calculates the total number of parameters in the model.
         * 
         * @return size_t The total number of parameters.
         */
        size_t parameters() const override {
            size_t total_parameters = 0;
            for ( const auto& module : modules_ ) {
                total_parameters += module->parameters();
            }
            return total_parameters;
        }

        /**
         * @brief Returns the number of modules in the model.
         * 
         * @return size_t The number of modules.
         */
        size_t size() const {
            return modules_.size();
        }
        
        /**
         * @brief Prints the model's structure and total number of parameters.
         */
        void print() const override {
            std::cout << "Modules: " << std::endl;
            for ( const auto& module : modules_ ) {
                module->print();
            }
            std::cout << "Total parameters: " << parameters() << std::endl;
        }

    private:
        std::vector<std::shared_ptr<Module<T>>> modules_;
        std::vector<std::string> module_names_;
    };
}