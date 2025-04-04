/**
 * @file Module.ixx
 * @brief Defines the base Module class for the Mila deep neural network framework.
 */

module;
#include <miniz.h>  
#include <vector>  
#include <string>  
#include <memory>  
#include <unordered_map>
#include <stdexcept>  
#include <type_traits>  
#include <sstream>  

export module Dnn.Module;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.DeviceType;
import Compute.MemoryResource;  
import Compute.CpuMemoryResource;  
import Compute.CudaMemoryResource;  

namespace Mila::Dnn
{
    /**
     * @brief Abstract base class for all modules in the Mila DNN framework.
     *
     * The Module class provides a common interface for all neural network layers
     * and components, enabling consistent handling of parameters, state, and
     * hierarchical relationships between modules.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TPrecision Data type of the compute tensor elements, defaults to TInput.
     * @tparam TDeviceType Device type where computation is performed, defaults to CUDA.
     */
    export
        template<typename TInput, typename TPrecision = TInput, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
        requires ValidTensorTypes<TInput, TPrecision>
    class Module {
    public:
        /** @brief Type alias for the memory resource based on device type */
        using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::DeviceMemoryResource, Compute::HostMemoryResource>;

        /** @brief Virtual destructor for proper cleanup in derived classes */
        virtual ~Module() = default;

        /**
         * @brief Set the training mode of the module.
         *
         * Many modules behave differently during training versus inference
         * (e.g., dropout, batch normalization).
         *
         * @param training True if the module is in training mode, false for inference mode.
         */
        void setTrainingMode( bool training ) {
            is_training_ = training;
        }

        /**
         * @brief Check if the module is in training mode.
         *
         * @return true If the module is in training mode.
         * @return false If the module is in inference mode.
         */
        bool isTraining() const {
            return is_training_;
        }

        /**
         * @brief Get the number of trainable parameters in the module.
         *
         * @return size_t Total number of parameters.
         */
        virtual size_t parameterCount() const = 0;

        /**
         * @brief Get the parameter tensors of this module.
         *
         * @return const std::unordered_map<std::string, std::shared_ptr<Tensor<TPrecision, MR>>>&
         *         Map of parameter names to tensor pointers.
         */
        const std::unordered_map<std::string, std::shared_ptr<Tensor<TPrecision, MR>>>& getParameterTensors() const {
            return parameter_map_;
        }

        /**
         * @brief Get the state tensors of this module.
         *
         * State tensors represent non-trainable tensors that may be updated during
         * forward/backward passes (e.g., running mean in batch normalization).
         *
         * @return const std::unordered_map<std::string, std::shared_ptr<Tensor<TPrecision, MR>>>&
         *         Map of state names to tensor pointers.
         */
        const std::unordered_map<std::string, std::shared_ptr<Tensor<TPrecision, MR>>>& getStateTensors() const {
            return state_map_;
        }

        /**
         * @brief Get the name of the module.
         *
         * @return std::string Name of the module.
         */
        std::string getName() const {
            return name_;
        }

        /**
         * @brief Set the name of the module.
         *
         * @param name The name to set. Must not be empty and cannot contain a dot ('.').
         * @throws std::invalid_argument If the name is empty or contains a dot.
         */
        void setName( const std::string& name ) {
            if ( name.empty() ) {
                throw std::invalid_argument( "Name must not be empty and cannot contain a dot ('.')." );
            }
            name_ = name;
        }

        /**
        * @brief Save the module state to a zip archive.
        *
        * Serializes the module's parameters and state to the provided zip archive.
        *
        * @param zip The zip archive to save the state to.
        */
        virtual void save( mz_zip_archive& zip ) const = 0;

        /**
         * @brief Load the module state from a zip archive.
         *
         * Deserializes the module's parameters and state from the provided zip archive.
         *
         * @param zip The zip archive to load the state from.
         */
        virtual void load( mz_zip_archive& zip ) = 0;

        /**
         * @brief Add a named child module to this module.
         *
         * This method allows building hierarchical neural network structures with
         * named sub-modules for easier access.
         *
         * @param name The name to identify the child module. Must be unique.
         * @param module The child module to register.
         * @throws std::invalid_argument If the name is empty or already exists.
         * @throws std::invalid_argument If the module pointer is null.
         * @return Reference to this module for method chaining.
        */
        Module& addModule( const std::string& name, std::shared_ptr<Module<TInput, TPrecision, TDeviceType>> module ) {
            if ( name.empty() ) {
                throw std::invalid_argument( "Sub-module name cannot be empty." );
            }

            if ( !module ) {
                throw std::invalid_argument( "Cannot add null module." );
            }

            if ( child_module_map_.find( name ) != child_module_map_.end() ) {
                throw std::invalid_argument( "Sub-module name '" + name + "' already exists." );
            }

            // Add to both map and vector for different access patterns
            child_module_map_[ name ] = module;
            child_modules_.push_back( module );

            // Propagate training mode to the newly added module
            module->setTraining( is_training_ );

            return *this;
        }

        /**
         * @brief Add an unnamed child module to this module.
         *
         * This method is a convenience wrapper that auto-generates a name
         * based on the module's type and position.
         *
         * @param module The child module to register.
         * @throws std::invalid_argument If the module pointer is null.
         * @return Reference to this module for method chaining.
         */
        Module& addModule( std::shared_ptr<Module<TInput, TPrecision, TDeviceType>> module ) {
            if ( !module ) {
                throw std::invalid_argument( "Cannot add null module." );
            }

            // Generate a unique name based on index
            std::string auto_name = "module_" + std::to_string( child_modules_.size() );
            return addModule( auto_name, module );
        }

        /**
         * @brief Get a specific sub-module by name.
         *
         * @param name The name of the sub-module to retrieve.
         * @return std::shared_ptr<Module<TInput, TPrecision, TDeviceType>> The requested module.
         * @throws std::out_of_range If no module with the given name exists.
         */
        std::shared_ptr<Module<TInput, TPrecision, TDeviceType>> getModule( const std::string& name ) const {
            auto it = child_module_map_.find( name );
            if ( it == child_module_map_.end() ) {
                throw std::out_of_range( "No module named '" + name + "' found." );
            }
            return it->second;
        }

        /**
         * @brief Check if a sub-module with the given name exists.
         *
         * @param name The name to check.
         * @return bool True if a sub-module with the given name exists.
         */
        bool hasModule( const std::string& name ) const {
            return child_module_map_.find( name ) != child_module_map_.end();
        }

        /**
         * @brief Get all sub-modules contained in this module.
         *
         * @return const std::vector<std::shared_ptr<Module<TInput, TPrecision, TDeviceType>>>&
         *         Vector of child module pointers.
         */
        const std::vector<std::shared_ptr<Module<TInput, TPrecision, TDeviceType>>>& getModules() const {
            return child_modules_;
        }

        /**
         * @brief Get all named sub-modules contained in this module.
         *
         * @return const std::unordered_map<std::string, std::shared_ptr<Module<TInput, TPrecision, TDeviceType>>>&
         *         Map of child module names to pointers.
         */
        const std::unordered_map<std::string, std::shared_ptr<Module<TInput, TPrecision, TDeviceType>>>& getNamedModules() const {
            return child_module_map_;
        }

        /**
         * @brief Remove a sub-module by name.
         *
         * @param name The name of the sub-module to remove.
         * @return bool True if a module was removed, false if no module with that name existed.
         */
        bool removeModule( const std::string& name ) {
            auto it = child_module_map_.find( name );
            if ( it == child_module_map_.end() ) {
                return false;
            }

            auto module_ptr = it->second;

            // Remove from map
            child_module_map_.erase( it );

            // Remove from vector
            auto vector_it = std::find( child_modules_.begin(), child_modules_.end(), module_ptr );
            if ( vector_it != child_modules_.end() ) {
                child_modules_.erase( vector_it );
            }

            return true;
        }

        /**
         * @brief Replace an existing sub-module with a new one.
         *
         * @param name The name of the sub-module to replace.
         * @param module The new module to use as replacement.
         * @return bool True if a module was replaced, false if no module with that name existed.
         * @throws std::invalid_argument If the replacement module pointer is null.
         */
        bool replaceModule( const std::string& name, std::shared_ptr<Module<TInput, TPrecision, TDeviceType>> module ) {
            if ( !module ) {
                throw std::invalid_argument( "Cannot replace with null module." );
            }

            auto it = child_module_map_.find( name );
            if ( it == child_module_map_.end() ) {
                return false;
            }

            auto old_module = it->second;

            // Update in map
            it->second = module;

            // Update in vector
            auto vector_it = std::find( child_modules_.begin(), child_modules_.end(), old_module );
            if ( vector_it != child_modules_.end() ) {
                *vector_it = module;
            }

            // Propagate training mode to the new module
            module->setTraining( is_training_ );

            return true;
        }

        /**
         * @brief Set the training mode for this module and all its sub-modules.
         *
         * This method propagates the training mode setting to all child modules.
         *
         * @param is_training True if the module is in training mode, false for inference mode.
         */
        void setTraining( bool is_training ) {
            is_training_ = is_training;

            // Propagate to all sub-modules
            for ( auto& module : child_modules_ ) {
                module->setTraining( is_training );
            }
        }

        /**
         * @brief Count the total number of parameters in this module and all sub-modules.
         *
         * @param include_submodules Whether to include parameters from sub-modules in the count.
         * @return size_t Total number of parameters.
         */
        size_t totalParameterCount( bool include_submodules = true ) const {
            size_t count = parameterCount();

            if ( include_submodules ) {
                for ( const auto& module : child_modules_ ) {
                    count += module->parameterCount();
                }
            }

            return count;
        }

        /**
         * @brief Convert the module to a string representation.
         *
         * This should include relevant information about the module structure,
         * parameters, and configuration.
         *
         * @return std::string String representation of the module.
         */
        virtual std::string toString() const = 0;

        /**
         * @brief Overload the << operator to print the module information.
         *
         * @param os Output stream.
         * @param module Module to print.
         * @return std::ostream& Reference to the output stream.
         */
        friend std::ostream& operator<<( std::ostream& os, const Module& module ) {
            os << module.toString();
            return os;
        }

    protected:
        /** @brief Map of parameter names to parameter tensors */
        std::unordered_map<std::string, std::shared_ptr<Tensor<TPrecision, MR>>> parameter_map_ = {};

        /** @brief Map of state names to state tensors */
        std::unordered_map<std::string, std::shared_ptr<Tensor<TPrecision, MR>>> state_map_ = {};

        /**
         * @brief Helper method to convert parameters to string representation.
         *
         * @return std::string String representation of all parameters.
         */
        const std::string parametersToString() const {
            std::ostringstream oss;
            for ( const auto& [name, tensor] : getParameterTensors() ) {
                oss << tensor->toString();
            }
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

        /**
         * @brief Helper method to convert state tensors to string representation.
         *
         * @return std::string String representation of all state tensors.
         */
        const std::string stateToString() const {
            std::ostringstream oss;
            for ( const auto& [name, tensor] : getStateTensors() ) {
                oss << tensor->toString();
            }
            return oss.str();
        }


    private:
        std::string name_; ///< The name of the module. Cannot be empty and cannot contain a dot ('.').  
        bool is_training_{ false }; ///< Whether the module is in training mode. Default is false.  

        std::vector<std::shared_ptr<Module<TInput, TPrecision, TDeviceType>>> child_modules_; ///< Child modules (ordered).
        std::unordered_map<std::string, std::shared_ptr<Module<TInput, TPrecision, TDeviceType>>> child_module_map_; ///< Named child modules.
    };
}
