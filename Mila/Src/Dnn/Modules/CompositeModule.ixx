/**
 * @file CompositeModule.ixx
 * @brief Implementation of the composite module pattern for neural network components
 */

module;
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <sstream>

export module Dnn.CompositeModule;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ComponentConfig;
import Dnn.TensorTraits;
import Compute.Precision;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.CudaMemoryResource;
import Compute.CpuMemoryResource;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
	using namespace Mila::Dnn::Serialization;

    /**
     * @brief A module class that can contain and manage child modules.
     *
     * CompositeModule extends the base Module class with functionality to
     * add, remove, and manage child modules. This is used for composite
     * neural network components like MLPs, transformers, etc. that are
     * built by composing simpler modules.
     *
     * A single type parameter is used for data consistency across the module,
     * as the output of one layer becomes the input of the next in a feed-forward
     * composite architecture.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which the module will operate.
     * @tparam TDataType Data type used for both input and output tensor elements.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TDataType = float>
        requires ValidTensorType<TDataType>
    class CompositeModule : public Module<TDeviceType, TDataType, TDataType> {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, HostMemoryResource>; ///< Memory resource type based on device type
        using ModuleBase = Module<TDeviceType, TDataType, TDataType>; ///< Base class type for the module

        /**
         * @brief Default constructor.
         */
        CompositeModule()
            : ModuleBase() {}

        /**
         * @brief Constructor with device name.
         *
         * @param device_name The device name to use for this module.
         * @param precision The compute precision policy to use (defaults to Auto).
         */
        explicit CompositeModule(
			const std::string& device_name, const ComponentConfig& config )
            : ModuleBase( device_name, config ) {}

        /**
         * @brief Constructor with device context.
         *
         * @param context The device context to use for this module.
         * @param precision The compute precision policy to use (defaults to Auto).
         */
        explicit CompositeModule(
            std::shared_ptr<DeviceContext> context, const ComponentConfig& config )
            : ModuleBase( context, config ) {}

        /**
         * @brief Virtual destructor.
         */
        virtual ~CompositeModule() = default;

        /**
         * @brief Add a named child module to this module.
         *
         * @param name The name to identify the child module.
         * @param module The child module to register.
         * @throws std::invalid_argument If the name is empty or already exists.
         * @throws std::invalid_argument If the module pointer is null.
         * @return Reference to this module for method chaining.
         */
        CompositeModule& addModule( const std::string& name, std::shared_ptr<Module<TDeviceType, TDataType, TDataType>> module ) {
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
            module->setTraining( this->isTraining() );

            return *this;
        }

        /**
         * @brief Add an unnamed child module to this module.
         *
         * @param module The child module to register.
         * @throws std::invalid_argument If the module pointer is null.
         * @return Reference to this module for method chaining.
         */
        CompositeModule& addModule( std::shared_ptr<Module<TDeviceType, TDataType, TDataType>> module ) {
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
         * @return std::shared_ptr<Module<TDeviceType, TDataType, TDataType>> The requested module.
         * @throws std::out_of_range If no module with the given name exists.
         */
        std::shared_ptr<Module<TDeviceType, TDataType, TDataType>> getModule( const std::string& name ) const {
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
         * @return const std::vector<std::shared_ptr<Module<TDeviceType, TDataType, TDataType>>>&
         *         Vector of child module pointers.
         */
        const std::vector<std::shared_ptr<Module<TDeviceType, TDataType, TDataType>>>& getModules() const {
            return child_modules_;
        }

        /**
         * @brief Get all named sub-modules contained in this module.
         *
         * @return const std::unordered_map<std::string, std::shared_ptr<Module<TDeviceType, TDataType, TDataType>>>&
         *         Map of child module names to pointers.
         */
        const std::unordered_map<std::string, std::shared_ptr<Module<TDeviceType, TDataType, TDataType>>>& getNamedModules() const {
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
        bool replaceModule( const std::string& name, std::shared_ptr<Module<TDeviceType, TDataType, TDataType>> module ) {
            if ( !module ) {
                throw std::invalid_argument( "Cannot replace with null module." );
            }

            auto it = child_module_map_.find( name );
            if ( it == child_module_map_.end() ) {
                return false;
            }

            auto old_module = it->second;
            it->second = module;

            // Update in vector
            auto vector_it = std::find( child_modules_.begin(), child_modules_.end(), old_module );
            if ( vector_it != child_modules_.end() ) {
                *vector_it = module;
            }

            // Propagate training mode to the new module
            module->setTraining( this->isTraining() );

            return true;
        }

        /**
         * @brief Set the training mode for this module and all its sub-modules.
         *
         * @param is_training True if the module is in training mode, false for inference mode.
         */
        void setTraining( bool is_training ) override {
            ModuleBase::setTraining( is_training );

            // Propagate to all sub-modules
            for ( auto& module : child_modules_ ) {
                module->setTraining( is_training );
            }
        }

        /**
         * @brief Count the total number of parameters in this module and all sub-modules.
         *
         * @return size_t Total number of parameters.
         */
        size_t parameterCount() const override {
            size_t count = 0;

            // Count parameters from sub-modules
            for ( const auto& module : child_modules_ ) {
                count += module->parameterCount();
            }

            return count;
        }

        /**
         * @brief Default save implementation for container modules.
         *
         * Saves all child modules. Override if container has its own parameters.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( ModelArchive& archive ) const override {
            for ( const auto& module : child_modules_ ) {
                module->save( archive );
            }
        }

        /**
         * @brief Default load implementation for container modules.
         *
         * Loads all child modules. Override if container has its own parameters.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( ModelArchive& archive ) override {
            for ( const auto& module : child_modules_ ) {
                module->load( archive );
            }
        }

        /**
         * @brief Default toString implementation for container modules.
         *
         * Lists all child modules. Override for custom string representation.
         *
         * @return std::string A string representation of the module information.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "CompositeModule: " << this->getName() << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;
            oss << "Child Modules:" << std::endl;
            for ( const auto& [name, module] : child_module_map_ ) {
                oss << "  - " << name << ": " << module->toString() << std::endl;
            }
            return oss.str();
        }

    private:
        /** @brief Child modules in the order they were added (ordered) */
        std::vector<std::shared_ptr<Module<TDeviceType, TDataType, TDataType>>> child_modules_;

        /** @brief Named child modules for efficient lookup by name */
        std::unordered_map<std::string, std::shared_ptr<Module<TDeviceType, TDataType, TDataType>>> child_module_map_;
    };

    // Convenient type aliases
    export template<typename TDataType = float>
        using CpuCompositeModule = CompositeModule<DeviceType::Cpu, TDataType>;

    export template<typename TDataType = float>
        using CudaCompositeModule = CompositeModule<DeviceType::Cuda, TDataType>;
}