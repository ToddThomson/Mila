/**
 * @file CompositeModule.ixx
 * @brief Composite container module for Mila DNN components.
 *
 * Provides a container that owns and manages child modules. CompositeModule
 * is intentionally an abstract container (it does not implement forward/backward)
 * and focuses on module composition, lifecycle and state propagation.
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
import Compute.DeviceType;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief A module that contains and manages child modules.
     *
     * CompositeModule is a device-parameterized abstract container. It does not
     * implement the computational interface (forward/backward/synchronize) so
     * derived types must provide execution semantics while benefiting from
     * standardized child management.
     */
    export template<DeviceType TDeviceType>
        class CompositeModule : public Module<TDeviceType>
    {
    public:
        using ModuleBase = Module<TDeviceType>;
        using ModulePtr = std::shared_ptr<Module<TDeviceType>>;

        /**
         * @brief Construct a composite module with an optional name.
         *
         * @param name Stable identifier returned by `getName()`.
         */
        explicit CompositeModule()
            : is_training_( false )
        {
        }

        virtual ~CompositeModule() = default;

        /**
         * @brief Add a named child module.
         *
         * Throws on empty name, null module or duplicate name.
         */
        CompositeModule& addModule( const std::string& name, ModulePtr module )
        {
            if (name.empty())
            {
                throw std::invalid_argument( "Sub-module name cannot be empty." );
            }

            if (!module)
            {
                throw std::invalid_argument( "Cannot add null module." );
            }

            if (child_module_map_.find( name ) != child_module_map_.end())
            {
                throw std::invalid_argument( "Sub-module name '" + name + "' already exists." );
            }

            child_module_map_[name] = module;
            child_modules_.push_back( module );

            module->setTraining( is_training_ );

            return *this;
        }

        /**
         * @brief Add an unnamed child module; a stable auto-generated name is assigned.
         */
        CompositeModule& addModule( ModulePtr module )
        {
            if (!module)
            {
                throw std::invalid_argument( "Cannot add null module." );
            }

            std::string auto_name = "module_" + std::to_string( child_modules_.size() );
            
            return addModule( auto_name, module );
        }

        /**
         * @brief Return a child module by name or throw std::out_of_range.
         */
        ModulePtr getModule( const std::string& name ) const
        {
            auto it = child_module_map_.find( name );
            if (it == child_module_map_.end())
            {
                throw std::out_of_range( "No module named '" + name + "' found." );
            }
            
            return it->second;
        }

        /**
         * @brief True if a named child exists.
         */
        bool hasModule( const std::string& name ) const
        {
            return child_module_map_.find( name ) != child_module_map_.end();
        }

        /**
         * @brief All child modules in insertion order.
         */
        const std::vector<ModulePtr>& getModules() const
        {
            return child_modules_;
        }

        /**
         * @brief Named child modules map.
         */
        const std::unordered_map<std::string, ModulePtr>& getNamedModules() const
        {
            return child_module_map_;
        }

        /**
         * @brief Remove a child module by name.
         *
         * Returns true if removed, false if not found.
         */
        bool removeModule( const std::string& name )
        {
            auto it = child_module_map_.find( name );
            if (it == child_module_map_.end())
            {
                return false;
            }

            auto module_ptr = it->second;

            child_module_map_.erase( it );

            auto vector_it = std::find( child_modules_.begin(), child_modules_.end(), module_ptr );
            if (vector_it != child_modules_.end())
            {
                child_modules_.erase( vector_it );
            }

            return true;
        }

        /**
         * @brief Replace a named child module.
         *
         * Returns true if replaced, false if not found. Throws if replacement is null.
         */
        bool replaceModule( const std::string& name, ModulePtr module )
        {
            if (!module)
            {
                throw std::invalid_argument( "Cannot replace with null module." );
            }

            auto it = child_module_map_.find( name );
            if (it == child_module_map_.end())
            {
                return false;
            }

            auto old_module = it->second;
            it->second = module;

            auto vector_it = std::find( child_modules_.begin(), child_modules_.end(), old_module );
            if (vector_it != child_modules_.end())
            {
                *vector_it = module;
            }

            module->setTraining( is_training_ );

            return true;
        }

        /**
         * @brief Set training mode and propagate to children.
         */
        void setTraining( bool is_training ) override
        {
            is_training_ = is_training;

            for (auto& module : child_modules_)
            {
                module->setTraining( is_training );
            }
        }

        /**
         * @brief Query training mode.
         */
        bool isTraining() const override
        {
            return is_training_;
        }

        /**
         * @brief Count parameters across all children.
         */
        size_t parameterCount() const override
        {
            size_t count = 0;

            for (const auto& module : child_modules_)
            {
                count += module->parameterCount();
            }

            return count;
        }

        /**
         * @brief Default save: delegate to child modules.
         */
        void save( ModelArchive& archive ) const override
        {
            for (const auto& module : child_modules_)
            {
                module->save( archive );
            }
        }

        /**
         * @brief Default load: delegate to child modules.
         */
        void load( ModelArchive& archive ) override
        {
            for (const auto& module : child_modules_)
            {
                module->load( archive );
            }
        }

        /**
         * @brief Human-readable description.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            //oss << name_ << " (Composite)";

            oss << " { children: [";

            bool first = true;
            for (const auto& [name, module] : child_module_map_)
            {
                if (!first)
                {
                    oss << ", ";
                }
                first = false;
                oss << name << ": " << module->toString();
            }

            oss << "] }";

            return oss.str();
        }

    private:
        std::vector<ModulePtr> child_modules_;
        std::unordered_map<std::string, ModulePtr> child_module_map_;
        bool is_training_;
    };
}