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
#include <algorithm>

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
            : is_training_( false ), is_built_(false)
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
            if (is_built_)
            {
                throw std::runtime_error(
                    "Cannot add modules after build() has been called"
                );
            }

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
            if (is_built_)
            {
                throw std::runtime_error(
                    "Cannot remove modules after build() has been called"
                );
            }

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
            if (is_built_)
            {
                throw std::runtime_error(
                    "Cannot replace modules after build() has been called"
                );
            }

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

        // ====================================================================
        // Build Lifecycle
		// ====================================================================

        /**
            * @brief Check if this module and all children are built.
            */
        bool isBuilt() const override
        {
            if (!is_built_)
            {
                return false;
            }

            // All children must also be built
            return std::all_of( child_modules_.begin(), child_modules_.end(),
                []( const auto& module ) {
                    return module->isBuilt();
                } );
        }

        /**
         * @brief Build this module and all children with given input shape.
         *
         * Derived classes should override buildImpl() to provide shape inference
         * logic. This method handles recursion and idempotency.
         *
         * @param input_shape Expected input tensor shape
         */
        void build( const shape_t& input_shape ) override
        {
            if (is_built_)
            {
                return;  // Idempotent
            }

            // Let derived class handle shape propagation to children
            buildImpl( input_shape );

            is_built_ = true;
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
            if (!isBuilt())
            {
                throw std::runtime_error(
                    "Cannot query parameter count before build() is called"
                );
            }

            size_t count = 0;

            for (const auto& module : child_modules_)
            {
                count += module->parameterCount();
            }

            return count;
        }

        /**
         * @brief Collect all parameters from children.
         *
         * @throws std::runtime_error if not built
         */
        /*std::vector<Tensor*> parameters() override
        {
            if (!isBuilt())
            {
                throw std::runtime_error(
                    "Cannot get parameters before build() is called"
                );
            }

            std::vector<Tensor*> params;
            for (auto& module : child_modules_)
            {
                auto child_params = module->parameters();
                params.insert( params.end(), child_params.begin(), child_params.end() );
            }

            return params;
        }*/

		// ====================================================================
		// Serialization
		// ====================================================================

        /**
         * @brief Save all child modules.
         *
         * @throws std::runtime_error if not built
         */
        void save( ModelArchive& archive ) const override
        {
            //if (!isBuilt())
            //{
            //    throw std::runtime_error(
            //        "Cannot save before build() is called"
            //    );
            //}

            //// Save build state
            //archive.write( "is_built", is_built_ );
            //archive.write( "num_children", child_modules_.size() );

            //// Save each child with its name
            //for (const auto& [name, module] : child_module_map_)
            //{
            //    archive.beginGroup( name );
            //    module->save( archive );
            //    archive.endGroup();
            //}
        }

        /**
         * @brief Load all child modules and mark as built.
         */
        void load( ModelArchive& archive ) override
        {
            //bool was_built;
            //archive.read( "is_built", was_built );

            //if (!was_built)
            //{
            //    throw std::runtime_error(
            //        "Cannot load model that was not built before saving"
            //    );
            //}

            //size_t num_children;
            //archive.read( "num_children", num_children );

            //if (num_children != child_modules_.size())
            //{
            //    throw std::runtime_error(
            //        "Model structure mismatch: expected " +
            //        std::to_string( num_children ) + " children, found " +
            //        std::to_string( child_modules_.size() )
            //    );
            //}

            //// Load each child
            //for (const auto& [name, module] : child_module_map_)
            //{
            //    archive.beginGroup( name );
            //    module->load( archive );
            //    archive.endGroup();
            //}

            //is_built_ = true;  // Now built after loading
        }

		// ====================================================================
		// Description
		// ====================================================================

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

    protected:

        /**
         * @brief Override in derived classes to implement shape propagation logic.
         *
         * Default implementation calls build() on all children with the same input shape.
         * Sequential modules would propagate shapes through the chain.
         *
         * @param input_shape Shape of input to this composite module
         */
        virtual void buildImpl( const shape_t& input_shape )
        {
            // Default: build all children with same input shape
            for (auto& module : child_modules_)
            {
                module->build( input_shape );
            }
        }


        std::vector<ModulePtr> child_modules_;
        std::unordered_map<std::string, ModulePtr> child_module_map_;
        bool is_training_;
		bool is_built_;
    };
}