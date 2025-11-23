/**
 * @file CompositeModule.ixx
 * @brief Abstract container for managing child modules.
 *
 * CompositeModule provides standardized child management (add, remove, get)
 * and aggregates parameters, gradients, and training state across children.
 * Derived classes define execution semantics (forward/backward/build).
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
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief A module that contains and manages child modules.
     *
     * CompositeModule is a device-parameterized abstract container. It does not
     * implement the computational interface (forward/backward) so derived types
     * must provide execution semantics while benefiting from standardized child
     * management.
     *
     * Features:
     * - Add, remove, replace, and query child modules
     * - Aggregate parameters and gradients across children
     * - Propagate training mode to all children
     * - Recursive serialization of child hierarchy
     * - Build state validation
     */
    export template<DeviceType TDeviceType>
        class CompositeModule : public Module<TDeviceType>
    {
    public:
        using ModuleBase = Module<TDeviceType>;
        using ModulePtr = std::shared_ptr<Module<TDeviceType>>;

        /**
         * @brief Construct an empty composite module.
         */
        explicit CompositeModule() noexcept
            : is_built_( false )
        {
        }

        virtual ~CompositeModule() = default;

        // Delete copy operations (manages shared_ptr children)
        CompositeModule( const CompositeModule& ) = delete;
        CompositeModule& operator=( const CompositeModule& ) = delete;

        // Enable move operations
        CompositeModule( CompositeModule&& ) noexcept = default;
        CompositeModule& operator=( CompositeModule&& ) noexcept = default;

        // ====================================================================
        // Child Module Management
        // ====================================================================

        /**
         * @brief Add a named child module.
         *
         * The child inherits the parent's current training mode.
         *
         * @param name Unique identifier for this child (cannot be empty)
         * @param module Child module to add (cannot be null)
         * @return Reference to this composite for method chaining
         *
         * @throws std::runtime_error if called after build()
         * @throws std::invalid_argument if name is empty, module is null, or name already exists
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
                throw std::invalid_argument( "Module name cannot be empty" );
            }

            if (!module)
            {
                throw std::invalid_argument( "Cannot add null module" );
            }

            if (child_module_map_.find( name ) != child_module_map_.end())
            {
                throw std::invalid_argument( "Module name '" + name + "' already exists" );
            }

            child_module_map_[name] = module;
            child_modules_.push_back( module );

            // Ensure newly-added child inherits parent's current training mode.
            // This prevents inconsistent state where parent is in training mode
            // but newly-added child defaults to eval mode.
            module->setTraining( this->isTraining() );

            return *this;
        }

        /**
         * @brief Add an unnamed child module with auto-generated name.
         *
         * Name format: "module_N" where N is the current child count.
         *
         * @param module Child module to add (cannot be null)
         * @return Reference to this composite for method chaining
         *
         * @throws std::runtime_error if called after build()
         * @throws std::invalid_argument if module is null
         */
        CompositeModule& addModule( ModulePtr module )
        {
            if (!module)
            {
                throw std::invalid_argument( "Cannot add null module" );
            }

            std::string auto_name = "module_" + std::to_string( child_modules_.size() );

            return addModule( auto_name, module );
        }

        /**
         * @brief Retrieve a child module by name.
         *
         * @param name Name of the child module
         * @return Shared pointer to the child module
         *
         * @throws std::out_of_range if no module with that name exists
         */
        ModulePtr getModule( const std::string& name ) const
        {
            auto it = child_module_map_.find( name );
            if (it == child_module_map_.end())
            {
                throw std::out_of_range( "No module named '" + name + "' found" );
            }

            return it->second;
        }

        /**
         * @brief Check if a named child module exists.
         *
         * @param name Name to query
         * @return true if a child with this name exists
         */
        bool hasModule( const std::string& name ) const
        {
            return child_module_map_.find( name ) != child_module_map_.end();
        }

        /**
         * @brief Get all child modules in insertion order.
         *
         * @return Vector of child module pointers
         */
        const std::vector<ModulePtr>& getModules() const
        {
            return child_modules_;
        }

        /**
         * @brief Get the named child modules map.
         *
         * @return Map of names to child module pointers
         */
        const std::unordered_map<std::string, ModulePtr>& getNamedModules() const
        {
            return child_module_map_;
        }

        /**
         * @brief Remove a child module by name.
         *
         * @param name Name of the module to remove
         * @return true if removed, false if not found
         *
         * @throws std::runtime_error if called after build()
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
         * The replacement inherits the parent's training mode.
         *
         * @param name Name of the module to replace
         * @param module Replacement module (cannot be null)
         * @return true if replaced, false if not found
         *
         * @throws std::runtime_error if called after build()
         * @throws std::invalid_argument if replacement module is null
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
                throw std::invalid_argument( "Cannot replace with null module" );
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

            module->setTraining( this->isTraining() );

            return true;
        }

        /**
         * @brief Clear all child modules.
         *
         * @throws std::runtime_error if called after build()
         */
        void clearModules()
        {
            if (is_built_)
            {
                throw std::runtime_error(
                    "Cannot clear modules after build() has been called"
                );
            }

            child_modules_.clear();
            child_module_map_.clear();
        }

        /**
         * @brief Get the number of direct children.
         *
         * @return Number of child modules
         */
        size_t childCount() const noexcept
        {
            return child_modules_.size();
        }

        /**
         * @brief Check if this composite has any children.
         *
         * @return true if at least one child module exists
         */
        bool hasChildren() const noexcept
        {
            return !child_modules_.empty();
        }

        // ====================================================================
        // Build Lifecycle
        // ====================================================================

        /**
         * @brief Check if this module and all children are built.
         *
         * @return true if this module and all children are successfully built
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
         * @brief Build this module with the given input shape.
         *
         * Derived classes must override to implement their specific shape
         * propagation and child building logic. After building children,
         * call validateChildrenBuilt() to ensure all children were successfully
         * built, then set is_built_ = true.
         *
         * @param input_shape Expected input tensor shape
         */
        void build( const shape_t& input_shape ) override
        {
            if (is_built_)
            {
                return;
            }

            // Ensure children observe parent's training mode and build them.
            for (auto& module : child_modules_)
            {
                module->setTraining( this->isTraining() );
                module->build( input_shape );
            }

            // Verify children succeeded and mark composite built.
            validateChildrenBuilt();

            is_built_ = true;
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        /**
         * @brief Count parameters across all children.
         *
         * @return Total number of trainable parameters
         *
         * @throws std::runtime_error if called before build()
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
         * @brief Get all parameters from all children.
         *
         * @return Vector of non-owning pointers to parameter tensors
         *
         * @throws std::runtime_error if called before build()
         */
        std::vector<ITensor*> getParameters() const override
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Cannot get parameters before build()" );
            }

            // Pre-calculate total size to avoid reallocations
            size_t total_count = 0;
            for (const auto& module : child_modules_)
            {
                total_count += module->parameterCount();
            }

            std::vector<ITensor*> params;
            params.reserve( total_count );

            for (const auto& module : child_modules_)
            {
                auto child_params = module->getParameters();
                params.insert( params.end(),
                    std::make_move_iterator( child_params.begin() ),
                    std::make_move_iterator( child_params.end() ) );
            }

            return params;
        }

        /**
         * @brief Get all parameter gradients from all children.
         *
         * @return Vector of non-owning pointers to gradient tensors
         *
         * @throws std::runtime_error if called before build() or not in training mode
         */
        std::vector<ITensor*> getGradients() const override
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Cannot get parameter gradients before build()" );
            }

            if (!this->isTraining())
            {
                throw std::runtime_error( "Cannot get parameter gradients when not in training mode" );
            }

            // Pre-calculate total size to avoid reallocations
            size_t total_count = 0;
            for (const auto& module : child_modules_)
            {
                total_count += module->parameterCount();
            }

            std::vector<ITensor*> grads;
            grads.reserve( total_count );

            for (const auto& module : child_modules_)
            {
                auto child_grads = module->getGradients();
                grads.insert( grads.end(),
                    std::make_move_iterator( child_grads.begin() ),
                    std::make_move_iterator( child_grads.end() ) );
            }

            return grads;
        }

        // ====================================================================
        // Synchronization
        // ====================================================================

        /**
         * @brief Synchronize all child modules.
         *
         * Waits for outstanding device operations on all children.
         */
        void synchronize() override
        {
            for (auto& module : child_modules_)
            {
                module->synchronize();
            }
        }

        // ====================================================================
        // Device Information
        // ====================================================================

        /**
         * @brief Get the compute device for this composite.
         *
         * Returns the device of the first child. Assumes all children share
         * the same device (should be validated during build).
         *
         * @return Shared pointer to compute device, or nullptr if no children
         */
        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            if (child_modules_.empty())
            {
                return nullptr;
            }

            return child_modules_[0]->getDevice();
        }

        // ====================================================================
        // Module Information
        // ====================================================================

        /**
         * @brief Get the name of this composite module.
         *
         * Derived classes should override to provide specific names.
         *
         * @return Module name for diagnostics and serialization
         */
        std::string getName() const override
        {
            return "CompositeModule";
        }

        /**
         * @brief Generate a human-readable description.
         *
         * @return String representation showing children
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << getName() << " { children: [";

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

        // ====================================================================
        // Serialization (Internal Protocol)
        // ====================================================================

        /**
         * @internal
         * @brief Save all child modules recursively.
         *
         * Follows the Module serialization contract:
         * - Writes type, version, and configuration metadata
         * - Recursively saves all children with scoped namespaces
         * - Each child's save_() handles its own state
         *
         * @param archive Archive to write to
         * @param mode What to save (Checkpoint, WeightsOnly, Architecture)
         */
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Cannot save unbuilt CompositeModule" );
            }

            // Metadata per Module contract
            archive.addMetadata( "type", getName() );
            archive.addMetadata( "version", "1" );

            // Configuration needed for reconstruction
            archive.addMetadata( "child_count", std::to_string( child_modules_.size() ) );

            // Save child names in insertion order for reconstruction
            std::ostringstream names_stream;
            bool first = true;
            for (const auto& [name, _] : child_module_map_)
            {
                if (!first) names_stream << ",";
                names_stream << name;
                first = false;
            }
            archive.addMetadata( "child_names", names_stream.str() );

            // Recursively save each child with scoped namespace
            for (const auto& [name, module] : child_module_map_)
            {
                //archive.pushScope( name );
                module->save_( archive, mode );
                //archive.popScope();
            }
        }

        // ====================================================================
        // Helpers for Derived Classes
        // ====================================================================

        /**
         * @brief Validate that all children were successfully built.
         *
         * Call this at the end of your build() override to ensure all
         * children are in a valid built state.
         *
         * @throws std::runtime_error if any child is not built
         */
        void validateChildrenBuilt() const
        {
            for (const auto& module : child_modules_)
            {
                if (!module->isBuilt())
                {
                    throw std::runtime_error(
                        "Child module '" + getModuleName( module.get() ) +
                        "' failed to build"
                    );
                }
            }
        }

        /**
         * @brief Get the name of a child module for diagnostics.
         *
         * @param module Pointer to the child module
         * @return Name of the module, or "unknown" if not found
         */
        std::string getModuleName( const Module<TDeviceType>* module ) const
        {
            for (const auto& [name, mod] : child_module_map_)
            {
                if (mod.get() == module)
                {
                    return name;
                }
            }
            return "unknown";
        }

        /**
         * @brief Mark this module as built without validation.
         *
         * Used exclusively by ModuleFactory when reconstructing from archive.
         * After reconstruction, the module structure is complete and should
         * be treated as built.
         *
         * Warning: This bypasses normal build validation. Only use during
         * deserialization after fully reconstructing the module tree.
         */
        void markBuilt_()
        {
            is_built_ = true;
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagates the new mode to all child modules. The hook runs with
         * the Module's training mutex held; it MUST NOT call setTraining().
         *
         * @param is_training New training mode (true = training, false = eval)
         */
        void onTrainingChanging( bool is_training ) override
        {
            for (auto& module : child_modules_)
            {
                module->setTraining( is_training );
            }
        }

        std::vector<ModulePtr> child_modules_;
        std::unordered_map<std::string, ModulePtr> child_module_map_;
        bool is_built_;

        // Friend declarations for factory access
        friend class ModuleFactory;
    };
}