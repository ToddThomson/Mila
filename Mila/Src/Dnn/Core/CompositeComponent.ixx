/**
 * @file CompositeComponent.ixx
 * @brief Abstract container for managing child modules.
 *
 * CompositeComponent provides standardized child management (add, remove, get)
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
#include <vector>
#include <iterator>

export module Dnn.CompositeComponent;

import Dnn.Component;
import Dnn.TensorDataType;
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
     * CompositeComponent is a device-parameterized abstract container. It does not
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
     *
     * Design note:
     * - The build lifecycle is centralized in this base class (template method).
     *   Derived classes supply architecture-specific shape propagation by
     *   implementing `onBuilding(const shape_t&)` and using the provided protected
     *   helpers to build children with the correct shapes.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
    class CompositeComponent : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using ComponentPtr = std::shared_ptr<Component<TDeviceType, TPrecision>>;

        /**
         * @brief Construct an empty composite module.
         */
        explicit CompositeComponent() noexcept
            : is_built_( false )
        {
        }

        virtual ~CompositeComponent() = default;

        // Delete copy operations (manages shared_ptr children)
        CompositeComponent( const CompositeComponent& ) = delete;
        CompositeComponent& operator=( const CompositeComponent& ) = delete;

        // Enable move operations
        CompositeComponent( CompositeComponent&& ) noexcept = default;
        CompositeComponent& operator=( CompositeComponent&& ) noexcept = default;

        // ====================================================================
        // Child Component Management
        // ====================================================================

        /**
         * @brief Add a child module that must already carry a valid name.
         *
         * Caller-provided module MUST expose a non-empty, valid name via
         * `getName()`. This enforces a project-wide invariant: all components
         * must have identifiers. The composite will use the child's name as the
         * registration key and will reject unnamed children.
         *
         * @param module Child module to add (cannot be null, must have non-empty name)
         * @return Reference to this composite for method chaining
         *
         * @throws std::runtime_error if called after build()
         * @throws std::invalid_argument if module is null, has empty name, or name already exists
         */
        CompositeComponent& addComponent( ComponentPtr component )
        {
            if (is_built_)
            {
                throw std::runtime_error(
                    "Cannot add Components after build() has been called"
                );
            }

            if (!component)
            {
                throw std::invalid_argument( "Cannot add null Component" );
            }

            // Require the child to already provide a valid name.
            const std::string child_name = component->getName();
            if (child_name.empty())
            {
                throw std::invalid_argument(
                    "Child Component must provide a non-empty name via getName()" );
            }

            if (child_component_map_.find( child_name ) != child_component_map_.end())
            {
                throw std::invalid_argument( "Component name '" + child_name + "' already exists" );
            }

            child_component_map_[child_name] = component;
            child_components_.push_back( component );

            return *this;
        }

        /**
         * @brief Retrieve a child module by name.
         *
         * @param name Name of the child module
         * @return Shared pointer to the child module
         *
         * @throws std::out_of_range if no module with that name exists
         */
        ComponentPtr getComponent( const std::string& name ) const
        {
            auto it = child_component_map_.find( name );
            if (it == child_component_map_.end())
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
        bool hasComponent( const std::string& name ) const
        {
            return child_component_map_.find( name ) != child_component_map_.end();
        }

        /**
         * @brief Get all child modules in insertion order.
         *
         * @return Vector of child module pointers
         */
        const std::vector<ComponentPtr>& getComponents() const
        {
            return child_components_;
        }

        /**
         * @brief Get the named child modules map.
         *
         * @return Map of names to child module pointers
         */
        const std::unordered_map<std::string, ComponentPtr>& getNamedComponents() const
        {
            return child_component_map_;
        }

        /**
         * @brief Remove a child module by name.
         *
         * @param name Name of the module to remove
         * @return true if removed, false if not found
         *
         * @throws std::runtime_error if called after build()
         */
        bool removeComponent( const std::string& name )
        {
            if (is_built_)
            {
                throw std::runtime_error(
                    "Cannot remove modules after build() has been called"
                );
            }

            auto it = child_component_map_.find( name );
            if (it == child_component_map_.end())
            {
                return false;
            }

            auto module_ptr = it->second;
            child_component_map_.erase( it );

            auto vector_it = std::find( child_components_.begin(), child_components_.end(), module_ptr );
            if (vector_it != child_components_.end())
            {
                child_components_.erase( vector_it );
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
        bool replaceComponent( const std::string& name, ComponentPtr module )
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

            auto it = child_component_map_.find( name );
            if (it == child_component_map_.end())
            {
                return false;
            }

            auto old_module = it->second;
            it->second = module;

            auto vector_it = std::find( child_components_.begin(), child_components_.end(), old_module );
            if (vector_it != child_components_.end())
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
        void clearComponents()
        {
            if (is_built_)
            {
                throw std::runtime_error(
                    "Cannot clear modules after build() has been called"
                );
            }

            child_components_.clear();
            child_component_map_.clear();
        }

        /**
         * @brief Get the number of direct children.
         *
         * @return Number of child modules
         */
        size_t childCount() const noexcept
        {
            return child_components_.size();
        }

        /**
         * @brief Check if this composite has any children.
         *
         * @return true if at least one child module exists
         */
        bool hasChildren() const noexcept
        {
            return !child_components_.empty();
        }

        // ====================================================================
        // Build Lifecycle
        // ====================================================================

        /**
         * @brief Check if this Component and all children are built.
         *
         * @return true if this Component and all children are successfully built
         */
        virtual bool isBuilt() const override
        {
            if (!is_built_)
            {
                return false;
            }

            // All children must also be built
            return std::all_of( child_components_.begin(), child_components_.end(),
                []( const auto& component ) {
                    return component->isBuilt();
                } );
        }

        /**
         * @brief Final build entry point (template method).
         *
         * Lifecycle responsibilities implemented here:
         * - guard against repeated builds
         * - invoke derived-class architecture-specific build hook `onBuilding`
         * - validate that children report built state
         * - mark composite as built
         *
         * Derived classes MUST implement `onBuilding(const shape_t&)` and use the
         * protected helpers to build their children with correct shapes.
         *
         * @param input_shape Expected input tensor shape
         */
        virtual void build( const shape_t& input_shape ) final override
        {
            if (is_built_)
            {
                return;
            }

            // Delegate architecture-specific shape propagation to derived class.
            onBuilding( input_shape );

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
                    "Cannot query parameter count before build() has been called"
                );
            }

            size_t count = 0;
            for (const auto& module : child_components_)
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
            for (const auto& module : child_components_)
            {
                total_count += module->parameterCount();
            }

            std::vector<ITensor*> params;
            params.reserve( total_count );

            for (const auto& component : child_components_)
            {
                auto child_params = component->getParameters();
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
            for (const auto& module : child_components_)
            {
                total_count += module->parameterCount();
            }

            std::vector<ITensor*> grads;
            grads.reserve( total_count );

            for (const auto& module : child_components_)
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
            for (auto& module : child_components_)
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
            if (child_components_.empty())
            {
                return nullptr;
            }

            return child_components_[0]->getDevice();
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
            return "CompositeComponent";
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
            for (const auto& [name, module] : child_component_map_)
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
                throw std::runtime_error( "Cannot save unbuilt CompositeComponent" );
            }

            // Metadata per Module contract
            archive.addMetadata( "type", getName() );
            archive.addMetadata( "version", "1" );

            // Configuration needed for reconstruction
            archive.addMetadata( "child_count", std::to_string( child_components_.size() ) );

            // Save child names in insertion order for reconstruction
            std::ostringstream names_stream;
            bool first = true;
            for (const auto& [name, _] : child_component_map_)
            {
                if (!first) names_stream << ",";
                names_stream << name;
                first = false;
            }
            archive.addMetadata( "child_names", names_stream.str() );

            // Recursively save each child with scoped namespace
            for (const auto& [name, module] : child_component_map_)
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
         * @brief Derived-class hook for architecture-specific build logic.
         *
         * Called from the final `build()` implementation. Derived classes MUST
         * implement this method to compute per-child shapes and invoke the
         * protected build helpers to construct child modules.
         *
         * Preconditions:
         * - Called only when the composite is not yet built.
         *
         * Postconditions:
         * - Children required by the composite should have been built when this
         *   method returns.
         */
        virtual void onBuilding( const shape_t& input_shape ) = 0;

        /**
         * @brief Build a specific child module with the provided shape.
         *
         * Sets the child's training mode to match the parent, then calls
         * `child->build(shape)`. Throws if `child` is null.
         *
         * This helper centralizes common preconditions and ensures children are
         * built consistently.
         */
        void buildChild( ComponentPtr child, const shape_t& shape )
        {
            if (!child)
            {
                throw std::invalid_argument( "buildChild: null child" );
            }

            child->setTraining( this->isTraining() );
            child->build( shape );
        }

        /**
         * @brief Build a named child (lookup by name) with the provided shape.
         *
         * Throws if the name is not found.
         */
        void buildChild( const std::string& name, const shape_t& shape )
        {
            auto it = child_component_map_.find( name );
            if (it == child_component_map_.end())
            {
                throw std::out_of_range( "buildChild: no child named '" + name + "'" );
            }

            buildChild( it->second, shape );
        }

        /**
         * @brief Convenience: build all direct children with the same input shape.
         *
         * Useful for top-level containers (Network) where children accept the same
         * input shape. This calls `buildChild(child, shape)` for each direct child
         * in insertion order.
         */
        void buildChildrenWithSameShape( const shape_t& shape )
        {
            for (auto& c : child_components_)
            {
                buildChild( c, shape );
            }
        }

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
            for (const auto& component : child_components_)
            {
                if (!component->isBuilt())
                {
                    throw std::runtime_error(
                        "Child Component '" + getComponentName( component.get() ) +
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
        std::string getComponentName( const Component<TDeviceType, TPrecision>* module ) const
        {
            for (const auto& [name, mod] : child_component_map_)
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
            for (auto& module : child_components_)
            {
                module->setTraining( is_training );
            }
        }

        /**
         * @brief Child modules in insertion order.
         *
         * This vector holds shared ownership of the composite's direct children and
         * preserves the order in which children were added. The insertion order is
         * used for build sequencing, ordered iteration, and determining serialization
         * ordering. Do not mutate this container directly outside of the Composite
         * API (use addComponent / replaceComponent / removeComponent) — mutations
         * must occur prior to calling build().
         *
         * Threading: access is not internally synchronized. Mutations and lifecycle
         * operations must be externally serialized.
         */
        std::vector<ComponentPtr> child_components_;

        /**
         * @brief Lookup map from child name to module pointer.
         *
         * Provides O(1) name-based lookup for children. Keys are expected to match
         * the child's stable identifier returned by `getName()` and must be unique.
         * This map is used for diagnostics, getComponent()/hasComponent(), and to
         * pair with `child_components_` when deterministic ordering is required.
         *
         * Note: insertion order is preserved by `child_components_`; the unordered_map
         * does not guarantee ordering.
         */
        std::unordered_map<std::string, ComponentPtr> child_component_map_;

        /**
         * @brief Indicates whether this composite and its children have been built.
         *
         * Set to true after a successful call to `build()` (or by `markBuilt_()` when
         * reconstructing from an archive). Public APIs rely on this flag to guard
         * operations that require the built state (for example parameter queries).
         *
         * Lifecycle: modifications follow the build/deserialization contract; this
         * flag is not internally synchronized and callers must ensure serialized
         * access to build-related operations.
         */
        bool is_built_;

        // Friend declarations for factory access
        friend class ComponentFactory;
    };
}