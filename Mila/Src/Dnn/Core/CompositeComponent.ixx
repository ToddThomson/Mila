/**
 * @file CompositeComponent.ixx
 * @brief Abstract container for managing child components.
 *
 * CompositeComponent provides standardized child management (add, remove, get)
 * and aggregates parameters, gradients, and training state across children.
 * Derived classes define execution semantics (forward/backward) and architecture
 * graph creation (createGraph()).
 */

module;
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <format>

export module Dnn.CompositeComponent;

import Dnn.Component;
import Dnn.ComponentFactory;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief A component that contains and manages child components.
     *
     * CompositeComponent is a device-parameterized abstract container that manages
     * child component lifecycle, aggregates operations (parameters, gradients, training
     * mode), and provides context propagation. Derived types implement execution
     * semantics (forward/backward) and architecture definition (createGraph()).
     *
     * Architecture Philosophy:
     * - Context-independent graph creation: Architecture defined without device knowledge
     * - Three-phase lifecycle: Graph creation ? Context binding ? Shape binding
     * - Automatic context propagation: Base class propagates context to all children
     * - Component-owns-name: Children manage their own identity via getName()
     *
     * Features:
     * - Component-based child registration via addComponent(component)
     * - Automatic context propagation to children via onExecutionContextSet()
     * - Aggregate parameters and gradients across children
     * - Propagate training mode to all children
     * - Recursive serialization of child hierarchy
     * - Template method pattern for build lifecycle
     *
     * ExecutionContext ownership:
     * - CompositeComponent NEVER owns ExecutionContext
     * - All instances share an ExecutionContext owned by parent (Network or test fixture)
     * - Context is propagated to all children automatically
     *
     * Child component construction pattern:
     * 1. Derived class constructs child components in createGraph() (called from constructor)
     * 2. Children are created in shared mode (no ExecutionContext initially)
     * 3. Children call setName() to establish hierarchical identity
     * 4. Children are registered via addComponent(component)
     * 5. When parent gets context, onExecutionContextSet() propagates to all children
     *
     * Design patterns:
     * - Template Method: build() calls onBuilding() to build children with shapes
     * - Composite: manages child lifecycle and aggregates operations
     * - Hook Method: onExecutionContextSet() propagates context to children
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
    class CompositeComponent : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using ComponentPtr = std::shared_ptr<Component<TDeviceType, TPrecision>>;

        /**
         * @brief Construct composite component with name.
         *
         * The composite component is named for identification in hierarchical structures.
         * Derived classes should call createGraph() from their constructor to define
         * the architecture graph (context-independent).
         *
         * All child components added via addComponent() will receive ExecutionContext
         * automatically when the composite receives its context (via onExecutionContextSet).
         *
         * @param name Component name identifier (mandatory)
         *
         * @throws std::invalid_argument if name is not a valid identifier
         */
        explicit CompositeComponent( const std::string& name )
            : ComponentBase( name )
        {}

        virtual ~CompositeComponent() = default;

        CompositeComponent( const CompositeComponent& ) = delete;
        CompositeComponent& operator=( const CompositeComponent& ) = delete;

        CompositeComponent( CompositeComponent&& ) noexcept = default;
        CompositeComponent& operator=( CompositeComponent&& ) noexcept = default;

        // ====================================================================
        // Child Component Management
        // ====================================================================

        /**
         * @brief Add a pre-constructed child component (chainable).
         *
         * Registers a component that was constructed externally (typically by the
         * derived class in its createGraph() method). The component's getName() is
         * used as the lookup key.
         *
         * Components are expected to be created in shared mode (no ExecutionContext).
         * Context will be automatically propagated to all children when this composite
         * receives its context via onExecutionContextSet().
         *
         * Usage pattern in derived class:
         * @code
         * void MLP::createGraph()
         * {
         *     auto fc1 = std::make_shared<LinearType>(config, std::nullopt);
         *     fc1->setName(this->getName() + ".fc1");
         *     this->addComponent(fc1);
         *     // ... more components
         * }
         * @endcode
         *
         * @param component Shared pointer to the constructed component
         *
         * @return Reference to *this for method chaining
         *
         * @throws std::runtime_error if called after build()
         * @throws std::invalid_argument if component is null
         * @throws std::invalid_argument if component name already exists
         */
        CompositeComponent& addComponent( ComponentPtr component )
        {
            if ( this->isBuilt() )
            {
                throw std::runtime_error(
                    "Cannot add components after build() has been called"
                );
            }

            if ( !component )
            {
                throw std::invalid_argument( "Component cannot be null" );
            }

            std::string name = component->getName();

            if ( child_component_map_.find( name ) != child_component_map_.end() )
            {
                throw std::invalid_argument(
                    std::format( "Component name '{}' already exists", name )
                );
            }

            child_component_map_[ name ] = component;
            child_components_.push_back( component );
            
            // setExecutionContext should be done after network building
            /*if ( this->hasExecutionContext() )
            {
                if ( !component->hasExecutionContext() )
                {
                    component->setExecutionContext( this->getExecutionContext() );
                }
            }*/

            return *this;
        }

        /**
         * @brief Retrieve a child component by name.
         *
         * @param name Name of the child component
         * @return Shared pointer to the child component (base type)
         *
         * @throws std::out_of_range if no component with that name exists
         */
        ComponentPtr getComponent( const std::string& name ) const
        {
            auto it = child_component_map_.find( name );

            if ( it == child_component_map_.end() )
            {
                throw std::out_of_range(
                    std::format( "No component named '{}' found", name )
                );
            }

            return it->second;
        }

        /**
         * @brief Check if a named child component exists.
         *
         * @param name Name to query
         * @return true if a child with this name exists
         */
        bool hasComponent( const std::string& name ) const
        {
            return child_component_map_.find( name ) != child_component_map_.end();
        }

        /**
         * @brief Get all child components in insertion order.
         *
         * @return Vector of child component pointers
         */
        const std::vector<ComponentPtr>& getComponents() const
        {
            return child_components_;
        }

        /**
         * @brief Get the named child components map.
         *
         * @return Map of names to child component pointers
         */
        const std::unordered_map<std::string, ComponentPtr>& getNamedComponents() const
        {
            return child_component_map_;
        }

        /**
         * @brief Remove a child component by name.
         *
         * @param name Name of the component to remove
         * @return true if removed, false if not found
         *
         * @throws std::runtime_error if called after build()
         */
        bool removeComponent( const std::string& name )
        {
            if ( this->isBuilt() )
            {
                throw std::runtime_error(
                    "Cannot remove components after build() has been called"
                );
            }

            auto it = child_component_map_.find( name );

            if ( it == child_component_map_.end() )
            {
                return false;
            }

            auto component_ptr = it->second;
            child_component_map_.erase( it );

            auto vector_it = std::find(
                child_components_.begin(),
                child_components_.end(),
                component_ptr
            );

            if ( vector_it != child_components_.end() )
            {
                child_components_.erase( vector_it );
            }

            return true;
        }

        /**
         * @brief Clear all child components.
         *
         * @throws std::runtime_error if called after build()
         */
        void clearComponents()
        {
            if ( this->isBuilt() )
            {
                throw std::runtime_error(
                    "Cannot clear components after build() has been called"
                );
            }

            child_components_.clear();
            child_component_map_.clear();
        }

        /**
         * @brief Get the number of direct children.
         *
         * @return Number of child components
         */
        size_t childCount() const noexcept
        {
            return child_components_.size();
        }

        /**
         * @brief Check if this composite has any children.
         *
         * @return true if at least one child component exists
         */
        bool hasChildren() const noexcept
        {
            return !child_components_.empty();
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
            if ( !this->isBuilt() )
            {
                throw std::runtime_error(
                    "Cannot query parameter count before build() has been called"
                );
            }

            size_t count = 0;

            for ( const auto& component : child_components_ )
            {
                count += component->parameterCount();
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
            if ( !this->isBuilt() )
            {
                throw std::runtime_error(
                    "Cannot get parameters before build()"
                );
            }

            size_t total_count = 0;

            for ( const auto& component : child_components_ )
            {
                total_count += component->parameterCount();
            }

            std::vector<ITensor*> params;
            params.reserve( total_count );

            for ( const auto& component : child_components_ )
            {
                auto child_params = component->getParameters();
                params.insert(
                    params.end(),
                    std::make_move_iterator( child_params.begin() ),
                    std::make_move_iterator( child_params.end() )
                );
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
            if ( !this->isBuilt() )
            {
                throw std::runtime_error(
                    "Cannot get parameter gradients before build()"
                );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error(
                    "Cannot get parameter gradients when not in training mode"
                );
            }

            size_t total_count = 0;

            for ( const auto& component : child_components_ )
            {
                total_count += component->parameterCount();
            }

            std::vector<ITensor*> grads;
            grads.reserve( total_count );

            for ( const auto& component : child_components_ )
            {
                auto child_grads = component->getGradients();
                grads.insert(
                    grads.end(),
                    std::make_move_iterator( child_grads.begin() ),
                    std::make_move_iterator( child_grads.end() )
                );
            }

            return grads;
        }

        // ====================================================================
        // Synchronization
        // ====================================================================

        /**
         * @brief Synchronize all child components.
         *
         * Waits for outstanding device operations on all children.
         */
        void synchronize() override
        {
            for ( auto& component : child_components_ )
            {
                component->synchronize();
            }
        }

        // ====================================================================
        // Device Information
        // ====================================================================

        /**
         * @brief Get the compute device for this composite.
         *
         * Returns the device from the shared execution context.
         *
         * @return DeviceId for this composite and its children
         */
        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        // ====================================================================
        // Component Information
        // ====================================================================

        /**
         * @brief Generate a human-readable description.
         *
         * @return String representation showing children
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << this->getName() << " { children: [";

            bool first = true;

            for ( const auto& [name, component] : child_component_map_ )
            {
                if ( !first )
                {
                    oss << ", ";
                }

                first = false;
                oss << name << ": " << component->getName();
            }

            oss << "] }";

            return oss.str();
        }

    protected:

        /**
         * @brief Retrieve a typed child component by name (for derived class use).
         *
         * Helper method for derived composites (like MLP) that need to cache typed
         * pointers to children in their onBuilding() hook. Performs dynamic_pointer_cast
         * and validates the cast succeeded.
         *
         * Intended usage pattern:
         * @code
         * void MLP::onBuilding(const shape_t& input_shape) override
         * {
         *     // Cache typed pointers once during build
         *     fc1_ = this->getComponentAs<LinearType>(this->getName() + ".fc1");
         *     activation_ = this->getComponentAs<GeluType>(this->getName() + ".act");
         *     fc2_ = this->getComponentAs<LinearType>(this->getName() + ".fc2");
         *
         *     // Build children with computed shapes
         *     fc1_->build(input_shape);
         *     // ...
         * }
         * @endcode
         *
         * @tparam TComponent Expected component type
         * @param name Name of the component to retrieve
         * @return Shared pointer to component with correct type
         *
         * @throws std::out_of_range if component name not found
         * @throws std::runtime_error if dynamic cast fails (type mismatch)
         */
        template<typename TComponent>
        std::shared_ptr<TComponent> getComponentAs( const std::string& name ) const
        {
            auto base = getComponent( name );
            auto typed = std::dynamic_pointer_cast<TComponent>( base );

            if ( !typed )
            {
                throw std::runtime_error(
                    std::format(
                        "Component '{}' cannot be cast to requested type", name )
                );
            }

            return typed;
        }

        /**
         * @brief Hook invoked after ExecutionContext is set.
         *
         * Propagates the execution context to all child components that don't
         * already have one. This enables the pattern where composites define
         * their architecture graph in the constructor (context-independent)
         * and context is bound later when available.
         *
         * Called by Component::setExecutionContext() after the context is registered.
         * Automatically invoked for both standalone mode (component creates own context)
         * and shared mode (parent provides context).
         *
         * Override this in derived classes if additional context-dependent initialization
         * is required beyond context propagation to children.
         */
        void onExecutionContextSet() override
        {
            for ( auto& component : child_components_ )
            {
                if ( !component->hasExecutionContext() )
                {
                    component->setExecutionContext( this->getExecutionContext() );
                }
            }
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagates the new mode to all child components. The hook runs with
         * the Component's training mutex held; it MUST NOT call setTraining().
         *
         * @param is_training New training mode (true = training, false = eval)
         */
        void onTrainingChanging( bool is_training ) override
        {
            for ( auto& component : child_components_ )
            {
                component->setTraining( is_training );
            }
        }

        /**
         * @brief Save all child components recursively.
         *
         * Follows the component serialization contract:
         * - Writes type, version, and configuration metadata
         * - Recursively saves all children with scoped namespaces
         * - Each child's save_() handles its own state
         *
         * @param archive Archive to write to
         * @param mode What to save (Checkpoint, WeightsOnly, Architecture)
         */
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error(
                    "Cannot save unbuilt CompositeComponent"
                );
            }

            archive.addMetadata( "type", this->getName() );
            archive.addMetadata( "version", "1" );
            archive.addMetadata( "child_count", std::to_string( child_components_.size() ) );

            std::ostringstream names_stream;
            bool first = true;

            for ( const auto& [name, _] : child_component_map_ )
            {
                if ( !first )
                {
                    names_stream << ",";
                }

                names_stream << name;
                first = false;
            }

            archive.addMetadata( "child_names", names_stream.str() );

            for ( const auto& [name, component] : child_component_map_ )
            {
                component->save_( archive, mode );
            }
        }

    private:

        /**
         * @brief Child components in insertion order.
         *
         * This vector holds shared ownership of the composite's direct children and
         * preserves the order in which children were added. The insertion order is
         * used for build sequencing, ordered iteration, and serialization ordering.
         *
         * Lifecycle invariants:
         * - Children are constructed in createGraph() (called from derived constructor)
         * - Children are registered via addComponent() before context is available
         * - Context is propagated to children via onExecutionContextSet() hook
         * - Children are built by parent's onBuilding() via template method pattern
         *
         * Threading: access is not internally synchronized. Mutations and lifecycle
         * operations must be externally serialized.
         */
        std::vector<ComponentPtr> child_components_;

        /**
         * @brief Lookup map from child name to component pointer.
         *
         * Provides O(1) name-based lookup for children. Keys are component names
         * (obtained via component->getName()) and must be unique within the composite.
         *
         * Used for:
         * - getComponent()/hasComponent() queries
         * - getComponentAs<T>() typed retrieval in derived classes
         * - Diagnostics and debugging
         *
         * Note: insertion order is preserved by `child_components_`; this unordered_map
         * does not guarantee ordering but provides fast lookup.
         */
        std::unordered_map<std::string, ComponentPtr> child_component_map_;

        friend class ComponentFactory;
    };
}