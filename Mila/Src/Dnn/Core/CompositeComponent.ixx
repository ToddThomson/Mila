/**
 * @file CompositeComponent.ixx
 * @brief Abstract container for managing child components.
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
#include <format>

export module Dnn.CompositeComponent;

import Dnn.Component;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief A component that contains and manages child components.
     *
     * CompositeComponent is a device-parameterized abstract container. It does not
     * implement the computational interface (forward/backward) so derived types
     * must provide execution semantics while benefiting from standardized child
     * management.
     *
     * Features:
     * - Fluent API for declarative child component construction
     * - Factory method pattern enforces ExecutionContext sharing
     * - Aggregate parameters and gradients across children
     * - Propagate training mode to all children
     * - Recursive serialization of child hierarchy
     * - Template method pattern for build lifecycle
     * - Shared ExecutionContext management across child hierarchy
     *
     * ExecutionContext ownership:
     * - CompositeComponent owns a shared ExecutionContext that is propagated to all children
     * - Children constructed via the composite share this context (efficient resource pooling)
     * - Public constructor accepts DeviceId and creates owned context
     * - Protected constructor accepts IExecutionContext* for factory/deserialization patterns
     *
     * Child component construction:
     * - Use fluent `addComponent<T>(name, config)` to construct children with shared context
     * - Factory method returns *this for chainable calls
     * - Components must be fully configured via constructor (configuration-driven design)
     * - Children are built by parent's `onBuilding()` hook, not by external callers
     *
     * Design pattern:
     * - Template Method: `build()` calls `onBuilding()` to build children with correct shapes
     * - Factory Method: `addComponent<T>()` constructs children using protected constructor
     * - Composite: manages child lifecycle and aggregates operations
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        class CompositeComponent : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using ComponentPtr = std::shared_ptr<Component<TDeviceType, TPrecision>>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;

        /**
         * @brief Construct composite component with owned ExecutionContext.
         *
         * Creates and owns an ExecutionContext for the specified device.
         * This context will be shared with all child components added to the composite.
         *
         * @param device_id DeviceId identifying the device for this composite and its children.
         *
         * @throws std::invalid_argument if device_id.type does not match TDeviceType.
         * @throws std::runtime_error if ExecutionContext creation fails.
         */
        explicit CompositeComponent( DeviceId device_id )
            : owned_context_( createOwnedContext( device_id ) )
            , exec_context_( owned_context_.get() )
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
         * @brief Add a child component using factory method pattern (chainable).
         *
         * Constructs a child component with the shared ExecutionContext and adds it
         * to the composite. Returns *this for method chaining, enabling fluent API
         * for declarative network construction.
         *
         * The component is constructed using its protected constructor that accepts
         * IExecutionContext*, ensuring all children share the parent's execution context.
         * Component must be fully configured via constructor arguments (config objects).
         *
         * Usage:
         * @code
         * network
         *     .addComponent<Linear>("encoder", encoder_config)
         *     .addComponent<Gelu>("activation", gelu_config)
         *     .addComponent<Linear>("decoder", decoder_config);
         * @endcode
         *
         * @tparam TComponent Component type to construct (must have protected constructor)
         * @tparam Args Constructor argument types (forwarded to component constructor)
         *
         * @param name Unique name for the child component (used for lookup and serialization)
         * @param args Arguments forwarded to component's protected constructor
         *
         * @return Reference to *this for method chaining
         *
         * @throws std::runtime_error if called after build()
         * @throws std::invalid_argument if name already exists
         * @throws Any exception from component constructor
         */
        template<typename TComponent, typename... Args>
        CompositeComponent& addComponent( const std::string& name, Args&&... args )
        {
            if ( this->isBuilt() )
            {
                throw std::runtime_error(
                    "Cannot add components after build() has been called"
                );
            }

            auto component = std::make_shared<TComponent>(
                getExecutionContext(),
                std::forward<Args>( args )...
            );

            if ( child_component_map_.find( name ) != child_component_map_.end() )
            {
                throw std::invalid_argument(
                    "Component name '" + name + "' already exists"
                );
            }

            child_component_map_[ name ] = component;
            child_components_.push_back( component );

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
                    "No component named '" + name + "' found"
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
         * Returns the device from the owned or shared execution context.
         *
         * @return DeviceId for this composite and its children
         */
        DeviceId getDeviceId() const override
        {
            return exec_context_->getDeviceId();
        }

        // ====================================================================
        // Component Information
        // ====================================================================

        /**
         * @brief Get the name of this composite component.
         *
         * Derived classes should override to provide specific names.
         *
         * @return Component name for diagnostics and serialization
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

            for ( const auto& [name, component] : child_component_map_ )
            {
                if ( !first )
                {
                    oss << ", ";
                }

                first = false;
                oss << name << ": " << component->toString();
            }

            oss << "] }";

            return oss.str();
        }

    protected:

        /**
         * @brief Construct composite as child component sharing parent's ExecutionContext.
         *
         * Used by Network or other composite containers to create child composites
         * that share a common execution context. The context is not owned; lifecycle
         * is managed by the parent.
         *
         * @param exec_context Non-owning pointer to shared execution context (must be non-null).
         *
         * @throws std::invalid_argument if exec_context is null or device type mismatches.
         */
        explicit CompositeComponent( IExecutionContext* exec_context )
            : exec_context_( exec_context )
        {
            if ( !exec_context_ )
            {
                throw std::invalid_argument(
                    "CompositeComponent: ExecutionContext cannot be null."
                );
            }

            validateExecutionContext_<TDeviceType>( exec_context_, "CompositeComponent" );
        }

        /**
         * @brief Get the shared execution context for child component construction.
         *
         * Provides access to the execution context so derived classes can construct
         * child components using the protected constructor pattern.
         *
         * @return Non-owning pointer to execution context
         */
        IExecutionContext* getExecutionContext() const noexcept
        {
            return exec_context_;
        }

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
         *     fc1_ = this->getComponentAs<LinearType>("fc1");
         *     activation_ = this->getComponentAs<GeluType>("act");
         *     fc2_ = this->getComponentAs<LinearType>("fc2");
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
            auto typed = std::dynamic_pointer_cast<TComponent>(base);

            if ( !typed )
            {
                throw std::runtime_error(
                    std::format(
                        "Component '{}' cannot be cast to requested type",
                        name
                    )
                );
            }

            return typed;
        }

        /**
         * @brief Validate that an execution context matches the expected device type.
         *
         * @tparam TExpectedDevice Expected compile-time device type
         * @param exec_context Context to validate
         * @param component_name Component name for error messages
         *
         * @throws std::invalid_argument if device types don't match
         */
        template<DeviceType TExpectedDevice>
        static void validateExecutionContext_(
            IExecutionContext* exec_context,
            const std::string& component_name )
        {
            if ( !exec_context )
            {
                throw std::invalid_argument(
                    component_name + ": ExecutionContext cannot be null."
                );
            }

            if ( exec_context->getDeviceId().type != TExpectedDevice )
            {
                throw std::invalid_argument(
                    std::format(
                        "{}: ExecutionContext device type mismatch: expected {}, got {}",
                        component_name,
                        deviceTypeToString( TExpectedDevice ),
                        deviceTypeToString( exec_context->getDeviceId().type )
                    )
                );
            }
        }

        // ====================================================================
        // Serialization (Internal Protocol)
        // ====================================================================

        /**
         * @internal
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

            archive.addMetadata( "type", getName() );
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

    private:

        /**
         * @brief Create and validate owned execution context.
         *
         * @param device_id Device identifier for context creation
         * @return Unique pointer to created execution context
         *
         * @throws std::invalid_argument if device type mismatches
         * @throws std::runtime_error if context creation fails
         */
        static std::unique_ptr<IExecutionContext> createOwnedContext( DeviceId device_id )
        {
            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format(
                        "CompositeComponent: constructor device type mismatch: expected {}, got {}",
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type )
                    )
                );
            }

            auto context = createExecutionContext( device_id );

            if ( !context )
            {
                throw std::runtime_error(
                    "CompositeComponent: failed to create execution context for device"
                );
            }

            return context;
        }

        /**
         * @brief Child components in insertion order.
         *
         * This vector holds shared ownership of the composite's direct children and
         * preserves the order in which children were added. The insertion order is
         * used for build sequencing, ordered iteration, and determining serialization
         * ordering.
         *
         * Invariant: Children are constructed but not built until parent's onBuilding()
         * is called via the template method pattern in Component::build().
         *
         * Threading: access is not internally synchronized. Mutations and lifecycle
         * operations must be externally serialized.
         */
        std::vector<ComponentPtr> child_components_;

        /**
         * @brief Lookup map from child name to component pointer.
         *
         * Provides O(1) name-based lookup for children. Keys must be unique and stable
         * (provided by caller at addComponent time). This map is used for diagnostics,
         * getComponent()/hasComponent(), and to pair with `child_components_` when
         * deterministic ordering is required.
         *
         * Note: insertion order is preserved by `child_components_`; the unordered_map
         * does not guarantee ordering.
         */
        std::unordered_map<std::string, ComponentPtr> child_component_map_;

        /**
         * @brief Execution context ownership and access.
         *
         * Ownership model:
         * - Standalone: owned_context_ is populated, exec_context_ points to it
         * - Child: owned_context_ is empty, exec_context_ points to parent's context
         */
        std::unique_ptr<IExecutionContext> owned_context_{ nullptr };
        IExecutionContext* exec_context_{ nullptr };

        friend class ComponentFactory;
    };
}