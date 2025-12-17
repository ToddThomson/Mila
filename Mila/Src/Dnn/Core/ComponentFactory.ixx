/**
 * @file ComponentFactory.ixx
 * @brief Factory helpers for reconstructing built-in components from archives.
 *
 * Provides helpers that read component-scoped metadata using the
 * ModelArchive scoping API and SerializationMetadata abstraction.
 * Full dispatch/instantiation is TODO and should use the scoped metadata
 * to call the appropriate component fromArchive_() factory methods.
 */

module;
#include <string>
#include <memory>
#include <stdexcept>
#include <format>

export module Dnn.ComponentFactory;

import Compute.DeviceType;
import Compute.ExecutionContextTemplate;
import Serialization.ModelArchive;
import Serialization.Metadata;
import Dnn.ComponentType;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Factory for reconstructing components from serialized archives.
     *
     * Provides type-safe helpers that read component metadata using
     * SerializationMetadata abstraction, eliminating JSON exposure in
     * component reconstruction code.
     *
     * Design Pattern:
     * - Uses ModelArchive scoping API for organized access
     * - Abstracts serialization format via SerializationMetadata
     * - Provides extensible dispatch infrastructure for component types
     */
    export class ComponentFactory
    {
    public:

        /**
         * @brief Read a component's metadata from the archive using scoped access.
         *
         * This helper pushes scope "components/<component_name>" on the
         * provided ModelArchive, reads the component-local "meta.json",
         * and returns the parsed metadata. Caller does not need to manage scope;
         * it's handled via RAII.
         *
         * @param archive ModelArchive to read from
         * @param component_name Name of the component to load
         * @return Type-safe metadata container
         *
         * @throws std::runtime_error if meta.json is missing or parse fails
         *
         * @example
         * auto meta = ComponentFactory::readComponentMeta(archive, "fc1");
         * std::string type = meta.getString("type");
         * int64_t version = meta.getInt("version");
         */
        static SerializationMetadata readComponentMeta( ModelArchive& archive, const std::string& component_name )
        {
            ModelArchive::ScopedScope scope( archive, std::string( "components/" ) + component_name );
            return archive.readMetadata( "meta.json" );
        }

        /**
         * @brief Parse the built-in component kind from metadata.
         *
         * Convenience helper that looks up the "type" field in the component
         * metadata and maps it to the typed ComponentType enum.
         *
         * @param meta Component metadata
         * @return ComponentType enum value
         *
         * @throws std::runtime_error if the metadata does not contain a "type" field
         * @throws std::runtime_error if the type string is not recognized
         *
         * @example
         * auto meta = readComponentMeta(archive, "fc1");
         * ComponentType type = parseComponentType(meta);
         * if (type == ComponentType::Linear) { ... }
         */
        static ComponentType parseComponentType( const SerializationMetadata& meta )
        {
            if ( !meta.has( "type" ) )
            {
                throw std::runtime_error( "ComponentFactory::parseComponentType: missing 'type' in component meta" );
            }

            const std::string type_str = meta.getString( "type" );
            return fromString( type_str );
        }

        /**
         * @brief Get component version from metadata.
         *
         * Retrieves the component version field for compatibility checking
         * during deserialization.
         *
         * @param meta Component metadata
         * @return Version number
         *
         * @throws std::runtime_error if version field is missing
         */
        static int64_t getComponentVersion( const SerializationMetadata& meta )
        {
            if ( !meta.has( "version" ) )
            {
                throw std::runtime_error( "ComponentFactory::getComponentVersion: missing 'version' in component meta" );
            }

            return meta.getInt( "version" );
        }

        /**
         * @brief Get component name from metadata.
         *
         * Retrieves the component name field.
         *
         * @param meta Component metadata
         * @return Component name
         *
         * @throws std::runtime_error if name field is missing
         */
        static std::string getComponentName( const SerializationMetadata& meta )
        {
            if ( !meta.has( "name" ) )
            {
                throw std::runtime_error( "ComponentFactory::getComponentName: missing 'name' in component meta" );
            }

            return meta.getString( "name" );
        }

        /**
         * @brief Get optional precision from metadata.
         *
         * Retrieves the precision field if present, otherwise returns empty string.
         *
         * @param meta Component metadata
         * @return Precision string or empty if not specified
         */
        static std::string getPrecision( const SerializationMetadata& meta )
        {
            auto precision_opt = meta.tryGetString( "precision" );
            return precision_opt.value_or( "" );
        }

        /**
         * @brief Placeholder: create a component instance from archive.
         *
         * Full instantiation requires dispatching on both component kind and
         * on-archive precision and is currently unimplemented. This method
         * demonstrates the scoped read pattern and throws on invocation.
         *
         * Implementation Strategy:
         *  - Push scope "components/<name>"
         *  - Read meta/config/tensors using relative paths
         *  - Dispatch to the concrete Component<...>::fromArchive_ factory
         *  - Return properly typed component instance
         *
         * @param archive ModelArchive containing serialized component
         * @param component_name Name of component to instantiate
         * @param exec_context Execution context for the component
         * @return Shared pointer to component (placeholder type)
         *
         * @throws std::runtime_error Always throws - dispatch not yet implemented
         *
         * @note Returning std::shared_ptr<void> is an intentional placeholder type —
         *       replace with the project's chosen type-erased component handle once
         *       the dispatch plumbing is implemented.
         *
         * @example Future usage pattern:
         * @code
         * auto component = ComponentFactory::createFromArchive<DeviceType::Cpu>(
         *     archive, "fc1", exec_context);
         * @endcode
         */
        template<DeviceType TDeviceType>
        static std::shared_ptr<void> createFromArchive(
            ModelArchive& archive,
            const std::string& component_name,
            std::shared_ptr<ExecutionContext<TDeviceType>> exec_context )
        {
            SerializationMetadata meta = readComponentMeta( archive, component_name );

            ComponentType kind = parseComponentType( meta );

            std::string precision = getPrecision( meta );

            // TODO: Implement full dispatch to concrete component factory methods.
            // Example pattern (pseudo):
            //  if (kind == ComponentType::Linear) {
            //      return dispatchPrecision<TDeviceType, Linear>( precision, archive, component_name, exec_context );
            //  }
            //  else if (kind == ComponentType::Gelu) {
            //      return dispatchPrecision<TDeviceType, Gelu>( precision, archive, component_name, exec_context );
            //  }
            //
            // Full implementation should:
            // 1. Switch on ComponentType to select component template
            // 2. Dispatch on precision string to select TensorDataType
            // 3. Instantiate Component<TDeviceType, TPrecision>::fromArchive_()
            // 4. Return properly typed shared_ptr (or type-erased wrapper)

            throw std::runtime_error(
                std::format( "ComponentFactory::createFromArchive: dispatch unimplemented for component '{}' (type='{}', precision='{}')",
                    component_name, toString( kind ), precision ) );
        }

    private:
        // Future: move common dispatch helpers here
        // - dispatchPrecision<TDeviceType, TComponent>
        // - dispatchComponentType<TDeviceType>
        // - validateComponentMetadata
    };
}