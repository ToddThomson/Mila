/**
 * @file ComponentFactory.ixx
 * @brief Factory helpers for reconstructing built-in components from archives.
 *
 * Provides small helpers that read component-scoped metadata using the
 * ModelArchive scoping API. Full dispatch/instantiation is TODO and should
 * use the scoped metadata to call the appropriate component fromArchive_()
 * factory methods.
 */

module;
#include <string>
#include <memory>
#include <stdexcept>
#include <format>

export module Dnn.ComponentFactory;

import Compute.DeviceType;
import Compute.ExecutionContext;
import Serialization.ModelArchive;
import nlohmann.json;
import Dnn.ComponentType;

namespace Mila::Dnn
{
    using json = nlohmann::json;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    export class ComponentFactory
    {
    public:

        /**
         * @brief Read a component's meta.json from the archive using scoped access.
         *
         * This helper will push a scope "components/<component_name>" on the
         * provided ModelArchive, read the component-local "meta.json", and
         * return the parsed JSON object. Caller does not need to manage scope;
         * it's handled via RAII.
         *
         * @throws std::runtime_error if meta.json is missing or parse fails
         */
        static json readComponentMeta( ModelArchive& archive, const std::string& component_name )
        {
            ModelArchive::ScopedScope scope( archive, std::string( "components/" ) + component_name );
            return archive.readJson( "meta.json" );
        }

        /**
         * @brief Parse the built-in component kind from metadata.
         *
         * Convenience helper that looks up the "type" field in the component
         * meta and maps it to the typed ComponentType enum.
         *
         * @throws std::runtime_error if the metadata does not contain a "type" field
         */
        static ComponentType parseComponentType( const json& meta )
        {
            if (!meta.contains( "type" ))
            {
                throw std::runtime_error( "ComponentFactory::parseComponentType: missing 'type' in component meta" );
            }

            const std::string type_str = meta.at( "type" ).get<std::string>();
            return fromString( type_str );
        }

        /**
         * @brief Placeholder: create a component instance from archive.
         *
         * Full instantiation requires dispatching on both component kind and
         * on-archive precision and is currently unimplemented. This method
         * demonstrates the scoped read pattern and returns nullptr.
         *
         * Implementers should:
         *  - push scope "components/<name>"
         *  - read meta/config/tensors using relative paths
         *  - dispatch to the concrete Component<...>::fromArchive_ factory
         *
         * Returning std::shared_ptr<void> is an intentional placeholder type —
         * replace with the project's chosen type-erased component handle once
         * the dispatch plumbing is implemented.
         */
        template<DeviceType TDeviceType>
        static std::shared_ptr<void> createFromArchive(
            ModelArchive& archive,
            const std::string& component_name,
            std::shared_ptr<ExecutionContext<TDeviceType>> exec_context )
        {
            // Read scoped metadata (throws if missing)
            json meta = readComponentMeta( archive, component_name );

            // Determine built-in kind
            ComponentType kind = parseComponentType( meta );

            // Determine precision if present (optional)
            std::string precision = "";
            if (meta.contains( "precision" ) && meta.at( "precision" ).is_string())
            {
                precision = meta.at( "precision" ).get<std::string>();
            }

            // TODO: Implement full dispatch to concrete component factory methods.
            // Example pattern (pseudo):
            //  if (kind == ComponentType::Linear) {
            //      return dispatchPrecision<TDeviceType, Linear>( precision, archive, component_name, exec_context );
            //  }
            // For now, return nullptr and fail loudly so callers know dispatch isn't ready.
            throw std::runtime_error(
                std::format( "ComponentFactory::createFromArchive: dispatch unimplemented for component '{}' (type='{}', precision='{}')",
                    component_name, toString( kind ), precision ) );
        }

    private:
        // Future: move common dispatch helpers here (dispatchPrecision, dispatchEncoderPrecision, etc.)
    };
}