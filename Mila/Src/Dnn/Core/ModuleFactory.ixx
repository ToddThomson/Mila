/**
 * @file ModuleFactory.ixx
 * @brief High-level factory for creating built-in modules from archives.
 */

module;
#include <string>
#include <memory>
#include <stdexcept>
#include <format>

export module Dnn.ModuleFactory;

//import Dnn.Module;
//import Dnn.TensorDataType;
//import Compute.DeviceType;
//import Compute.ExecutionContext;
//import Serialization.ModelArchive;
//import nlohmann.json;

// Built-in Mila Modules
// Modules.MilaModules;

namespace Mila::Dnn
{
    //using json = nlohmann::json;
    //using namespace Mila::Dnn::Compute;
    //using namespace Mila::Dnn::Serialization;

    /**
     * @brief Factory for deserializing built-in modules from ModelArchive.
     *
     * Responsibilities:
     *  - Read module metadata from archive
     *  - Determine module type and precision
     *  - Dispatch to appropriate built-in module's fromArchive_() method
     *
     * Note: Custom user modules are not yet supported. Future versions will
     * support custom modules via a plugin system.
     *
     * Usage:
     * ```cpp
     * auto module = ModuleFactory::create<DeviceType::Cpu>(
     *     archive, "gelu_layer", exec_context );
     * ```
     */
    export class ModuleFactory
    {
    public:
        /**
         * @brief Create a built-in module from archive
         *
         * @tparam TDeviceType Device type (Cpu, Cuda)
         * @param archive Archive to read from
         * @param module_name Name of the module in the archive
         * @param exec_context Execution context for the module
         * @return Shared pointer to reconstructed module
         * @throws std::runtime_error if module type unknown or unsupported
         */
        /*template<DeviceType TDeviceType>
        static std::shared_ptr<Module<TDeviceType>> create(
            Serialization::ModelArchive& archive,
            const std::string& module_name,
            std::shared_ptr<ExecutionContext<TDeviceType>> exec_context )
        {
            const std::string prefix = "modules/" + module_name;
            json meta = archive.readJson( prefix + "/meta.json" );

            std::string module_type = meta.at( "type" ).get<std::string>();
            std::string precision_str = meta.at( "precision" ).get<std::string>();

            TensorDataType precision = parseTensorDataType( precision_str );

            return dispatchBuiltInModule<TDeviceType>(
                module_type,
                precision,
                archive,
                module_name,
                exec_context );
        }*/

    private:
        
        /**
         * @brief Dispatch to built-in module type based on module_type string
         */
        //template<DeviceType TDeviceType>
        //static std::shared_ptr<Module<TDeviceType>> dispatchBuiltInModule(
        //    const std::string& module_type,
        //    TensorDataType precision,
        //    Serialization::ModelArchive& archive,
        //    const std::string& module_name,
        //    std::shared_ptr<ExecutionContext<TDeviceType>> exec_context )
        //{
        //    // Dispatch to specific module type
        //    if (module_type == "Gelu")
        //    {
        //        //return dispatchPrecision<TDeviceType, Gelu>(
        //        //    precision, archive, module_name, exec_context );
        //    }
        //    else if (module_type == "Linear")
        //    {
        //        //return dispatchPrecision<TDeviceType, Linear>(
        //        //    precision, archive, module_name, exec_context );
        //    }
        //    
        //    // Add more built-in module types here as you implement them:
        //    // else if (module_type == "Conv2d")
        //    // {
        //    //     return dispatchPrecision<TDeviceType, Conv2d>(
        //    //         precision, archive, module_name, exec_context );
        //    // }
        //    // else if (module_type == "Encoder")
        //    // {
        //    //     // Encoder has multiple template params, handle separately
        //    //     return dispatchEncoderPrecision<TDeviceType>(
        //    //         precision, archive, module_name, exec_context, meta );
        //    // }

        //    // TODO: Check plugin system when implemented
        //    // auto* plugin = PluginManager::instance().findPlugin(module_type);
        //    // if (plugin) { return plugin->createFromArchive(...); }

        //    throw std::runtime_error(
        //        std::format( "ModuleFactory: unknown built-in module type '{}'", module_type ) );
        //}

        /**
         * @brief Dispatch on precision for a given module template
         *
         * This helper handles the precision dispatch for modules with the
         * standard template signature: Module<TDeviceType, TPrecision>
         */
        /*template<DeviceType TDeviceType, template<DeviceType, TensorDataType> class ModuleTemplate>
        static std::shared_ptr<Module<TDeviceType>> dispatchPrecision(
            TensorDataType precision,
            Serialization::ModelArchive& archive,
            const std::string& module_name,
            std::shared_ptr<ExecutionContext<TDeviceType>> exec_context )
        {
            switch (precision)
            {
                case TensorDataType::FP32:
                    return ModuleTemplate<TDeviceType, TensorDataType::FP32>::fromArchive_(
                        archive, module_name, exec_context );

                case TensorDataType::FP16:
                    return ModuleTemplate<TDeviceType, TensorDataType::FP16>::fromArchive_(
                        archive, module_name, exec_context );

                case TensorDataType::BF16:
                    return ModuleTemplate<TDeviceType, TensorDataType::BF16>::fromArchive_(
                        archive, module_name, exec_context );

                case TensorDataType::FP8_E4M3:
                    return ModuleTemplate<TDeviceType, TensorDataType::FP8_E4M3>::fromArchive_(
                        archive, module_name, exec_context );

                case TensorDataType::FP8_E5M2:
                    return ModuleTemplate<TDeviceType, TensorDataType::FP8_E5M2>::fromArchive_(
                        archive, module_name, exec_context );

                case TensorDataType::INT8:
                    return ModuleTemplate<TDeviceType, TensorDataType::INT8>::fromArchive_(
                        archive, module_name, exec_context );

                case TensorDataType::INT16:
                    return ModuleTemplate<TDeviceType, TensorDataType::INT16>::fromArchive_(
                        archive, module_name, exec_context );

                case TensorDataType::INT32:
                    return ModuleTemplate<TDeviceType, TensorDataType::INT32>::fromArchive_(
                        archive, module_name, exec_context );

                case TensorDataType::UINT8:
                    return ModuleTemplate<TDeviceType, TensorDataType::UINT8>::fromArchive_(
                        archive, module_name, exec_context );

                case TensorDataType::UINT16:
                    return ModuleTemplate<TDeviceType, TensorDataType::UINT16>::fromArchive_(
                        archive, module_name, exec_context );

                case TensorDataType::UINT32:
                    return ModuleTemplate<TDeviceType, TensorDataType::UINT32>::fromArchive_(
                        archive, module_name, exec_context );

                default:
                    throw std::runtime_error(
                        std::format( "ModuleFactory: unsupported precision for module '{}'",
                            module_name ) );
            }
        }*/

        /**
         * @brief Example: Special dispatch for modules with non-standard template params
         *
         * Some modules like Encoder have additional template parameters beyond
         * DeviceType and Precision. Handle those with custom dispatch functions.
         */
         // template<DeviceType TDeviceType>
         // static std::shared_ptr<Module<TDeviceType>> dispatchEncoderPrecision(
         //     TensorDataType precision,
         //     ModelArchive& archive,
         //     const std::string& module_name,
         //     std::shared_ptr<ExecutionContext<TDeviceType>> exec_context,
         //     const json& meta )
         // {
         //     // Encoder has template params: <TDeviceType, TIndex, TPrecision>
         //     // Need to dispatch on both index type and precision
         //     std::string index_type_str = meta.at( "index_type" ).get<std::string>();
         //     TensorDataType index_type = parsePrecision( index_type_str );
         //     
         //     // Nested dispatch
         //     if (index_type == TensorDataType::Int32 && precision == TensorDataType::Float32)
         //     {
         //         return Encoder<TDeviceType, TensorDataType::Int32, TensorDataType::Float32>
         //             ::fromArchive_( archive, module_name, exec_context );
         //     }
         //     // ... other combinations
         //     
         //     throw std::runtime_error("Unsupported Encoder template parameter combination");
         // }
    };
}