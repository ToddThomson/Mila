module;
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <cstdint>

export module ModelMetadata;

import nlohmann.json;

namespace Mila::Dnn::Modeling
{
    using json = nlohmann::json;

    /**
     * @brief Metadata for exported models
     */
    export struct ModelMetadata
    {
        std::string name;
        std::string version = "1.0.0";
        std::string framework_version = "Mila 1.0";
        std::chrono::system_clock::time_point export_time;
        std::size_t training_epochs = 0;
        double final_loss = 0.0;
        std::map<std::string, std::string> custom_metadata;

        // Input/Output specifications
        struct TensorSpec
        {
            std::vector<std::size_t> shape;
            std::string dtype;
            std::string name;

            // Serialize to JSON
            json to_json() const
            {
                json j;
                j["shape"] = shape;
                j["dtype"] = dtype;
                j["name"] = name;

                return j;
            }

            // Deserialize from JSON
            static TensorSpec from_json( const json& j )
            {
                TensorSpec spec;
                j.at( "shape" ).get_to( spec.shape );
                j.at( "dtype" ).get_to( spec.dtype );
                j.at( "name" ).get_to( spec.name );

                return spec;
            }
        };

        std::vector<TensorSpec> inputs;
        std::vector<TensorSpec> outputs;

        // Serialize metadata to JSON
        json to_json() const
        {
            json j;
            j["name"] = name;
            j["version"] = version;
            j["framework_version"] = framework_version;
            j["export_time"] = std::chrono::system_clock::to_time_t( export_time );
            j["training_epochs"] = training_epochs;
            j["final_loss"] = final_loss;
            j["custom_metadata"] = custom_metadata;

            json inputs_array = json::array();
            for (const auto& input : inputs)
            {
                inputs_array.push_back( input.to_json() );
            }
            j["inputs"] = inputs_array;

            json outputs_array = json::array();
            for (const auto& output : outputs)
            {
                outputs_array.push_back( output.to_json() );
            }
            j["outputs"] = outputs_array;

            return j;
        }

        // Deserialize metadata from JSON
        static ModelMetadata from_json( const json& j )
        {
            ModelMetadata meta;
            j.at( "name" ).get_to( meta.name );
            j.at( "version" ).get_to( meta.version );
            j.at( "framework_version" ).get_to( meta.framework_version );

            int64_t epoch_time = 0;
            j.at( "export_time" ).get_to( epoch_time );
            meta.export_time = std::chrono::system_clock::from_time_t( epoch_time );

            j.at( "training_epochs" ).get_to( meta.training_epochs );
            j.at( "final_loss" ).get_to( meta.final_loss );
            j.at( "custom_metadata" ).get_to( meta.custom_metadata );

            if (j.contains( "inputs" ))
            {
                for (const auto& input_json : j["inputs"])
                {
                    meta.inputs.push_back( TensorSpec::from_json( input_json ) );
                }
            }

            if (j.contains( "outputs" ))
            {
                for (const auto& output_json : j["outputs"])
                {
                    meta.outputs.push_back( TensorSpec::from_json( output_json ) );
                }
            }

            return meta;
        }
    };
}