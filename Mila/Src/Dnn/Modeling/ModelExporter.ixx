module;
#include <filesystem>
#include <string>
#include <stdexcept>
#include <format>
#include <system_error>
#include <exception>

export module Modeling.ModelExporter;

namespace Mila::Dnn::Modeling
{
    /**
     * @brief Helper class for model exporting
     */
    export class ModelExporter
    {
    public:
        /**
         * @brief Export a trained model for inference
         * @param network The trained network
         * @param filepath Path to save the exported model
         * @param metadata Model metadata
         */
        static void export_model( const Network& network,
            const std::filesystem::path& filepath,
            const ModelMetadata& metadata )
        {
            ZipSerializer serializer;

            if (!serializer.openForWrite( filepath.string() ))
            {
                throw std::runtime_error(
                    std::format( "ModelExporter: Failed to create export file: {}",
                        filepath.string() )
                );
            }

            try
            {
                // Save format marker
                serializer.addMetadata( "format", "mila-inference-v1" );

                // Save metadata as JSON file
                json meta_json = metadata.to_json();
                std::string meta_str = meta_json.dump( 2 ); // Pretty print with indent=2
                serializer.addData( "metadata.json", meta_str.data(), meta_str.size() );

                // Save architecture as JSON
                auto arch_json = network.toJson();
                serializer.addData( "architecture.json", arch_json.data(), arch_json.size() );

                // Save weights (delegate to network's save method)
                network.save( serializer );

                serializer.close();

                auto file_size = std::filesystem::file_size( filepath );
                std::println( "? Model exported: {} ({:.2f} MB)",
                    filepath.filename().string(),
                    static_cast<double>(file_size) / (1024.0 * 1024.0) );

            }
            catch (const std::exception& e)
            {
                serializer.close();
                // Clean up partial file
                std::error_code ec;
                std::filesystem::remove( filepath, ec );
                
                throw std::runtime_error(
                    std::format( "ModelExporter: Failed to export model: {}", e.what() )
                );
            }
        }

        /**
         * @brief Load only metadata from an exported model without loading weights
         */
        static ModelMetadata loadMetadata( const std::filesystem::path& filepath )
        {
            if (!std::filesystem::exists( filepath ))
            {
                throw std::runtime_error(
                    std::format( "ModelExporter: Model file not found: {}",
                        filepath.string() )
                );
            }

            ZipSerializer serializer;
            
            if (!serializer.openForRead( filepath.string() ))
            {
                throw std::runtime_error(
                    std::format( "ModelExporter: Failed to open model: {}",
                        filepath.string() )
                );
            }

            try
            {
                auto metadata_size = serializer.getFileSize( "metadata.json" );
                std::string metadata_json( metadata_size, '\0' );
                serializer.extractData( "metadata.json", metadata_json.data(), metadata_size );

                json meta_j = json::parse( metadata_json );
                ModelMetadata metadata = ModelMetadata::from_json( meta_j );

                serializer.close();
                return metadata;

            }
            catch (const std::exception& e)
            {
                serializer.close();
                throw std::runtime_error(
                    std::format( "ModelExporter: Failed to load metadata: {}", e.what() )
                );
            }
        }
    };
}