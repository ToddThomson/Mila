/**
 * @file ExportArtifact.ixx
 * @brief Load a Mila model at a chosen quantization and write it back as safetensors.
 *
 * Quantization is a load-time policy, so quantized weights exist nowhere until a model has
 * been built with one. Producing a pre-quantized artifact therefore means loading the BF16
 * source and writing out what ended up on the device.
 */

module;
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <vector>

export module Tools.ExportArtifact;

import Mila;

namespace Mila::Tools
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    export struct ExportOptions
    {
        std::filesystem::path source;
        std::filesystem::path destination;
        WeightQuantization quantization{ WeightQuantization::FP4 };
        int64_t context_length{ 4096 };
    };

    /**
     * @brief Parse a quantization name, or report the accepted set.
     */
    export bool parseQuantization( std::string_view text, WeightQuantization& out )
    {
        if ( text == "fp4" )
        {
            out = WeightQuantization::FP4;

            return true;
        }

        if ( text == "fp8" )
        {
            out = WeightQuantization::FP8;

            return true;
        }

        if ( text == "none" || text == "bf16" )
        {
            out = WeightQuantization::None;

            return true;
        }

        return false;
    }

    /**
     * @brief Require the artifact to carry every tensor the source carried, and no strays.
     *
     * The structural checks a reopen performs -- header parses, spans tile the data region,
     * every offset in range -- all passed on an export that had silently dropped 48 tensors
     * and duplicated a 0.94 GB table. None of them compare the artifact against what it was
     * made from, which is the only question that catches an omission.
     *
     * Two directions, both fatal:
     *   missing  a source tensor absent from the artifact -- a component that owns parameters
     *            and does not emit them, which loads as an initialized-but-untrained weight
     *   extra    anything beyond the scale companions quantization legitimately adds, e.g. a
     *            tied head written as its own copy instead of borrowing the donor's
     *
     * @return 0 when the sets reconcile, 3 otherwise.
     */
    int compareAgainstSource(
        const std::filesystem::path& source_path,
        Serialization::PretrainedModelReader& artifact )
    {
        Serialization::PretrainedModelReader source( source_path );

        const auto source_names = source.getTensorNames();
        const auto artifact_names = artifact.getTensorNames();

        std::set<std::string> in_artifact( artifact_names.begin(), artifact_names.end() );
        std::set<std::string> in_source( source_names.begin(), source_names.end() );

        std::vector<std::string> missing;

        for ( const auto& name : in_source )
        {
            if ( !in_artifact.contains( name ) )
            {
                missing.push_back( name );
            }
        }

        std::vector<std::string> extra;

        for ( const auto& name : in_artifact )
        {
            if ( in_source.contains( name ) || name.ends_with( "_scale" ) )
            {
                continue;
            }

            extra.push_back( name );
        }

        if ( missing.empty() && extra.empty() )
        {
            std::cout << std::format(
                "Reconciled against source: {} source tensors, {} artifact tensors "
                "({} scale companions added)\n",
                in_source.size(), in_artifact.size(),
                in_artifact.size() - in_source.size() );

            return 0;
        }

        std::cerr << std::format( "Artifact does not reconcile with {}\n", source_path.string() );

        if ( !missing.empty() )
        {
            std::cerr << std::format( "  {} tensor(s) MISSING from the artifact:\n", missing.size() );

            for ( size_t index = 0; index < missing.size() && index < 8; ++index )
            {
                std::cerr << std::format( "    {}\n", missing[ index ] );
            }

            if ( missing.size() > 8 )
            {
                std::cerr << std::format( "    ... and {} more\n", missing.size() - 8 );
            }
        }

        if ( !extra.empty() )
        {
            std::cerr << std::format(
                "  {} unexpected tensor(s) in the artifact:\n", extra.size() );

            for ( size_t index = 0; index < extra.size() && index < 8; ++index )
            {
                std::cerr << std::format( "    {}\n", extra[ index ] );
            }
        }

        return 3;
    }

    /**
     * @brief Load the source at the requested quantization and write the artifact.
     *
     * The context length is kept small by default: the export never runs a forward pass, but
     * the network is built before weights load, and build allocates a KV cache proportional
     * to it. A large value would cost gigabytes of device memory the export never touches.
     *
     * @return Process exit code.
     */
    export int runExport( const ExportOptions& options )
    {
        if ( !std::filesystem::exists( options.source ) )
        {
            std::cerr << std::format( "Source not found: {}\n", options.source.string() );

            return 2;
        }

        try
        {
            // The model load path logs, and the logger is not implicitly created; without
            // this the load throws before reading a byte.
            Mila::initialize();

            // Built in place rather than chained off a temporary: the fluent setters
            // return an lvalue reference to *this.
            GemmaModelConfig model_config;
            model_config.withContextLength( options.context_length )
                .withWeightQuantization( options.quantization )
                .withKvCacheCompression( KvCacheCompression::None );

            std::cout << std::format( "Loading {}\n", options.source.string() );

            auto model = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::fromPretrained(
                options.source, model_config, DeviceId{ DeviceType::Cuda, 0 } );

            std::cout << std::format( "Writing {}\n", options.destination.string() );

            model->saveArtifact( options.destination );

            const auto bytes = std::filesystem::file_size( options.destination );

            std::cout << std::format( "Wrote {:.2f} GB\n",
                static_cast<double>( bytes ) / ( 1024.0 * 1024.0 * 1024.0 ) );

            // Reopening is the only check that the header agrees with the data region; a
            // writer bug produces a file that looks finished and fails at load.
            Serialization::PretrainedModelReader verify( options.destination );

            std::cout << std::format( "Verified {} tensors, architecture '{}'\n",
                verify.getTensorNames().size(),
                verify.getPretrainedMetadata().architecture );

            return compareAgainstSource( options.source, verify );
        }
        catch ( const std::exception& error )
        {
            std::cerr << std::format( "Export failed: {}\n", error.what() );

            return 1;
        }
    }
}
