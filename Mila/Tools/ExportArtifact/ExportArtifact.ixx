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
#include <cstdio>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <set>
#include <string>
#include <memory>
#include <stdexcept>
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

        /// Load only, print a logits fingerprint, and write nothing.
        bool fingerprint_only{ false };

        /// Emit mila.json beside the artifact. Costs a full re-read to hash it.
        bool emit_manifest{ false };

        /// Tokenizer to record in the manifest. Optional.
        std::filesystem::path tokenizer;
    };

    /**
     * @brief Policy name as recorded in the artifact metadata.
     */
    std::string weightQuantizationName( WeightQuantization quantization )
    {
        switch ( quantization )
        {
            case WeightQuantization::FP4: return "per_group_fp4_128";
            case WeightQuantization::FP8: return "per_channel_fp8_e4m3";
            default:                      return "none";
        }
    }

    /**
     * @brief Short variant key used in a coordinate and as the manifest variant name.
     */
    std::string weightQuantizationVariantName( WeightQuantization quantization )
    {
        switch ( quantization )
        {
            case WeightQuantization::FP4: return "fp4";
            case WeightQuantization::FP8: return "fp8";
            default:                      return "bf16";
        }
    }

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
     * @brief Fetch one URL through Mila's own HTTP client and report what arrived.
     *
     * Exists because a corrupted 6.33 GB download is an intolerable debugging loop. Pointed
     * at a small file from the same repository it reproduces the transport in seconds, and
     * reports the byte count and digest so the result can be diffed against a known-good
     * local copy.
     *
     * @return Process exit code.
     */
    export int runFetch( const std::string& url, const std::filesystem::path& destination )
    {
        std::unique_ptr<std::FILE, int( * )( std::FILE* )> output(
            std::fopen( destination.string().c_str(), "wb" ), &std::fclose );

        if ( output == nullptr )
        {
            std::cerr << "Cannot open " << destination.string() << " for writing\n";

            return 2;
        }

        Mila::Distribution::Sha256 hash;
        uint64_t written = 0;

        Mila::Distribution::HttpRequest request;
        request.url = url;
        request.token = Mila::Distribution::discoverHuggingFaceToken();

        std::cout << std::format( "Fetching {}\n", url );

        const auto result = Mila::Distribution::httpGet( request,
            [&]( const char* data, size_t length ) -> bool
            {
                if ( std::fwrite( data, 1, length, output.get() ) != length )
                {
                    return false;
                }

                hash.update( data, length );
                written += length;

                return true;
            } );

        output.reset();

        std::cout << std::format( "  status         {}\n", toString( result.status ) );
        std::cout << std::format( "  http code      {}\n", result.http_code );
        std::cout << std::format( "  final url      {}\n", result.final_url );
        std::cout << std::format( "  content-length {}\n", result.content_length );
        std::cout << std::format( "  bytes written  {}\n", written );
        std::cout << std::format( "  sha256         {}\n", hash.finish() );

        if ( !result.message.empty() )
        {
            std::cout << std::format( "  message        {}\n", result.message );
        }

        return result.ok() ? 0 : 1;
    }

    /**
     * @brief Hash a file, reporting progress for anything large enough to notice.
     */
    std::string hashFile( const std::filesystem::path& path )
    {
        std::unique_ptr<std::FILE, int( * )( std::FILE* )> file(
            std::fopen( path.string().c_str(), "rb" ), &std::fclose );

        if ( file == nullptr )
        {
            throw std::runtime_error( "cannot open for hashing: " + path.string() );
        }

        Mila::Distribution::Sha256 hash;
        std::string buffer( 4u << 20, '\0' );

        for ( ;; )
        {
            const size_t read = std::fread( buffer.data(), 1, buffer.size(), file.get() );

            if ( read == 0 )
            {
                break;
            }

            hash.update( buffer.data(), read );
        }

        return hash.finish();
    }

    /**
     * @brief Emit the publishable manifest fragment beside the artifact.
     *
     * Derived, never hand-written: the digests must track the bytes, and a manifest edited by
     * hand after a re-export is a repository that fails verification on every download. A
     * repository manifest covers all its variants, so this writes one variant for merging
     * rather than pretending to be the whole file.
     */
    int writeManifestFragment(
        const ExportOptions& options,
        const std::string& architecture,
        const std::filesystem::path& tokenizer )
    {
        std::cout << "Hashing artifact for the manifest\n";

        const std::string weights_digest = hashFile( options.destination );
        const auto weights_bytes = std::filesystem::file_size( options.destination );

        const std::string variant = weightQuantizationVariantName( options.quantization );

        std::string tokenizer_entry;

        if ( !tokenizer.empty() && std::filesystem::exists( tokenizer ) )
        {
            const std::string tokenizer_digest = hashFile( tokenizer );
            const auto tokenizer_bytes = std::filesystem::file_size( tokenizer );

            tokenizer_entry = std::format(
                ",\n        \"tokenizer\": {{ \"path\": \"{}\", \"sha256\": \"{}\", \"bytes\": {} }}",
                tokenizer.filename().string(), tokenizer_digest, tokenizer_bytes );
        }

        const std::string body = std::format(
            "{{\n"
            "  \"manifest_version\": 1,\n"
            "  \"architecture\": \"{}\",\n"
            "  \"default_variant\": \"{}\",\n"
            "  \"variants\": {{\n"
            "    \"{}\": {{\n"
            "      \"minimum_mila_version\": \"0.20.0\",\n"
            "      \"weight_quantization\": \"{}\",\n"
            "      \"files\": {{\n"
            "        \"weights\": {{ \"path\": \"{}\", \"sha256\": \"{}\", \"bytes\": {} }}{}\n"
            "      }}\n"
            "    }}\n"
            "  }}\n"
            "}}\n",
            architecture, variant, variant,
            weightQuantizationName( options.quantization ),
            options.destination.filename().string(), weights_digest,
            static_cast<uint64_t>( weights_bytes ),
            tokenizer_entry );

        const auto manifest_path = options.destination.parent_path() / "mila.json";

        std::unique_ptr<std::FILE, int( * )( std::FILE* )> output(
            std::fopen( manifest_path.string().c_str(), "wb" ), &std::fclose );

        if ( output == nullptr )
        {
            std::cerr << "Cannot write " << manifest_path.string() << "\n";

            return 4;
        }

        std::fwrite( body.data(), 1, body.size(), output.get() );
        output.reset();

        std::cout << std::format( "Wrote {}\n  sha256 {}\n",
            manifest_path.string(), weights_digest );

        return 0;
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
     * @brief Build context length used for the load, and deliberately not a CLI option.
     *
     * It cannot affect the artifact: the weights are what they are, and the architectural
     * max_seq_length travels in the source metadata. Its only effect is the KV cache the
     * build allocates before weights load, which the export never touches -- so the right
     * value is the smallest one that builds, always.
     */
    inline constexpr int64_t kExportContextLength = 512;

    /**
     * @brief Load the source at the requested quantization and write the artifact.
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
            model_config.withContextLength( kExportContextLength )
                .withWeightQuantization( options.quantization )
                .withKvCacheCompression( KvCacheCompression::None );

            std::cout << std::format( "Loading {}\n", options.source.string() );

            auto model = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::fromPretrained(
                options.source, model_config, DeviceId{ DeviceType::Cuda, 0 } );

            if ( options.fingerprint_only )
            {
                // Fixed, arbitrary token ids: no tokenizer is involved, so two runs over
                // different files are comparable by construction. Values are within any
                // Gemma vocabulary and their meaning is irrelevant -- only that both loads
                // see the same input.
                const std::vector<int32_t> probe{ 2, 1000, 2000, 3000, 4000, 5000, 6000, 7000 };

                std::cout << std::format( "Fingerprint of {}\n", options.source.string() );
                std::cout << "  " << model->fingerprintPrefill( probe ) << "\n";

                return 0;
            }

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

            const int reconciled = compareAgainstSource( options.source, verify );

            if ( reconciled != 0 )
            {
                return reconciled;
            }

            if ( options.emit_manifest )
            {
                return writeManifestFragment(
                    options, verify.getPretrainedMetadata().architecture, options.tokenizer );
            }

            return 0;
        }
        catch ( const std::exception& error )
        {
            std::cerr << std::format( "Export failed: {}\n", error.what() );

            return 1;
        }
    }
}
