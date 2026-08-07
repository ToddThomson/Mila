/**
 * @file ExportArtifact.ixx
 * @brief Load a Mila model at a chosen quantization and write it back as safetensors.
 *
 * Quantization is a load-time policy, so quantized weights exist nowhere until a model has
 * been built with one. Producing a pre-quantized artifact therefore means loading the BF16
 * source and writing out what ended up on the device.
 */

module;
#include <cctype>
#include <cstring>
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

        /**
         * @brief Where to assemble the package. Empty means none.
         *
         * Packaging hashes every file it declares, so exporting straight into the package
         * directory is the cheap arrangement: a file already there is described in place rather
         * than copied.
         */
        std::filesystem::path package_directory;

        /// Tokenizer to record in the manifest. Optional.
        std::filesystem::path tokenizer;

        /// Source model's license text, required by every license Mila republishes.
        std::filesystem::path license;

        /// Model card, carrying the statement that the weights were modified.
        std::filesystem::path model_card;

        /// Build over a package directory that already describes a model.
        bool replace_package{ false };
    };

    /**
     * @brief Where a package is going: the local store, with a coordinate to go under.
     */
    export struct InstallRequest
    {
        std::filesystem::path package_directory;

        /// Empty takes the manifest's name, then the package directory's name.
        std::string name;

        /// Replace an existing record of the same name rather than refusing the collision.
        bool replace{ false };

        /// Copy into the store rather than moving, leaving the package intact for a hub upload.
        bool keep_package{ false };
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
     * @brief Replay bytes already on disk into the hash, and report how many there were.
     *
     * SHA-256 is sequential and cannot be restored from a byte offset alone, so a resumed
     * transfer has to re-read what it already holds. Mirrors ModelStore's own replay so the
     * probe exercises the real protocol rather than an approximation of it.
     */
    uint64_t replayIntoHash(
        const std::filesystem::path& path, Mila::Distribution::Sha256& hash )
    {
        std::unique_ptr<std::FILE, int( * )( std::FILE* )> input(
            std::fopen( path.string().c_str(), "rb" ), &std::fclose );

        if ( input == nullptr )
        {
            return 0;
        }

        std::vector<char> buffer( 1u << 20 );
        uint64_t total = 0;

        for ( ;; )
        {
            const size_t read = std::fread( buffer.data(), 1, buffer.size(), input.get() );

            if ( read == 0 )
            {
                break;
            }

            hash.update( buffer.data(), read );
            total += read;
        }

        return total;
    }

    /**
     * @brief Fetch one URL through Mila's own HTTP transport and report what arrived.
     *
     * Exists because a corrupted 6.33 GB download is an intolerable debugging loop. Pointed
     * at a small file from the same repository it reproduces the transport in seconds, and
     * reports the byte count and digest so the result can be diffed against a known-good
     * local copy.
     *
     * Works in URLs rather than coordinates deliberately: what it diagnoses is the transport,
     * below the level at which a hub knows anything. In a build with no HTTP client the
     * transport says so and this reports that, which is the honest answer.
     *
     * With resume set, the offset comes from the destination's own size rather than from an
     * argument: that is the only offset the store can ever produce, so an arbitrary one would
     * exercise a path the real code cannot reach. Truncate a verified copy and re-run to drive
     * the Range branch, and point it at a server that ignores Range to drive the other one.
     *
     * @return Process exit code.
     */
    export int runFetch(
        const std::string& url, const std::filesystem::path& destination, bool resume )
    {
        Mila::Distribution::Sha256 hash;
        uint64_t resume_from = 0;

        if ( resume )
        {
            resume_from = replayIntoHash( destination, hash );

            if ( resume_from == 0 )
            {
                std::cerr << std::format(
                    "Nothing to resume from: {} is missing or empty\n", destination.string() );

                return 2;
            }
        }

        // Append so a resumed transfer extends the prefix rather than truncating it.
        std::unique_ptr<std::FILE, int( * )( std::FILE* )> output(
            std::fopen( destination.string().c_str(), resume_from > 0 ? "ab" : "wb" ),
            &std::fclose );

        if ( output == nullptr )
        {
            std::cerr << "Cannot open " << destination.string() << " for writing\n";

            return 2;
        }

        uint64_t written = 0;

        Mila::Distribution::HttpRequest request;
        request.url = url;
        request.token = Mila::Distribution::discoverHuggingFaceToken();
        request.resume_from = resume_from;

        if ( resume_from > 0 )
        {
            std::cout << std::format( "Resuming {} from byte {}\n", url, resume_from );
        }
        else
        {
            std::cout << std::format( "Fetching {}\n", url );
        }

        // A client, not a raw transport: --fetch is meant to reproduce what a real pull does,
        // which includes following the CDN redirect and dropping the token on the way.
        const Mila::Distribution::HttpClient client(
            Mila::Distribution::makeDefaultHttpTransport() );

        const auto result = client.get( request,
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

        std::cout << std::format( "  transport      {}\n", client.transportName() );
        std::cout << std::format( "  status         {}\n", toString( result.status ) );
        std::cout << std::format( "  http code      {}\n", result.http_code );
        std::cout << std::format( "  final url      {}\n", result.final_url );
        std::cout << std::format( "  content-length {}\n", result.content_length );

        if ( resume_from > 0 )
        {
            std::cout << std::format( "  resumed from   {}\n", resume_from );
            std::cout << std::format( "  bytes appended {}\n", written );
            std::cout << std::format( "  total bytes    {}\n", resume_from + written );
        }
        else
        {
            std::cout << std::format( "  bytes written  {}\n", written );
        }

        std::cout << std::format( "  sha256         {}\n", hash.finish() );

        if ( !result.message.empty() )
        {
            std::cout << std::format( "  message        {}\n", result.message );
        }

        // The store discards the partial here, because a prefix followed by a second copy from
        // byte zero is not salvageable. The probe keeps it: inspecting the wreck is the point.
        if ( result.status == Mila::Distribution::HttpStatus::RangeIgnored )
        {
            std::cout << std::format(
                "\n  The server ignored the Range header and sent the whole file, so {} now holds\n"
                "  a prefix followed by a full copy and the digest above means nothing. ModelStore\n"
                "  deletes the partial at this point and restarts; this kept it for inspection.\n",
                destination.string() );
        }

        return result.ok() ? 0 : 1;
    }

    /**
     * @brief The oldest Mila that can read what this build writes.
     *
     * Stamped into every variant it packages. It tracks the artifact format, not the build, so
     * it is a constant here rather than the running version -- an artifact this build produces
     * is readable by any 0.20 Mila.
     */
    inline constexpr const char* kArtifactMinimumMilaVersion = "0.20.0";

    /**
     * @brief Assemble the package around the artifact.
     *
     * Derived, never hand-written: the digests must track the bytes, and a manifest edited by
     * hand after a re-export is a repository that fails verification on every download.
     */
    int writePackage( const ExportOptions& options, const std::string& architecture )
    {
        std::cout << std::format( "Packaging into {}\n", options.package_directory.string() );

        Mila::Distribution::PackageRequest request;
        request.directory = options.package_directory;
        request.architecture = architecture;
        request.variant = weightQuantizationVariantName( options.quantization );
        request.weight_quantization = weightQuantizationName( options.quantization );
        request.minimum_mila_version = kArtifactMinimumMilaVersion;
        request.weights = options.destination;
        request.tokenizer = options.tokenizer;
        request.license = options.license;
        request.model_card = options.model_card;
        request.replace = options.replace_package;

        const auto package = Mila::Distribution::buildPackage( request );

        std::cout << std::format( "Wrote {}\n",
            ( package.directory() / Mila::Distribution::kManifestFileName ).string() );

        for ( const auto& file : package.manifest().files )
        {
            std::cout << std::format( "  {:<10} {} ({} bytes)\n",
                file.role, file.sha256, file.bytes );
        }

        // Warnings only: a package missing its license is still self-consistent, and saying so
        // at export time is what stops it reaching a hub that way.
        const auto validation = package.validate();

        for ( const auto& warning : validation.warnings )
        {
            std::cout << std::format( "  warning: {}\n", warning );
        }

        return 0;
    }

    /**
     * @brief Report whether a package agrees with its own manifest.
     *
     * @return Process exit code.
     */
    export int runValidate( const std::filesystem::path& directory )
    {
        try
        {
            const auto package = Mila::Distribution::ModelPackage::open( directory );

            std::cout << std::format(
                "Package {}\n  name         {}\n  architecture {}\n  variant      {}\n",
                package.directory().string(),
                package.manifest().name,
                package.manifest().architecture,
                package.manifest().variant );

            const auto validation = package.validate();

            for ( const auto& problem : validation.problems )
            {
                std::cerr << std::format( "  PROBLEM: {}\n", problem );
            }

            for ( const auto& warning : validation.warnings )
            {
                std::cout << std::format( "  warning: {}\n", warning );
            }

            std::cout << std::format( "  verified {} file(s), {:.2f} GB\n",
                validation.files_verified,
                static_cast<double>( validation.bytes_verified )
                / ( 1024.0 * 1024.0 * 1024.0 ) );

            if ( !validation.ok() )
            {
                std::cerr << "Package is not self-consistent and must not be published.\n";

                return 3;
            }

            std::cout << "Package is self-consistent.\n";

            return 0;
        }
        catch ( const std::exception& error )
        {
            std::cerr << std::format( "Validation failed: {}\n", error.what() );

            return 1;
        }
    }

    /**
     * @brief Install a package into the local store.
     *
     * The move is the default because it is free on one volume and leaves one copy of a file
     * that may be several gigabytes. --keep is for the case where the same directory still has
     * to be uploaded.
     *
     * @return Process exit code.
     */
    export int runInstall( const InstallRequest& request )
    {
        try
        {
            const auto package =
                Mila::Distribution::ModelPackage::open( request.package_directory );

            Mila::Distribution::ModelStore store;

            Mila::Distribution::InstallOptions options;

            options.name = request.name;
            options.replace = request.replace;
            options.move_files = !request.keep_package;

            std::cout << std::format( "Installing {} into {}\n",
                request.package_directory.string(), store.root().string() );

            const auto installed = store.install( package, options );

            std::cout << std::format(
                "Installed {}\n  architecture  {}\n  quantization  {}\n  origin        {}\n"
                "  weights       {}\n",
                installed.record.name,
                installed.record.architecture,
                installed.record.weight_quantization,
                installed.record.origin(),
                installed.weights_path.string() );

            if ( !installed.tokenizer_path.empty() )
            {
                std::cout << std::format( "  tokenizer     {}\n",
                    installed.tokenizer_path.string() );
            }

            std::cout << std::format( "  on disk       {:.2f} GB\n",
                static_cast<double>( installed.bytes_on_disk )
                / ( 1024.0 * 1024.0 * 1024.0 ) );

            return installed.complete ? 0 : 3;
        }
        catch ( const std::exception& error )
        {
            std::cerr << std::format( "Install failed: {}\n", error.what() );

            return 1;
        }
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
     * @brief Rewrite a model file as a safetensors artifact, tensor for tensor.
     *
     * Family-agnostic on purpose, and the reason it can be: an unquantized export changes the
     * container, never the numbers. Nothing here builds a model, so no GPU is involved, no
     * architecture is named, and every family the reader can open transcodes the same way --
     * which is what runExport, hardcoded to Gemma, cannot do.
     *
     * Two passes because safetensors records byte ranges in its header: everything is declared,
     * the header is emitted, then bodies stream in declaration order. Both passes walk the
     * reader's offset-ordered index, so the orders agree by construction. The first touches
     * only each blob's metadata and never dereferences the mapped bytes, so it costs no I/O.
     *
     * @return Process exit code.
     */
    export int runTranscode(
        const std::filesystem::path& source, const std::filesystem::path& destination )
    {
        if ( !std::filesystem::exists( source ) )
        {
            std::cerr << std::format( "Source not found: {}\n", source.string() );

            return 2;
        }

        try
        {
            std::cout << std::format( "Reading {}\n", source.string() );

            Serialization::PretrainedModelReader reader( source );

            const auto& metadata = reader.getPretrainedMetadata();

            std::cout << std::format( "  architecture {}\n  tensors      {}\n",
                metadata.architecture, reader.getTensorNames().size() );

            Serialization::SafeTensorsWriter writer( destination );

            writer.setMetadata(
                Serialization::kMilaConfigMetadataKey,
                Serialization::toMetadataJSON( metadata ) );

            // Carried verbatim rather than assumed absent: a .bin reports empty, but a
            // safetensors source may already be pre-quantized, and re-containering it must
            // not silently drop the policy that says how to read its bytes.
            if ( !reader.getWeightQuantization().empty() )
            {
                writer.setMetadata(
                    Serialization::kMilaQuantizationMetadataKey, reader.getWeightQuantization() );
            }

            reader.streamTensorBlobs(
                [&writer]( const std::string& name, const Serialization::ITensorBlob& blob )
                {
                    const auto& tensor = blob.getMetadata();

                    writer.declareTensor( name, tensor.dtype, tensor.shape );
                } );

            writer.beginData();

            uint64_t written = 0;

            reader.streamTensorBlobs(
                [&writer, &written]( const std::string& name, const Serialization::ITensorBlob& blob )
                {
                    writer.writeTensorData( name, blob.data(), blob.sizeBytes() );

                    written += blob.sizeBytes();
                } );

            writer.close();

            std::cout << std::format( "Wrote {} ({:.2f} GB)\n",
                destination.string(),
                static_cast<double>( written ) / ( 1024.0 * 1024.0 * 1024.0 ) );

            // Reopening is the only check that the header agrees with the data region, and the
            // reconciliation is what catches a tensor quietly dropped or duplicated.
            Serialization::PretrainedModelReader verify( destination );

            return compareAgainstSource( source, verify );
        }
        catch ( const std::exception& error )
        {
            std::cerr << std::format( "Transcode failed: {}\n", error.what() );

            return 1;
        }
    }

    /**
     * @brief Rename an installed model.
     *
     * @return Process exit code.
     */
    export int runRename( const std::string& from, const std::string& to )
    {
        try
        {
            Mila::Distribution::ModelStore store;

            if ( !store.rename( from, to ) )
            {
                std::cerr << std::format( "No model named '{}' is installed.\n", from );

                return 3;
            }

            std::cout << std::format( "Renamed {} -> {}\n", from, to );

            return 0;
        }
        catch ( const std::exception& error )
        {
            std::cerr << std::format( "Rename failed: {}\n", error.what() );

            return 1;
        }
    }

    /**
     * @brief Compare two model files tensor by tensor, bytes included.
     *
     * `compareAgainstSource` reconciles tensor *names*, which is what catches a dropped or
     * duplicated tensor. It says nothing about the contents, so "the transcode changes the
     * container, never the numbers" was until now an argument rather than a measurement. This
     * is the measurement.
     *
     * Streams both files in offset order and holds one tensor from each at a time, so it costs
     * two sequential reads and no more memory than the largest tensor.
     *
     * @return Process exit code.
     */
    export int runCompare(
        const std::filesystem::path& left_path, const std::filesystem::path& right_path )
    {
        try
        {
            Serialization::PretrainedModelReader left( left_path );
            Serialization::PretrainedModelReader right( right_path );

            const auto left_names = left.getTensorNames();

            std::cout << std::format( "Comparing {} tensors\n  {}\n  {}\n",
                left_names.size(), left_path.string(), right_path.string() );

            int mismatched = 0;
            int compared = 0;
            uint64_t bytes = 0;

            for ( const auto& name : left_names )
            {
                if ( !right.hasTensor( name ) )
                {
                    std::cerr << std::format( "  MISSING  {}\n", name );
                    ++mismatched;

                    continue;
                }

                const auto left_blob = left.readTensorBlob( name );
                const auto right_blob = right.readTensorBlob( name );

                const auto& left_meta = left_blob.getMetadata();
                const auto& right_meta = right_blob.getMetadata();

                if ( left_meta.dtype != right_meta.dtype
                    || left_meta.shape != right_meta.shape
                    || left_blob.sizeBytes() != right_blob.sizeBytes() )
                {
                    std::cerr << std::format( "  SHAPE    {} ({} vs {} bytes)\n",
                        name, left_blob.sizeBytes(), right_blob.sizeBytes() );
                    ++mismatched;

                    continue;
                }

                if ( std::memcmp( left_blob.data(), right_blob.data(),
                    left_blob.sizeBytes() ) != 0 )
                {
                    std::cerr << std::format( "  BYTES    {}\n", name );
                    ++mismatched;

                    continue;
                }

                ++compared;
                bytes += left_blob.sizeBytes();
            }

            std::cout << std::format( "  {} identical, {} mismatched, {:.2f} GB compared\n",
                compared, mismatched,
                static_cast<double>( bytes ) / ( 1024.0 * 1024.0 * 1024.0 ) );

            if ( mismatched > 0 )
            {
                return 3;
            }

            std::cout << "Every tensor is byte-identical.\n";

            return 0;
        }
        catch ( const std::exception& error )
        {
            std::cerr << std::format( "Compare failed: {}\n", error.what() );

            return 1;
        }
    }

    /**
     * @brief Package an artifact that already exists on disk.
     *
     * The counterpart to --package after an export: it takes a finished artifact rather than
     * producing one, so migrating a file, adding a licence, or fixing a model card costs a
     * manifest write instead of a multi-gigabyte reload on the GPU.
     */
    export struct PackageArtifactRequest
    {
        std::filesystem::path directory;
        std::filesystem::path weights;
        std::filesystem::path tokenizer;
        std::filesystem::path license;
        std::filesystem::path model_card;

        /// Empty takes the package directory's name.
        std::string name;

        /// Empty derives the variant from the artifact itself.
        std::string variant;

        /// Lineage, published with the model.
        std::string base_model;
        std::string license_id;

        /// Instruction-tuned. Decides the prompt template a consumer applies.
        bool instruct{ false };

        /// Build over a package directory that already describes a model.
        bool replace{ false };
    };

    /**
     * @brief The variant an artifact is, according to the artifact.
     *
     * A pre-quantized file names its policy in metadata, which is decisive. An unquantized one
     * does not, so the variant is its weight dtype -- taken from the largest tensor, because
     * that is the token embedding in every family here and it is unambiguously a weight, where
     * the first tensor in file order is whatever the converter happened to write first.
     */
    std::string deriveVariantName( Serialization::PretrainedModelReader& reader )
    {
        const std::string& quantization = reader.getWeightQuantization();

        if ( quantization == "per_group_fp4_128" )
        {
            return "fp4";
        }

        if ( quantization == "per_channel_fp8_e4m3" )
        {
            return "fp8";
        }

        TensorDataType widest{ TensorDataType::FP32 };
        size_t widest_bytes = 0;

        reader.streamTensorBlobs(
            [&widest, &widest_bytes]( const std::string&, const Serialization::ITensorBlob& blob )
            {
                if ( blob.sizeBytes() > widest_bytes )
                {
                    widest_bytes = blob.sizeBytes();
                    widest = blob.getMetadata().dtype;
                }
            } );

        std::string name = tensorDataTypeToString( widest );

        for ( char& character : name )
        {
            character = static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );
        }

        return name;
    }

    /**
     * @brief Assemble a package around an artifact that already exists.
     *
     * @return Process exit code.
     */
    export int runPackageArtifact( const PackageArtifactRequest& request )
    {
        if ( !std::filesystem::exists( request.weights ) )
        {
            std::cerr << std::format( "Weights not found: {}\n", request.weights.string() );

            return 2;
        }

        try
        {
            Serialization::PretrainedModelReader reader( request.weights );

            const std::string architecture = reader.getPretrainedMetadata().architecture;

            const std::string variant = request.variant.empty()
                ? deriveVariantName( reader ) : request.variant;

            const std::string quantization = reader.getWeightQuantization().empty()
                ? std::string( "none" ) : reader.getWeightQuantization();

            std::cout << std::format(
                "Packaging {}\n  architecture {}\n  variant      {}\n  quantization {}\n",
                request.weights.string(), architecture, variant, quantization );

            Mila::Distribution::PackageRequest package_request;
            package_request.directory = request.directory;
            package_request.name = request.name;
            package_request.architecture = architecture;
            package_request.variant = variant;
            package_request.weight_quantization = quantization;
            package_request.minimum_mila_version = kArtifactMinimumMilaVersion;
            package_request.base_model = request.base_model;
            package_request.license_id = request.license_id;
            package_request.instruct = request.instruct;
            package_request.weights = request.weights;
            package_request.tokenizer = request.tokenizer;
            package_request.license = request.license;
            package_request.model_card = request.model_card;
            package_request.replace = request.replace;

            const auto package = Mila::Distribution::buildPackage( package_request );

            std::cout << std::format( "Wrote {}\n  name         {}\n",
                ( package.directory() / Mila::Distribution::kManifestFileName ).string(),
                package.manifest().name );

            const auto validation = package.validate();

            for ( const auto& warning : validation.warnings )
            {
                std::cout << std::format( "  warning: {}\n", warning );
            }

            for ( const auto& problem : validation.problems )
            {
                std::cerr << std::format( "  PROBLEM: {}\n", problem );
            }

            return validation.ok() ? 0 : 3;
        }
        catch ( const std::exception& error )
        {
            std::cerr << std::format( "Packaging failed: {}\n", error.what() );

            return 1;
        }
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

            if ( !options.package_directory.empty() )
            {
                return writePackage( options, verify.getPretrainedMetadata().architecture );
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
