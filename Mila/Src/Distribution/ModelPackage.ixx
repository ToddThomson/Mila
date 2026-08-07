/**
 * @file ModelPackage.ixx
 * @brief A package directory: the manifest, the files it declares, and whether they agree.
 *
 * The same directory installs locally and uploads to a hub, so a package that is not
 * self-consistent must never be emitted. See Specifications/ModelDistribution.md.
 */

module;
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

export module Distribution.ModelPackage;

import Distribution.Sha256;
import Distribution.ModelManifest;

namespace Mila::Distribution
{
    /// The manifest's name at a package root and at a hub repository root. One name, both places.
    export inline constexpr std::string_view kManifestFileName = "mila.json";

    /**
     * @brief Digest a file on disk.
     *
     * Exposed because everything that produces a manifest entry needs it, and a digest computed
     * two different ways is a digest that disagrees with itself eventually.
     */
    export std::string sha256OfFile( const std::filesystem::path& path )
    {
        std::ifstream input( path, std::ios::binary );

        if ( !input.is_open() )
        {
            throw std::runtime_error( "cannot open for hashing: " + path.string() );
        }

        Sha256 hash;
        std::string buffer( 4u << 20, '\0' );

        for ( ;; )
        {
            input.read( buffer.data(), static_cast<std::streamsize>( buffer.size() ) );

            const auto read = static_cast<size_t>( input.gcount() );

            if ( read == 0 )
            {
                break;
            }

            hash.update( buffer.data(), read );
        }

        return hash.finish();
    }

    /**
     * @brief What validation found.
     *
     * Problems and warnings are separate because they have different consequences: a problem
     * means the bytes do not match what the manifest claims, and nothing may be emitted or
     * installed; a warning means the package would ship without a license or a model card, which
     * is a publishing decision rather than a corruption.
     */
    export struct PackageValidation
    {
        std::vector<std::string> problems;
        std::vector<std::string> warnings;

        /// Bytes actually read and digested.
        uint64_t bytes_verified{ 0 };

        int files_verified{ 0 };

        bool ok() const noexcept { return problems.empty(); }
    };

    /**
     * @brief Everything a published or installable model is, as a directory.
     *
     * ```
     * <package>/
     *   mila.json              the manifest
     *   <weights>.safetensors  the artifact
     *   <tokenizer>.bin
     *   LICENSE                the source model's license text
     *   README.md              model card, including the statement that changes were made
     * ```
     */
    export class ModelPackage
    {
    public:

        /**
         * @brief Read a package's manifest. Hashes nothing -- that is validate()'s job.
         *
         * @throws std::runtime_error if the directory holds no manifest, or a malformed one.
         */
        static ModelPackage open( const std::filesystem::path& directory )
        {
            const auto manifest_path = directory / kManifestFileName;

            std::error_code ignored;

            if ( !std::filesystem::exists( manifest_path, ignored ) )
            {
                throw std::runtime_error( std::format(
                    "{}: not a model package -- no {}",
                    directory.string(), kManifestFileName ) );
            }

            return ModelPackage( directory,
                parseModelManifest( readWholeFile( manifest_path ), directory.string() ) );
        }

        const std::filesystem::path& directory() const noexcept { return directory_; }
        const ModelManifest& manifest() const noexcept { return manifest_; }

        /**
         * @brief Where a declared file lives in this package.
         *
         * @throws std::runtime_error if the declared path would leave the package directory. A
         *         manifest can arrive from a hub, so a path is untrusted input and a `..` in one
         *         is an attempt to write or read outside the package.
         */
        std::filesystem::path pathOf( const ModelFile& file ) const
        {
            const std::filesystem::path relative( file.path );

            if ( relative.is_absolute() || relative.has_root_name() )
            {
                throw std::runtime_error( std::format(
                    "{}: declared path '{}' is absolute", directory_.string(), file.path ) );
            }

            for ( const auto& component : relative )
            {
                if ( component == ".." )
                {
                    throw std::runtime_error( std::format(
                        "{}: declared path '{}' escapes the package",
                        directory_.string(), file.path ) );
                }
            }

            return directory_ / relative;
        }

        /**
         * @brief Check the package against its own manifest, reading every declared byte.
         *
         * The byte count is checked before the digest so a truncated file reports its length
         * rather than an opaque mismatch, which is the difference between "the transfer stopped"
         * and "the bytes are wrong".
         */
        PackageValidation validate() const
        {
            PackageValidation validation;

            std::error_code ignored;

            for ( const auto& file : manifest_.files )
            {
                validateFile( file, validation );
            }

            if ( manifest_.architecture.empty() )
            {
                validation.problems.push_back( "the manifest declares no architecture" );
            }

            if ( !std::filesystem::exists( directory_ / "LICENSE", ignored ) )
            {
                validation.warnings.push_back(
                    "no LICENSE -- every source license Mila republishes requires the text to "
                    "travel with the model" );
            }

            if ( !std::filesystem::exists( directory_ / "README.md", ignored ) )
            {
                validation.warnings.push_back(
                    "no README.md -- the model card carries the statement that the weights were "
                    "modified, which quantization makes true" );
            }

            return validation;
        }

    private:

        ModelPackage( std::filesystem::path directory, ModelManifest manifest )
            : directory_( std::move( directory ) ), manifest_( std::move( manifest ) )
        {}

        void validateFile( const ModelFile& file, PackageValidation& validation ) const
        {
            std::filesystem::path path;

            try
            {
                path = pathOf( file );
            }
            catch ( const std::exception& error )
            {
                validation.problems.push_back( error.what() );

                return;
            }

            std::error_code ignored;

            if ( !std::filesystem::exists( path, ignored ) )
            {
                validation.problems.push_back( std::format(
                    "'{}' declares {}, which is not in the package", file.role, file.path ) );

                return;
            }

            const uint64_t bytes = std::filesystem::file_size( path, ignored );

            if ( file.bytes != 0 && bytes != file.bytes )
            {
                validation.problems.push_back( std::format(
                    "'{}' is {} bytes, the manifest declares {}", file.path, bytes, file.bytes ) );

                return;
            }

            const std::string digest = sha256OfFile( path );

            if ( digest != file.sha256 )
            {
                validation.problems.push_back( std::format(
                    "'{}' hashes to {}, the manifest declares {}",
                    file.path, digest, file.sha256 ) );

                return;
            }

            validation.bytes_verified += bytes;
            ++validation.files_verified;
        }

        static std::string readWholeFile( const std::filesystem::path& path )
        {
            std::ifstream input( path, std::ios::binary );

            if ( !input.is_open() )
            {
                throw std::runtime_error( "cannot open " + path.string() );
            }

            std::string contents;
            char buffer[ 4096 ];

            for ( ;; )
            {
                input.read( buffer, sizeof( buffer ) );

                const auto read = static_cast<size_t>( input.gcount() );

                if ( read == 0 )
                {
                    break;
                }

                contents.append( buffer, read );
            }

            return contents;
        }

        std::filesystem::path directory_;
        ModelManifest manifest_;
    };

    /**
     * @brief One model's files, as a caller hands them to the packager.
     *
     * Paths outside the package directory are brought in; paths already inside it are left where
     * they are. A 6.8 GB artifact exported straight into the package therefore costs no copy.
     */
    export struct PackageRequest
    {
        std::filesystem::path directory;

        /// Empty takes the package directory's name.
        std::string name;

        std::string architecture;
        std::string variant;
        std::string weight_quantization;
        std::string minimum_mila_version;

        /// Lineage, published with the model because attribution has to travel with the weights.
        std::string base_model;
        std::string license_id;

        /// Instruction-tuned. Decides the prompt template a consumer applies.
        bool instruct{ false };

        std::filesystem::path weights;
        std::filesystem::path tokenizer;

        std::filesystem::path license;
        std::filesystem::path model_card;

        /**
         * @brief Build over a directory that already describes a model.
         *
         * Off by default. One package is one model, so a manifest already there is either a
         * rebuild the caller meant or a directory they have forgotten holds something else, and
         * buildPackage cannot tell which -- the same reason InstallOptions::replace exists.
         */
        bool replace{ false };
    };

    namespace
    {
        /**
         * @brief Refuse to build over a directory that already describes a model.
         *
         * The manifest is the package's identity, and buildPackage derives it fresh rather than
         * merging: writing one over another silently discards a description of bytes that may
         * still be sitting beside it. Naming both models is the useful part of the refusal --
         * it separates the rebuild the caller meant from the directory they misremembered.
         */
        void requireDirectoryIsFree(
            const std::filesystem::path& directory,
            const std::string& incoming_name,
            bool replace )
        {
            if ( replace )
            {
                return;
            }

            std::error_code ignored;
            const auto manifest_path = directory / kManifestFileName;

            if ( !std::filesystem::exists( manifest_path, ignored ) )
            {
                return;
            }

            std::string existing = "an unreadable manifest";

            try
            {
                existing = std::format(
                    "\"{}\"", ModelPackage::open( directory ).manifest().name );
            }
            catch ( const std::exception& )
            {
                // Unreadable is still a manifest, and still not this call's to destroy.
            }

            throw std::runtime_error( std::format(
                "ModelPackage: {} already describes {}\n"
                "  packaging \"{}\" here would overwrite it, and one package is one model.\n"
                "  Use a separate directory, or set replace to rebuild over it deliberately.",
                manifest_path.string(), existing, incoming_name ) );
        }

        /**
         * @brief Bring a file into the package if it is not already there, and describe it.
         */
        ModelFile adoptIntoPackage(
            const std::filesystem::path& directory,
            const std::filesystem::path& source,
            std::string role )
        {
            const auto destination = directory / source.filename();

            std::error_code ignored;

            if ( !std::filesystem::equivalent( source, destination, ignored ) )
            {
                std::filesystem::copy_file( source, destination,
                    std::filesystem::copy_options::overwrite_existing );
            }

            ModelFile file;
            file.role = std::move( role );
            file.path = destination.filename().string();
            file.bytes = std::filesystem::file_size( destination, ignored );
            file.sha256 = sha256OfFile( destination );

            return file;
        }

        /**
         * @brief Place a supporting file under its conventional package name.
         */
        void copyIfOutside(
            const std::filesystem::path& directory,
            const std::filesystem::path& source,
            const char* name )
        {
            if ( source.empty() )
            {
                return;
            }

            std::error_code ignored;

            if ( !std::filesystem::exists( source, ignored ) )
            {
                throw std::runtime_error( std::format( "not found: {}", source.string() ) );
            }

            const auto destination = directory / name;

            if ( std::filesystem::equivalent( source, destination, ignored ) )
            {
                return;
            }

            std::filesystem::copy_file( source, destination,
                std::filesystem::copy_options::overwrite_existing );
        }

        void writeWholeFile( const std::filesystem::path& path, const std::string& text )
        {
            std::ofstream output( path, std::ios::binary | std::ios::trunc );

            if ( !output.is_open() )
            {
                throw std::runtime_error( "cannot open " + path.string() + " for writing" );
            }

            output.write( text.data(), static_cast<std::streamsize>( text.size() ) );
            output.close();

            if ( !output )
            {
                throw std::runtime_error( "short write for " + path.string() );
            }
        }
    }

    /**
     * @brief Assemble a package.
     *
     * The manifest is rewritten rather than merged: one package is one model at one precision, so
     * a re-export replaces what was there instead of leaving stale digests beside fresh ones. The
     * digests are computed here and never authored -- a manifest edited by hand after a re-export
     * is a repository that fails verification on every download.
     *
     * @throws std::runtime_error if a required input is missing.
     */
    export ModelPackage buildPackage( const PackageRequest& request )
    {
        if ( request.architecture.empty() )
        {
            throw std::runtime_error( "a package needs an architecture" );
        }

        std::error_code ignored;

        if ( !std::filesystem::exists( request.weights, ignored ) )
        {
            throw std::runtime_error( std::format(
                "weights not found: {}", request.weights.string() ) );
        }

        std::filesystem::create_directories( request.directory );

        ModelManifest manifest;
        manifest.name = request.name.empty()
            ? request.directory.filename().string() : request.name;

        // Refused before anything is adopted: adoptIntoPackage moves multi-gigabyte weights in,
        // and a refusal that fires after that has already rearranged the caller's disk.
        requireDirectoryIsFree( request.directory, manifest.name, request.replace );

        manifest.architecture = request.architecture;
        manifest.variant = request.variant;
        manifest.weight_quantization = request.weight_quantization;
        manifest.minimum_mila_version = request.minimum_mila_version;
        manifest.base_model = request.base_model;
        manifest.license = request.license_id;
        manifest.instruct = request.instruct;

        manifest.files.push_back(
            adoptIntoPackage( request.directory, request.weights, std::string( kWeightsRole ) ) );

        if ( !request.tokenizer.empty() )
        {
            if ( !std::filesystem::exists( request.tokenizer, ignored ) )
            {
                throw std::runtime_error( std::format(
                    "tokenizer not found: {}", request.tokenizer.string() ) );
            }

            manifest.files.push_back( adoptIntoPackage(
                request.directory, request.tokenizer, std::string( kTokenizerRole ) ) );
        }

        copyIfOutside( request.directory, request.license, "LICENSE" );
        copyIfOutside( request.directory, request.model_card, "README.md" );

        writeWholeFile( request.directory / kManifestFileName, toJsonText( manifest ) );

        return ModelPackage::open( request.directory );
    }
}
