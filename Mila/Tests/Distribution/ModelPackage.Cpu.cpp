/**
 * @file ModelPackage.Cpu.cpp
 * @brief The manifest schema, package assembly and validation, and installing into the store.
 *
 * Everything here is filesystem-only: packaging and installing are what a build with no hub
 * still has to do, so these cases run on the MILA_ENABLE_CUDA=OFF gate and take no network.
 *
 * C stdio rather than fstream throughout: MSVC C++23 raises C2079 on basic_istream::sentry
 * when stream I/O meets `import Mila;` in a .cpp.
 */

#include <gtest/gtest.h>
#include <cstdio>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

import Mila;

namespace Mila::Tests::Distribution
{
    using namespace Mila::Distribution;

    namespace
    {
        using FilePointer = std::unique_ptr<std::FILE, int( * )( std::FILE* )>;

        FilePointer openFile( const std::filesystem::path& path, const char* mode )
        {
            return FilePointer( std::fopen( path.string().c_str(), mode ), &std::fclose );
        }

        void writeWholeFile( const std::filesystem::path& path, std::string_view contents )
        {
            std::filesystem::create_directories( path.parent_path() );

            auto file = openFile( path, "wb" );
            ASSERT_NE( file.get(), nullptr );
            std::fwrite( contents.data(), 1, contents.size(), file.get() );
        }

        std::string readWholeFile( const std::filesystem::path& path )
        {
            auto file = openFile( path, "rb" );

            if ( file == nullptr )
            {
                return {};
            }

            std::string contents;
            char buffer[ 4096 ];

            for ( ;; )
            {
                const size_t read = std::fread( buffer, 1, sizeof( buffer ), file.get() );

                if ( read == 0 )
                {
                    break;
                }

                contents.append( buffer, read );
            }

            return contents;
        }

        /**
         * @brief A directory under the temp directory, removed on destruction.
         */
        class ScratchDirectory
        {
        public:

            explicit ScratchDirectory( const char* label )
            {
                static int counter = 0;

                path_ = std::filesystem::temp_directory_path()
                    / std::format( "mila_{}_test_{}", label, counter++ );

                std::error_code ignored;
                std::filesystem::remove_all( path_, ignored );
                std::filesystem::create_directories( path_ );
            }

            ~ScratchDirectory()
            {
                std::error_code ignored;
                std::filesystem::remove_all( path_, ignored );
            }

            ScratchDirectory( const ScratchDirectory& ) = delete;
            ScratchDirectory& operator=( const ScratchDirectory& ) = delete;

            const std::filesystem::path& path() const { return path_; }

        private:

            std::filesystem::path path_;
        };

        constexpr std::string_view kWeightsPayload = "not really a safetensors file, but bytes";
        constexpr std::string_view kTokenizerPayload = "tokenizer bytes";

        /**
         * @brief A package holding one variant, its weights and a tokenizer.
         */
        ModelPackage buildScratchPackage(
            const std::filesystem::path& directory,
            const std::string& name = "gemma-4-12b-it-fp4",
            bool with_license = true,
            std::string_view weights_payload = kWeightsPayload )
        {
            writeWholeFile( directory / "sources" / "model.safetensors", weights_payload );
            writeWholeFile( directory / "sources" / "tokenizer.bin", kTokenizerPayload );

            if ( with_license )
            {
                writeWholeFile( directory / "sources" / "LICENSE", "Apache 2.0, in spirit" );
                writeWholeFile( directory / "sources" / "CARD.md", "# A model\nChanges: yes" );
            }

            PackageRequest request;
            request.directory = directory / "package";
            request.name = name;
            request.architecture = "gemma";
            request.variant = "fp4";
            request.weight_quantization = "per_group_fp4_128";
            request.minimum_mila_version = "0.20.0";
            request.instruct = true;
            request.weights = directory / "sources" / "model.safetensors";
            request.tokenizer = directory / "sources" / "tokenizer.bin";

            if ( with_license )
            {
                request.license = directory / "sources" / "LICENSE";
                request.model_card = directory / "sources" / "CARD.md";
            }

            return buildPackage( request );
        }

        /// The spec's own example, minus the digests, which the tests supply.
        std::string manifestJson(
            const std::string& weights_digest,
            const std::string& minimum_version = "0.20.0" )
        {
            return std::format( R"({{
              "manifest_version": 1,
              "name": "gemma-4-12b-it-fp4",
              "architecture": "gemma",
              "variant": "fp4",
              "weight_quantization": "per_group_fp4_128",
              "minimum_mila_version": "{}",
              "base_model": "google/gemma-4-12b-it",
              "license": "apache-2.0",
              "files": {{
                "weights": {{ "path": "m.safetensors", "sha256": "{}", "bytes": 40 }},
                "tokenizer": {{ "path": "t.bin", "sha256": "abc", "bytes": 15 }}
              }}
            }})", minimum_version, weights_digest );
        }
    }

    // ================================================================
    // The manifest schema
    // ================================================================

    TEST( ModelManifestTests, ParsesAModelAndItsFiles )
    {
        const auto manifest = parseModelManifest( manifestJson( "aaa" ), "test" );

        EXPECT_EQ( manifest.name, "gemma-4-12b-it-fp4" );
        EXPECT_EQ( manifest.architecture, "gemma" );
        EXPECT_EQ( manifest.variant, "fp4" );
        EXPECT_EQ( manifest.weight_quantization, "per_group_fp4_128" );

        // Lineage is published, because attribution has to travel with the weights.
        EXPECT_EQ( manifest.base_model, "google/gemma-4-12b-it" );
        EXPECT_EQ( manifest.license, "apache-2.0" );

        ASSERT_NE( manifest.file( kWeightsRole ), nullptr );
        EXPECT_EQ( manifest.file( kWeightsRole )->sha256, "aaa" );
        EXPECT_EQ( manifest.file( kWeightsRole )->bytes, 40u );
        ASSERT_NE( manifest.file( kTokenizerRole ), nullptr );
    }

    TEST( ModelManifestTests, RefusesAModelWithNoWeights )
    {
        // A model declaring only a tokenizer is not one, and the failure belongs at the parse
        // rather than at a load that finds no weights_path.
        EXPECT_THROW( parseModelManifest(
            R"({"architecture":"gemma","files":{"tokenizer":{"path":"t.bin","sha256":"abc"}}})",
            "test" ), std::runtime_error );
    }

    TEST( ModelManifestTests, RefusesAFileWithNoDigest )
    {
        EXPECT_THROW( parseModelManifest(
            R"({"files":{"weights":{"path":"m.safetensors"}}})", "test" ), std::runtime_error );
    }

    TEST( ModelManifestTests, RefusesAManifestWithNoFiles )
    {
        EXPECT_THROW( parseModelManifest( R"({"architecture":"gemma"})", "test" ),
            std::runtime_error );
        EXPECT_THROW( parseModelManifest( "this is not json", "test" ), std::runtime_error );
    }

    TEST( ModelManifestTests, RoundTripsThroughItsOwnSerializer )
    {
        const auto manifest = parseModelManifest( manifestJson( "aaa" ), "test" );
        const std::string text = toJsonText( manifest );
        const auto reparsed = parseModelManifest( text, "test" );

        EXPECT_EQ( reparsed.name, manifest.name );
        EXPECT_EQ( reparsed.architecture, manifest.architecture );
        EXPECT_EQ( reparsed.variant, manifest.variant );
        EXPECT_EQ( reparsed.base_model, manifest.base_model );
        EXPECT_EQ( reparsed.file( kWeightsRole )->sha256, "aaa" );
        EXPECT_EQ( reparsed.file( kTokenizerRole )->bytes, 15u );

        // The emitted file is read by people: the weights lead, whatever order a map would put
        // the roles in.
        EXPECT_LT( text.find( "\"weights\"" ), text.find( "\"tokenizer\"" ) );
    }

    TEST( ModelManifestTests, RefusesAModelRequiringANewerMila )
    {
        const auto too_new = parseModelManifest( manifestJson( "aaa", "99.0.0" ), "test" );

        EXPECT_THROW( requireCompatibleMilaVersion( too_new, "test" ), std::runtime_error );

        const auto current = parseModelManifest( manifestJson( "aaa", "0.1.0" ), "test" );

        EXPECT_NO_THROW( requireCompatibleMilaVersion( current, "test" ) );
    }

    TEST( ModelManifestTests, RefusesAMinimumVersionThatIsNotAVersion )
    {
        const auto manifest = parseModelManifest( manifestJson( "aaa", "soon" ), "test" );

        // Refused rather than read as one this build happens to satisfy.
        EXPECT_THROW( requireCompatibleMilaVersion( manifest, "test" ), std::runtime_error );
    }

    // ================================================================
    // Package assembly
    // ================================================================

    TEST( ModelPackageTests, AssemblesAPackageWhoseDigestsAreDerivedFromTheBytes )
    {
        ScratchDirectory scratch( "package" );

        const auto package = buildScratchPackage( scratch.path() );

        EXPECT_EQ( package.manifest().name, "gemma-4-12b-it-fp4" );
        EXPECT_EQ( package.manifest().architecture, "gemma" );
        EXPECT_EQ( package.manifest().variant, "fp4" );

        const ModelFile* weights = package.manifest().file( kWeightsRole );
        ASSERT_NE( weights, nullptr );
        EXPECT_EQ( weights->path, "model.safetensors" );
        EXPECT_EQ( weights->bytes, kWeightsPayload.size() );
        EXPECT_EQ( weights->sha256,
            sha256Hex( kWeightsPayload.data(), kWeightsPayload.size() ) );

        // The supporting files land under the names the package layout fixes, whatever they
        // were called at the source.
        EXPECT_TRUE( std::filesystem::exists( package.directory() / "LICENSE" ) );
        EXPECT_TRUE( std::filesystem::exists( package.directory() / "README.md" ) );

        const auto validation = package.validate();
        EXPECT_TRUE( validation.ok() ) << validation.problems.front();
        EXPECT_TRUE( validation.warnings.empty() );
        EXPECT_EQ( validation.files_verified, 2 );
    }

    TEST( ModelPackageTests, RepackagingReplacesTheManifestRatherThanAccumulating )
    {
        ScratchDirectory scratch( "package" );

        buildScratchPackage( scratch.path() );

        // A re-export changes the bytes. The old digest must not survive beside the new one.
        writeWholeFile( scratch.path() / "package" / "model.safetensors", "re-exported bytes" );

        PackageRequest request;
        request.directory = scratch.path() / "package";
        request.name = "gemma-4-12b-it-fp4";
        request.architecture = "gemma";
        request.variant = "fp4";
        request.weight_quantization = "per_group_fp4_128";
        request.weights = scratch.path() / "package" / "model.safetensors";
        request.tokenizer = scratch.path() / "package" / "tokenizer.bin";

        const auto package = buildPackage( request );

        EXPECT_EQ( package.manifest().file( kWeightsRole )->sha256,
            sha256Hex( "re-exported bytes", 17 ) );
        EXPECT_TRUE( package.validate().ok() );
    }

    TEST( ModelPackageTests, TakesItsNameFromTheDirectoryWhenNoneIsGiven )
    {
        ScratchDirectory scratch( "package" );

        writeWholeFile( scratch.path() / "my-model-bf16" / "w.safetensors", kWeightsPayload );

        PackageRequest request;
        request.directory = scratch.path() / "my-model-bf16";
        request.architecture = "llama";
        request.variant = "bf16";
        request.weights = scratch.path() / "my-model-bf16" / "w.safetensors";

        EXPECT_EQ( buildPackage( request ).manifest().name, "my-model-bf16" );
    }

    // ================================================================
    // Validation
    // ================================================================

    TEST( ModelPackageTests, ValidationCatchesAlteredBytes )
    {
        ScratchDirectory scratch( "package" );

        const auto package = buildScratchPackage( scratch.path() );

        // Same length, different content: only the digest can tell.
        writeWholeFile( package.directory() / "tokenizer.bin", "tokenizer bytez" );

        const auto validation = package.validate();

        ASSERT_FALSE( validation.ok() );
        EXPECT_NE( validation.problems.front().find( "hashes to" ), std::string::npos );
    }

    TEST( ModelPackageTests, ValidationReportsALengthBeforeADigest )
    {
        ScratchDirectory scratch( "package" );

        const auto package = buildScratchPackage( scratch.path() );

        writeWholeFile( package.directory() / "model.safetensors", "short" );

        const auto validation = package.validate();

        ASSERT_FALSE( validation.ok() );

        // "the transfer stopped" and "the bytes are wrong" are different diagnoses, and a
        // truncated file should say which one it is.
        EXPECT_NE( validation.problems.front().find( "bytes" ), std::string::npos );
        EXPECT_EQ( validation.problems.front().find( "hashes to" ), std::string::npos );
    }

    TEST( ModelPackageTests, ValidationCatchesAMissingFile )
    {
        ScratchDirectory scratch( "package" );

        const auto package = buildScratchPackage( scratch.path() );

        std::error_code ignored;
        std::filesystem::remove( package.directory() / "tokenizer.bin", ignored );

        const auto validation = package.validate();

        ASSERT_FALSE( validation.ok() );
        EXPECT_NE( validation.problems.front().find( "not in the package" ), std::string::npos );
    }

    TEST( ModelPackageTests, WarnsAboutAMissingLicenseAndModelCard )
    {
        ScratchDirectory scratch( "package" );

        const auto package =
            buildScratchPackage( scratch.path(), "gemma-4-12b-it-fp4", false );

        const auto validation = package.validate();

        // Publishable in the sense that the bytes agree; not publishable in the sense that
        // every license Mila republishes requires its text to travel with the model.
        EXPECT_TRUE( validation.ok() );
        EXPECT_EQ( validation.warnings.size(), 2u );
    }

    TEST( ModelPackageTests, RefusesADeclaredPathThatEscapesThePackage )
    {
        ScratchDirectory scratch( "package" );

        // A manifest can arrive from a hub, so a declared path is untrusted input.
        writeWholeFile( scratch.path() / "mila.json", R"({
          "name": "escaping",
          "architecture": "gemma",
          "files": {
            "weights": { "path": "../../escaped.safetensors", "sha256": "aaa", "bytes": 1 } }
        })" );

        const auto package = ModelPackage::open( scratch.path() );
        const auto validation = package.validate();

        ASSERT_FALSE( validation.ok() );
        EXPECT_NE( validation.problems.front().find( "escapes" ), std::string::npos );
    }

    TEST( ModelPackageTests, OpeningADirectoryWithNoManifestSaysSo )
    {
        ScratchDirectory scratch( "package" );

        EXPECT_THROW( ModelPackage::open( scratch.path() ), std::runtime_error );
    }

    // ================================================================
    // Installing into the local store
    // ================================================================

    TEST( ModelStoreInstallTests, InstallsAPackageUnderItsName )
    {
        ScratchDirectory scratch( "package" );
        ScratchDirectory store_root( "store" );

        const auto package = buildScratchPackage( scratch.path() );

        ModelStore store( store_root.path() );

        const auto installed = store.install( package );

        EXPECT_EQ( installed.record.name, "gemma-4-12b-it-fp4" );
        EXPECT_EQ( installed.record.architecture, "gemma" );
        EXPECT_EQ( installed.record.variant, "fp4" );
        EXPECT_EQ( installed.record.weight_quantization, "per_group_fp4_128" );
        EXPECT_TRUE( installed.complete );

        // Pinned because it was silently wrong on three real models: the flag decides the
        // prompt template, and nothing downstream can tell that it was dropped.
        EXPECT_TRUE( installed.record.instruct );
        EXPECT_TRUE( store.locate( "gemma-4-12b-it-fp4" )->record.instruct );

        // Nothing served this, so it is local -- a field on the record, not a namespace.
        EXPECT_TRUE( installed.record.isLocal() );
        EXPECT_EQ( installed.record.origin(), "local" );
        EXPECT_TRUE( installed.record.hub.empty() );

        // The bytes are in the store, at their digest, and the load path finds them there.
        EXPECT_EQ( readWholeFile( installed.weights_path ), kWeightsPayload );
        EXPECT_EQ( installed.weights_path,
            store.blobPath( sha256Hex( kWeightsPayload.data(), kWeightsPayload.size() ) ) );

        ASSERT_EQ( store.list().size(), 1u );
        EXPECT_EQ( store.list().front().record.name, "gemma-4-12b-it-fp4" );
        EXPECT_TRUE( store.locate( "gemma-4-12b-it-fp4" ).has_value() );

        // The record is one file at one level: the name is the key.
        EXPECT_TRUE( std::filesystem::exists(
            store_root.path() / "models" / "gemma-4-12b-it-fp4.json" ) );
    }

    TEST( ModelStoreInstallTests, AMoveLeavesOneCopyAndKeepLeavesTwo )
    {
        ScratchDirectory moved( "package" );
        ScratchDirectory kept( "package" );
        ScratchDirectory store_root( "store" );

        ModelStore store( store_root.path() );

        const auto moved_package = buildScratchPackage( moved.path(), "moved-bf16" );

        store.install( moved_package );

        // A move is free on one volume and leaves one copy of a file that may be gigabytes.
        EXPECT_FALSE( std::filesystem::exists(
            moved_package.directory() / "model.safetensors" ) );

        // Different bytes on purpose: an identical payload would already be in the store, and
        // adoption would skip the file rather than keep it.
        const auto kept_package =
            buildScratchPackage( kept.path(), "kept-bf16", true, "entirely different weights" );

        InstallOptions keep_options;
        keep_options.move_files = false;
        store.install( kept_package, keep_options );

        // Kept because the same directory still has to be uploaded.
        EXPECT_TRUE( std::filesystem::exists(
            kept_package.directory() / "model.safetensors" ) );
        EXPECT_TRUE( kept_package.validate().ok() );
    }

    TEST( ModelStoreInstallTests, TwoModelsShareOneTokenizerBlob )
    {
        ScratchDirectory first( "package" );
        ScratchDirectory second( "package" );
        ScratchDirectory store_root( "store" );

        ModelStore store( store_root.path() );

        // Same tokenizer, different weights: exactly the FP4/FP8 pair of one model, which are
        // two names now. Sharing is content addressing's doing, not the naming's.
        const auto fp4_package = buildScratchPackage( first.path(), "gemma-4-12b-it-fp4" );
        const auto fp8_package =
            buildScratchPackage( second.path(), "gemma-4-12b-it-fp8", true, "fp8 weight bytes" );

        const auto fp4 = store.install( fp4_package );
        const auto fp8 = store.install( fp8_package );

        EXPECT_EQ( fp4.tokenizer_path, fp8.tokenizer_path );
        EXPECT_NE( fp4.weights_path, fp8.weights_path );
        EXPECT_TRUE( fp8.complete );

        // Removing one must not take the blob the other still names.
        const auto report = store.remove( "gemma-4-12b-it-fp4" );

        EXPECT_EQ( report.records_removed, 1 );
        EXPECT_TRUE( std::filesystem::exists( fp8.tokenizer_path ) );
        EXPECT_TRUE( store.locate( "gemma-4-12b-it-fp8" ).has_value() );
    }

    TEST( ModelStoreInstallTests, RefusesASecondModelUnderAnInstalledName )
    {
        ScratchDirectory first( "package" );
        ScratchDirectory second( "package" );
        ScratchDirectory store_root( "store" );

        ModelStore store( store_root.path() );

        store.install( buildScratchPackage( first.path(), "gemma-4-12b-it-fp4" ) );

        // One name is one model. A second thing under that name is the state the flat layout
        // exists to make impossible, so it is refused rather than namespaced away.
        const auto other =
            buildScratchPackage( second.path(), "gemma-4-12b-it-fp4", true, "different bytes" );

        EXPECT_THROW( store.install( other ), std::runtime_error );

        InstallOptions replace;
        replace.replace = true;

        EXPECT_NO_THROW( store.install( other, replace ) );
        EXPECT_EQ( store.list().size(), 1u );
    }

    TEST( ModelStoreInstallTests, RefusesAFileThatDoesNotMatchItsDeclaredDigest )
    {
        ScratchDirectory scratch( "package" );
        ScratchDirectory store_root( "store" );

        const auto package = buildScratchPackage( scratch.path() );

        writeWholeFile( package.directory() / "model.safetensors", "tampered with entirely" );

        ModelStore store( store_root.path() );

        EXPECT_THROW( store.install( package ), std::runtime_error );

        // Adoption verifies before it publishes, so nothing that looks installed is left
        // behind, and the caller's file is not moved or quarantined.
        EXPECT_TRUE( store.list().empty() );
        EXPECT_TRUE( std::filesystem::exists( package.directory() / "model.safetensors" ) );
    }

    TEST( ModelStoreInstallTests, RenamesAModelWithoutMovingItsBytes )
    {
        ScratchDirectory scratch( "package" );
        ScratchDirectory store_root( "store" );

        ModelStore store( store_root.path() );

        const auto installed = store.install( buildScratchPackage( scratch.path() ) );
        const auto weights_before = installed.weights_path;

        ASSERT_TRUE( store.rename( "gemma-4-12b-it-fp4", "gemma-4-12b-it" ) );

        EXPECT_FALSE( store.locate( "gemma-4-12b-it-fp4" ).has_value() );

        const auto renamed = store.locate( "gemma-4-12b-it" );
        ASSERT_TRUE( renamed.has_value() );

        // The blobs are content-addressed, so the name has no bearing on where the bytes live.
        EXPECT_EQ( renamed->weights_path, weights_before );
        EXPECT_TRUE( std::filesystem::exists( weights_before ) );

        // Renaming is not reinstalling, so the install time carries over.
        EXPECT_EQ( renamed->record.installed_at, installed.record.installed_at );

        EXPECT_EQ( store.list().size(), 1u );
        EXPECT_FALSE( store.rename( "not-installed", "whatever" ) );
    }

    TEST( ModelStoreInstallTests, RefusesToRenameOntoAnInstalledName )
    {
        ScratchDirectory first( "package" );
        ScratchDirectory second( "package" );
        ScratchDirectory store_root( "store" );

        ModelStore store( store_root.path() );

        store.install( buildScratchPackage( first.path(), "gemma-4-12b-it-fp4" ) );
        store.install( buildScratchPackage(
            second.path(), "gemma-4-12b-it-fp8", true, "fp8 weight bytes" ) );

        // One name is one model, whether it is reached by installing or by renaming.
        EXPECT_THROW( store.rename( "gemma-4-12b-it-fp8", "gemma-4-12b-it-fp4" ),
            std::runtime_error );
        EXPECT_THROW( store.rename( "gemma-4-12b-it-fp8", "not a name" ), std::runtime_error );

        EXPECT_EQ( store.list().size(), 2u );
    }

    TEST( ModelStoreInstallTests, RefusesANameThatIsNotUsable )
    {
        ScratchDirectory scratch( "package" );
        ScratchDirectory store_root( "store" );

        const auto package = buildScratchPackage( scratch.path() );

        ModelStore store( store_root.path() );

        InstallOptions options;
        options.name = "not a model name";

        // The name is the record's filename, so anything path-shaped would escape the store.
        EXPECT_THROW( store.install( package, options ), std::runtime_error );
        EXPECT_TRUE( store.list().empty() );
    }
}
