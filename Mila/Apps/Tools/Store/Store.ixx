/**
 * @file Store.ixx
 * @brief The `mila-store` command: installs, lists and removes the models in the local store.
 *
 * Mila usage lives here rather than in main.cpp, matching ExportArtifact and ProfileModel.
 */

module;
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>

export module Mila.Store;

import Mila;

namespace Mila::StoreApp
{
    namespace
    {
        /**
         * @brief Render a byte count in the binary units it is actually divided by.
         *
         * Both divisions are powers of 1024, so labelling them MB and GB reported every size
         * 5% and 7% low against a figure in those units -- and this is what `models` shows for a
         * download the hub states in the same units.
         */
        std::string formatBytes( std::uint64_t bytes )
        {
            constexpr double kMebibyte = 1024.0 * 1024.0;
            constexpr double kGibibyte = kMebibyte * 1024.0;

            if ( bytes >= static_cast<std::uint64_t>( kGibibyte ) )
            {
                return std::format( "{:.2f} GiB", static_cast<double>( bytes ) / kGibibyte );
            }

            return std::format( "{:.1f} MiB", static_cast<double>( bytes ) / kMebibyte );
        }

        /**
         * A directory holding a manifest is a package someone handed the user; anything else is a
         * published name. A directory without one is neither, and saying which it is beats a hub
         * lookup for a name that was never meant as one.
         */
        enum class InstallSource
        {
            Published,
            Package,
            NotAPackage
        };

        InstallSource classify( std::string_view argument )
        {
            std::error_code ignored;
            const std::filesystem::path path( argument );

            if ( !std::filesystem::is_directory( path, ignored ) )
            {
                return InstallSource::Published;
            }

            return std::filesystem::exists( path / Mila::Distribution::kManifestFileName, ignored )
                ? InstallSource::Package
                : InstallSource::NotAPackage;
        }

        void printInstalled( const Mila::Distribution::StoredModel& installed )
        {
            std::cout << std::format( "Installed {} ({}, {}).\n",
                installed.record.name,
                installed.record.architecture.empty()
                    ? "unknown architecture" : installed.record.architecture,
                formatBytes( installed.bytes_on_disk ) );

            // Said at the one moment the user is certain to be reading. The identifier only -- the
            // text is published with the model, which is the copy that actually governs.
            if ( !installed.record.license.empty() )
            {
                std::cout << std::format(
                    "License: {}. The terms are published with the model.\n",
                    installed.record.license );
            }
        }

        /**
         * Copies rather than moves: the directory is the user's, and a store that emptied it would
         * delete what they may still mean to keep or pass on.
         */
        void installPackage( Mila::Distribution::ModelStore& store, std::string_view directory )
        {
            const auto package = Mila::Distribution::ModelPackage::open( std::filesystem::path( directory ) );

            std::cout << std::format( "Installing {} from {} ...\n",
                package.manifest().name, directory ) << std::flush;

            Mila::Distribution::InstallOptions options;
            options.move_files = false;

            printInstalled( store.install( package, options ) );
        }

        int installModels( std::span<const std::string_view> arguments )
        {
            Mila::Distribution::ModelStore store;

            std::uint64_t reported_total = 0;
            int last_percent = -1;

            // One redraw per whole percent. Unthrottled this fires on every chunk, which at
            // several GB is tens of thousands of writes to a line with 101 distinct states.
            auto progress = [&reported_total, &last_percent](
                std::uint64_t received, std::uint64_t total ) -> bool
                {
                    // A manifest is a few hundred bytes, so a bar for it renders as
                    // "83%  0.0 MB / 0.0 MB" and then vanishes -- noise that reads like a
                    // glitch. Below a megabyte there is nothing to wait for.
                    if ( total < 1'000'000 )
                    {
                        return true;
                    }

                    if ( total != reported_total )
                    {
                        reported_total = total;
                        last_percent = -1;
                    }

                    const int percent = static_cast<int>( ( received * 100 ) / total );

                    if ( percent == last_percent && received < total )
                    {
                        return true;
                    }

                    last_percent = percent;

                    // Erase to end of line rather than padding: padding writes past the content,
                    // so a narrow terminal wraps and \r then returns to the last visual row only.
                    std::cout << std::format( "\r  {:3}%  {} / {}",
                        percent, formatBytes( received ), formatBytes( total ) )
                        << "\x1b[K" << std::flush;

                    return true;
                };

            int failures = 0;

            for ( const auto& argument : arguments )
            {
                try
                {
                    switch ( classify( argument ) )
                    {
                        case InstallSource::Package:
                            installPackage( store, argument );
                            continue;

                        case InstallSource::NotAPackage:
                            std::cerr << std::format(
                                "Could not install '{}': it is a directory with no {} in it, so it is not a "
                                "model package.\n", argument, Mila::Distribution::kManifestFileName );
                            ++failures;
                            continue;

                        case InstallSource::Published:
                            break;
                    }

                    if constexpr ( !Mila::Distribution::kHttpTransportAvailable )
                    {
                        std::cerr << std::format( "Could not install '{}': this build has no HTTP transport "
                            "(MILA_ENABLE_LIBCURL=OFF), so it installs only from a package directory.\n",
                            argument );
                        ++failures;
                        continue;
                    }

                    const auto hub = Mila::Distribution::makeDefaultModelHub( progress );

                    Mila::Distribution::ModelResolver resolver( store, *hub );

                    // Says what is happening BEFORE the bar appears. Without it the first thing
                    // a user sees is a bare "16%  469.0 MB / 2.86 GB" with nothing naming what is
                    // being fetched or from where -- and in the container that lands straight
                    // after Docker's own pull narrative, reading as a continuation of it.
                    std::cout << std::format( "Installing {} from {} ...\n",
                        argument, Mila::Distribution::kDefaultHubOwner ) << std::flush;

                    const auto pulled = resolver.pull(
                        std::string( argument ),
                        std::string( Mila::Distribution::kDefaultHubOwner ) );

                    if ( last_percent >= 0 )
                    {
                        std::cout << "\n";
                        last_percent = -1;
                    }

                    printInstalled( pulled );
                }
                catch ( const std::exception& error )
                {
                    if ( last_percent >= 0 )
                    {
                        std::cout << "\n";
                        last_percent = -1;
                    }

                    std::cerr << std::format( "Could not install '{}': {}\n", argument, error.what() );

                    ++failures;
                }
            }

            return failures == 0 ? 0 : 1;
        }

        int listInstalled()
        {
            const auto models = Mila::Distribution::ModelStore{}.list();

            if ( models.empty() )
            {
                std::cout << "No models installed. 'mila-store models --online' lists what is "
                    "published, and 'mila-store install <name>' installs one.\n";

                return 0;
            }

            for ( const auto& model : models )
            {
                std::cout << std::format( "{:<28}  {:>10}{}\n",
                    model.record.name,
                    formatBytes( model.bytes_on_disk ),
                    model.complete ? "" : "  (incomplete)" );
            }

            return 0;
        }

        int listPublished()
        {
            if constexpr ( !Mila::Distribution::kHttpTransportAvailable )
            {
                std::cerr << "This build has no HTTP transport (MILA_ENABLE_LIBCURL=OFF), "
                    "so it cannot reach the hub.\n";

                return 2;
            }

            const auto hub = Mila::Distribution::makeDefaultModelHub();

            const auto published = hub->listModels(
                std::string( Mila::Distribution::kDefaultHubOwner ) );

            int loadable = 0;

            for ( const auto& model : published )
            {
                // A repository without a manifest is not something this runtime can load, and
                // listing it as available would be a lie.
                if ( !model.hasManifest() )
                {
                    continue;
                }

                std::cout << std::format( "{:<28}{}\n",
                    model.repository, model.gated ? "  (gated)" : "" );

                ++loadable;
            }

            if ( loadable == 0 )
            {
                std::cout << std::format( "Nothing published under '{}' carries a Mila manifest.\n",
                    Mila::Distribution::kDefaultHubOwner );
            }

            return 0;
        }

        int removeModels( std::span<const std::string_view> names )
        {
            Mila::Distribution::ModelStore store;

            int failures = 0;

            for ( const auto& requested : names )
            {
                // The record, not locate(): locate hides an incomplete model, and an incomplete model
                // is the one a user most needs to remove. The record also carries the store's own
                // spelling of the name.
                const auto record = store.readRecord( std::string( requested ) );

                if ( !record.has_value() )
                {
                    std::cerr << std::format( "{} is not installed.\n", requested );
                    ++failures;

                    continue;
                }

                const std::string& name = record->name;
                const auto report = store.remove( name );

                if ( report.records_removed == 0 )
                {
                    std::cerr << std::format( "{} was not removed: some of its files are in use. Close "
                        "whatever has it loaded, then remove it again.\n", name );
                    ++failures;
                } else {
                    std::cout << std::format( "Removed {} ({} reclaimed).\n",
                        name, formatBytes( report.bytes_reclaimed ) );
                }

                if ( !report.unreadable_records.empty() )
                {
                    std::cout << "  Its files were kept, because these records cannot be read:\n";

                    for ( const auto& unreadable : report.unreadable_records )
                    {
                        std::cout << std::format( "    {}\n", unreadable );
                    }
                } else if ( report.records_removed == 1 && report.blobs_removed == 0 ) {
                    // Shared files survive; saying so pre-empts "why did that free so little".
                    std::cout << "  Its files are shared with another installed model, so none were deleted.\n";
                }

                for ( const auto& retained : report.retained )
                {
                    std::cerr << std::format( "  Could not delete (in use?): {}\n", retained );
                }
            }

            return failures == 0 ? 0 : 1;
        }

        int reportUsage( bool clean )
        {
            Mila::Distribution::ModelStore store;

            if ( clean )
            {
                // Asked for by name, so an interrupted download goes too: the user wants the space,
                // and the next install of that model starts again from the beginning.
                Mila::Distribution::CleanOptions options;
                options.discard_partials = true;

                const auto report = store.clean( options );

                std::cout << std::format( "Cleaned {} file(s), {} reclaimed.\n",
                    report.files_removed, formatBytes( report.bytes_reclaimed ) );

                for ( const auto& retained : report.retained )
                {
                    std::cerr << std::format( "  Could not delete (in use?): {}\n", retained );
                }
            }

            const auto usage = store.usage();

            std::cout << std::format( "Store            {}\n", store.root().string() );
            std::cout << std::format( "Models           {}\n", usage.model_count );
            std::cout << std::format( "On disk          {}\n", formatBytes( usage.blob_bytes ) );

            if ( usage.partial_bytes > 0 )
            {
                std::cout << std::format( "Interrupted      {}  (an install of the same model resumes it)\n",
                    formatBytes( usage.partial_bytes ) );
            }

            if ( usage.reclaimable_bytes > 0 )
            {
                std::cout << std::format( "Reclaimable      {}\n", formatBytes( usage.reclaimable_bytes ) );
            }

            if ( !clean && usage.partial_bytes + usage.reclaimable_bytes > 0 )
            {
                std::cout << "'mila-store usage --clean' frees both.\n";
            }

            return 0;
        }

        void printUsage()
        {
            std::cout <<
                "mila-store -- install and manage the models Mila runs.\n"
                "\n"
                "  mila-store install <name>...      install a published model\n"
                "  mila-store install <directory>    install a model package from disk (copied, not moved)\n"
                "  mila-store models                 list what is installed\n"
                "  mila-store models --online        list what is published\n"
                "  mila-store remove <name>...       remove an installed model\n"
                "  mila-store usage [--clean]        show disk use; --clean frees interrupted downloads\n"
                "  mila-store --version              show the version\n"
                "\n"
                "The store is MILA_CACHE_DIR when set, and the per-user cache otherwise.\n"
                "Run a model with mila-chat, or serve it with mila-server.\n";
        }

        int refuseOperands( std::string_view verb, std::span<const std::string_view> rest )
        {
            std::cerr << std::format( "mila-store {}: unexpected '{}'. Try 'mila-store help'.\n",
                verb, rest.front() );

            return 2;
        }
    }

    /**
     * @brief Dispatch one command. argv[0] is expected at the front, as main receives it.
     */
    export int run( std::span<const std::string_view> arguments )
    {
        const auto operands = arguments.subspan( arguments.empty() ? 0 : 1 );

        if ( operands.empty() )
        {
            printUsage();

            return 0;
        }

        const std::string_view verb = operands.front();
        const auto rest = operands.subspan( 1 );

        try
        {
            if ( verb == "install" )
            {
                if ( rest.empty() )
                {
                    std::cerr << "usage: mila-store install <name | directory>...\n";

                    return 2;
                }

                return installModels( rest );
            }

            if ( verb == "models" )
            {
                if ( rest.empty() )
                {
                    return listInstalled();
                }

                if ( rest.size() == 1 && rest.front() == "--online" )
                {
                    return listPublished();
                }

                return refuseOperands( verb, rest.front() == "--online" ? rest.subspan( 1 ) : rest );
            }

            if ( verb == "remove" )
            {
                if ( rest.empty() )
                {
                    std::cerr << "usage: mila-store remove <name>...\n";

                    return 2;
                }

                return removeModels( rest );
            }

            if ( verb == "usage" )
            {
                if ( rest.empty() )
                {
                    return reportUsage( false );
                }

                if ( rest.size() == 1 && rest.front() == "--clean" )
                {
                    return reportUsage( true );
                }

                return refuseOperands( verb, rest.front() == "--clean" ? rest.subspan( 1 ) : rest );
            }

            if ( verb == "--version" || verb == "version" )
            {
                std::cout << std::format( "mila-store {}\n", Mila::getAPIVersion().toString() );

                return 0;
            }

            if ( verb == "help" || verb == "--help" || verb == "-h" )
            {
                printUsage();

                return 0;
            }
        }
        catch ( const std::exception& error )
        {
            std::cerr << std::format( "mila-store: {}\n", error.what() );

            return 1;
        }

        std::cerr << std::format( "mila-store: unknown command '{}'. Try 'mila-store help'.\n", verb );

        return 2;
    }
}
