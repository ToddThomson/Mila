/**
 * @file Cli.ixx
 * @brief The `mila` command: model store management, and a front door to the server.
 *
 * Mila usage lives here rather than in main.cpp, matching ExportArtifact and ProfileModel.
 */

module;
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

export module Tools.Cli;

import Mila;

namespace Mila::Tools::Cli
{
    namespace
    {
        /**
         * Where this installation lives, resolved once at startup: the image directory main.cpp
         * found, unless MILA_APP_DIR overrides it for a layout that splits the parts. Only the
         * virtual environment is found through it now.
         *
         * The working directory is the last resort and not the answer -- what is found here gets
         * executed, so guessing runs whatever sits where the user happened to be.
         */
        std::filesystem::path application_directory;

        std::filesystem::path applicationDirectory()
        {
            if ( const char* configured = std::getenv( "MILA_APP_DIR" ) )
            {
                return std::filesystem::path( configured );
            }

            if ( !application_directory.empty() )
            {
                return application_directory;
            }

            return std::filesystem::current_path();
        }

        std::filesystem::path virtualEnvironmentDirectory()
        {
            if ( const char* configured = std::getenv( "MILA_VENV_DIR" ) )
            {
                return std::filesystem::path( configured );
            }

            return applicationDirectory() / "venv";
        }

        std::string formatBytes( std::uint64_t bytes )
        {
            constexpr double kMegabyte = 1024.0 * 1024.0;
            constexpr double kGigabyte = kMegabyte * 1024.0;

            if ( bytes >= static_cast<std::uint64_t>( kGigabyte ) )
            {
                return std::format( "{:.2f} GB", static_cast<double>( bytes ) / kGigabyte );
            }

            return std::format( "{:.1f} MB", static_cast<double>( bytes ) / kMegabyte );
        }

        /**
         * Run the server, replacing nothing: it is long-lived and wants the terminal this
         * process already owns. std::system keeps that behaviour on both platforms without a
         * preprocessor branch.
         */
        int runProgram( const std::filesystem::path& program,
            std::span<const std::string_view> arguments )
        {
            std::error_code ignored;

            if ( !std::filesystem::exists( program, ignored ) )
            {
                std::cerr << std::format(
                    "'{}' is not part of this build ({} is missing).\n",
                    program.filename().string(), program.string() );

                return 2;
            }

            std::string command = "\"" + program.string() + "\"";

            for ( const auto& argument : arguments )
            {
                command += " \"";
                command += argument;
                command += "\"";
            }

            // The launched program resolves its own data against its image directory, so this
            // leaves the working directory alone: it is the user's, and a relative path they
            // pass on the command line has to keep meaning what they typed.
            return std::system( command.c_str() );
        }

        int installModels( std::span<const std::string_view> names )
        {
            if constexpr ( !Mila::Distribution::kHttpTransportAvailable )
            {
                std::cerr << "This build has no HTTP transport (MILA_ENABLE_LIBCURL=OFF), "
                    "so it cannot install models.\n";

                return 2;
            }

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

            const auto hub = Mila::Distribution::makeDefaultModelHub( progress );

            Mila::Distribution::ModelResolver resolver( store, *hub );

            int failures = 0;

            for ( const auto& name : names )
            {
                try
                {
                    // Says what is happening BEFORE the bar appears. Without it the first thing
                    // a user sees is a bare "16%  469.0 MB / 2.86 GB" with nothing naming what is
                    // being fetched or from where -- and in the container that lands straight
                    // after Docker's own pull narrative, reading as a continuation of it.
                    std::cout << std::format( "Installing {} from {} ...\n",
                        name, Mila::Distribution::kDefaultHubOwner ) << std::flush;

                    const auto pulled = resolver.pull(
                        std::string( name ),
                        std::string( Mila::Distribution::kDefaultHubOwner ) );

                    if ( last_percent >= 0 )
                    {
                        std::cout << "\n";
                        last_percent = -1;
                    }

                    std::cout << std::format( "Installed {} ({}, {}).\n",
                        pulled.record.name,
                        pulled.record.architecture.empty()
                            ? "unknown architecture" : pulled.record.architecture,
                        formatBytes( pulled.bytes_on_disk ) );

                    // Said at the one moment the user is certain to be reading. The identifier
                    // only -- the text is published with the model, which is the copy that
                    // actually governs.
                    if ( !pulled.record.license.empty() )
                    {
                        std::cout << std::format(
                            "License: {}. The terms are published with the model.\n",
                            pulled.record.license );
                    }
                }
                catch ( const std::exception& error )
                {
                    if ( last_percent >= 0 )
                    {
                        std::cout << "\n";
                        last_percent = -1;
                    }

                    std::cerr << std::format( "Could not install '{}': {}\n", name, error.what() );

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
                std::cout << "No models installed. 'mila models --online' lists what is "
                    "published, and 'mila install <name>' installs one.\n";

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

        void printUsage()
        {
            std::cout <<
                "mila -- the Mila command line.\n"
                "\n"
                "  mila install <name>...   install a published model into the store\n"
                "  mila models              list what is installed\n"
                "  mila models --online     list what is published\n"
                "  mila serve [args]        start the inference server\n"
                "\n"
                "The store is MILA_CACHE_DIR when set, and the per-user cache otherwise.\n"
                "\n"
                // Named because its absence here is deliberate: mila-chat is a binary beside
                // this one and needs no front door, while the server is a console script in a
                // virtual environment and does.
                "The chat harness is its own command: run mila-chat.\n";
        }
    }

    /**
     * @brief Dispatch one command. argv[0] is expected at the front, as main receives it.
     *
     * @param image_directory Where this executable lives, which is where its virtual
     *        environment sits. Resolving it is per-platform, so main.cpp does it and passes the
     *        answer in. Empty is accepted and means unavailable.
     */
    export int run( std::span<const std::string_view> arguments,
        const std::filesystem::path& image_directory = {} )
    {
        application_directory = image_directory;

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
                    std::cerr << "usage: mila install <name>...\n";

                    return 2;
                }

                return installModels( rest );
            }

            if ( verb == "models" )
            {
                const bool online = !rest.empty() && rest.front() == "--online";

                return online ? listPublished() : listInstalled();
            }

            if ( verb == "serve" )
            {
                // A venv puts console scripts in bin/ on POSIX and Scripts/ on Windows. Probed
                // at run time rather than branched at compile time, so this stays one unit.
                const auto venv = virtualEnvironmentDirectory();

                std::error_code ignored;
                const auto posix_server = venv / "bin" / "mila-server";

                if ( std::filesystem::exists( posix_server, ignored ) )
                {
                    return runProgram( posix_server, rest );
                }

                return runProgram( venv / "Scripts" / "mila-server.exe", rest );
            }

            if ( verb == "help" || verb == "--help" || verb == "-h" )
            {
                printUsage();

                return 0;
            }
        }
        catch ( const std::exception& error )
        {
            std::cerr << std::format( "mila: {}\n", error.what() );

            return 1;
        }

        std::cerr << std::format( "mila: unknown command '{}'. Try 'mila help'.\n", verb );

        return 2;
    }
}
