/**
 * @file ExportArtifact.Fetch.Null.ixx
 * @brief The --fetch diagnostic for a build with no HTTP transport: it says so.
 *
 * One of two candidate files for Tools.ExportArtifact.Fetch; CMake compiles this one when
 * MILA_ENABLE_MODEL_HUB is OFF. The command diagnoses a transport, so without one it has
 * nothing to report -- every other verb of this tool is unaffected.
 */

module;
#include <filesystem>
#include <iostream>
#include <string>

export module Tools.ExportArtifact.Fetch;

namespace Mila::Tools
{
    /**
     * @brief Report that this build cannot fetch, and why.
     *
     * @return Process exit code.
     */
    export int runFetch( const std::string&, const std::filesystem::path& )
    {
        std::cerr <<
            "--fetch diagnoses Mila's HTTP client, and this build was compiled with "
            "MILA_ENABLE_MODEL_HUB=OFF, so there is none. Packaging, validation and "
            "installation are unaffected.\n";

        return 2;
    }
}
