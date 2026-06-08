/**
 * @file Version.ixx
 * @brief Semantic version type and Mila library version constants.
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2021..2026 Todd J. Thomson
 */

module;
#include <string>
#include <sstream>

export module Mila.Version;

namespace Mila
{
    /// <summary>
    /// Semantic Version data.
    /// </summary>
    export struct Version
    {
    public:

        /// <summary>
        /// Semantic Version Constructor
        /// </summary>
        /// <param name="major">Major API version</param>
        /// <param name="minor">Minor version for functional changes</param>
        /// <param name="patch">Patch for bug fixes</param>
        /// <param name="prerelease_tag">Optional pre-release text</param>
        Version( int major, int minor, int patch, const std::string& prerelease_tag = "")
            : major_( major ), minor_( minor ), patch_( patch ), pre_release_tag_( prerelease_tag )
        {
        }

        std::string toString() const
        {
            std::stringstream ss;

            ss << std::to_string( major_ )
                << "." << std::to_string( minor_ )
                << "." << std::to_string( patch_ );

            if (!pre_release_tag_.empty())
            {
                ss << "-" << pre_release_tag_;
            }

            return ss.str();
        }

        /// <summary>
        /// Gets the major version number.
        /// </summary>
        /// <returns>Major version</returns>
        int getMajor() { return major_; };

        /// <summary>
        /// Gets the minor version number.
        /// </summary>
        /// <returns>Minor version</returns>
        int getMinor() { return minor_; };

        /// <summary>
        /// Gets the patch version number.
        /// </summary>
        /// <returns>Patch version</returns>
        int getPatch() { return patch_; };

        /// <summary>
        /// Gets the pre-release tag.
        /// </summary>
        /// <returns>Optional pre-release tag</returns>
        std::string getPreRelease()
        {
            return pre_release_tag_;
        };

    private:

        int major_;
        int minor_;
        int patch_;
        std::string pre_release_tag_;
    };

    /// <summary>
    /// Gets the current Mila runtime API version.
    /// </summary>
    /// <returns>A Version object describing the running library.</returns>
    /// <remarks>
    /// The version components are injected by the build as PUBLIC compile
    /// definitions (see the Versioning block in Mila/CMakeLists.txt), sourced
    /// from Version.txt. find_package consumers recompile this module unit and
    /// pick the definitions up via the exported target's interface, so no
    /// version header is shipped.
    /// </remarks>
    export Version getAPIVersion()
    {
        return Version{
            MILA_VERSION_MAJOR,
            MILA_VERSION_MINOR,
            MILA_VERSION_PATCH,
            MILA_VERSION_PRERELEASE_TAG };
    }
}