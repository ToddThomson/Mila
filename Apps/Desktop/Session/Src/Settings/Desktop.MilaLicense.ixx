/**
 * @file Desktop.MilaLicense.ixx
 * @brief The licence Mila is used under, as the package carries it beside the session (AssistantOnboarding.md 4.1).
 */

export module Desktop.MilaLicense;

import std;

namespace Mila::Desktop
{
    /**
     * @brief The licence text a person reads and accepts before the session installs, loads or answers anything.
     *
     * An acceptance is of this text: its digest is what the settings keep, so a package that carries a different
     * text asks again.
     */
    export struct MilaLicense
    {
        std::string text;

        /// SHA-256 of the text's bytes, lowercase hex.
        std::string sha256;

        /**
         * @brief The licence in the file at @p path.
         *
         * @return The licence, or the sentence the person is shown when it cannot be read.
         */
        [[nodiscard]] static std::expected<MilaLicense, std::string> read( const std::filesystem::path& path );
    };
}
