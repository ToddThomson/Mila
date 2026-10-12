/**
 * @file ModelFile.ixx
 * @brief One file composing a model, as its manifest declares it.
 */

export module Distribution.ModelFile;

import std.compat;

namespace Mila::Distribution
{
    /**
     * @brief One file composing a model, as the manifest declares it.
     *
     * `path` is the name the file has at its origin, kept for diagnostics and for republishing; the
     * bytes live at the digest.
     */
    export struct ModelFile
    {
        std::string role;
        std::string path;
        std::string sha256;
        uint64_t bytes{ 0 };
    };
}
