/**
 * @file PackageValidation.ixx
 * @brief What checking a package against its own manifest found.
 */

module;
#include <cstdint>
#include <string>
#include <vector>

export module Distribution.PackageValidation;

namespace Mila::Distribution
{
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
}
