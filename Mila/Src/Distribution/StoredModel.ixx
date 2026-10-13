/**
 * @file StoredModel.ixx
 * @brief An installed model's record together with where its bytes are.
 */

export module Distribution.StoredModel;

import std;
export import Distribution.ModelRecord;

namespace Mila::Distribution
{
    /**
     * @brief A record together with where its bytes actually are.
     *
     * `bytes_on_disk` sums the model's own files. A blob shared between variants is counted in
     * each, so the per-model figures do not sum to the store total.
     */
    export struct StoredModel
    {
        ModelRecord record;

        std::filesystem::path weights_path;
        std::filesystem::path tokenizer_path;

        /// Empty when the package carries no drafter.
        std::filesystem::path drafter_path;

        /// False when a declared file is missing from blobs/, which makes the model unloadable.
        bool complete{ false };

        std::uint64_t bytes_on_disk{ 0 };
    };
}
