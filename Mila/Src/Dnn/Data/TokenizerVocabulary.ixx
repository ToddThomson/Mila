/*!
 * \file TokenizerVocabulary.ixx
 * \brief Abstract interface for tokenizer vocabularies used by data pipelines.
 *
 * Defines the minimal API for mapping between token strings and numeric ids,
 * and for persisting/loading vocabulary contents.
 */

module;
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <filesystem>

export module Data.TokenizerVocabulary;

namespace Mila::Dnn::Data
{
    /**
     * @brief Generic tokenizer vocabulary interface.
     *
     * TokenizerVocabulary provides a small, implementation-agnostic contract
     * for converting between token strings and numeric ids and for
     * serializing/deserializing vocabulary state.
     *
     * Notes for implementers:
     * - Token strings are expected to be UTF-8 encoded; implementations must
     *   document any different encoding assumptions.
     * - The use of std::optional in lookup methods indicates a missing token
     *   or id. Callers should handle the empty optional as "not found".
     * - Thread-safety is implementation-defined; callers should synchronize
     *   concurrent modifications (e.g., load) and/or consult concrete docs.
     */
    export class TokenizerVocabulary {
    public:

        /**
         * @brief Virtual destructor.
         *
         * Ensures derived vocabulary implementations are properly destroyed
         * when deleted via base pointers.
         */
        virtual ~TokenizerVocabulary() = default;

        /**
         * @brief Get the number of tokens in the vocabulary.
         *
         * @return size_t Number of entries (tokens) present in the vocabulary.
         */
        virtual size_t getSize() const = 0;

        /**
         * @brief Map a token string to its numeric id.
         *
         * The lookup returns an empty optional when the token is not present in
         * the vocabulary. Implementations may provide an explicit unknown-token
         * id; callers should interpret an empty optional as "no mapping".
         *
         * @param token UTF-8 encoded token string to look up.
         * @return std::optional<uint32_t> The token id if present, otherwise empty.
         */
        virtual std::optional<uint32_t> tokenToId( const std::string& token ) const = 0;

        /**
         * @brief Map a numeric id back to its token string.
         *
         * Returns an empty optional if the id is out of range or not defined.
         *
         * @param id Token id to convert.
         * @return std::optional<std::string> The token string if present, otherwise empty.
         */
        virtual std::optional<std::string> idToToken( uint32_t id ) const = 0;

        /**
         * @brief Serialize the vocabulary to disk at the given path.
         *
         * Implementations should produce a deterministic on-disk representation
         * that can be consumed by the corresponding load() implementation.
         *
         * Behavior on error:
         * - Implementations may throw exceptions (e.g., std::runtime_error) on I/O
         *   or format errors. Callers should handle such exceptions as appropriate.
         *
         * @param path Filesystem path to write the vocabulary to.
         */
        virtual void save( const std::filesystem::path& path ) const = 0;

        /**
         * @brief Load vocabulary state from the given path.
         *
         * After a successful load(), subsequent lookups and getSize() must
         * reflect the loaded state. Implementations may replace the current
         * in-memory state or merge based on their documented behavior.
         *
         * Behavior on error:
         * - Implementations may throw exceptions (e.g., std::runtime_error) on I/O
         *   or format errors. Callers should handle such exceptions as appropriate.
         *
         * @param path Filesystem path to read the vocabulary from.
         */
        virtual void load( const std::filesystem::path& path ) = 0;
    };
}