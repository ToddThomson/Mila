/**
 * \file
 * \brief Abstract trainer interface for building tokenizers' vocabularies.
 *
 * Defines the minimal lifecycle for constructing a tokenizer vocabulary
 * from a text corpus. Tokenizer Trainers perform training and return the resulting
 * vocabulary object; ownership is transferred to the caller.
 */

module;
#include <string>
#include <memory>
#include <filesystem>
#include <istream>
#include <fstream>
#include <sstream>

export module Data.TokenizerTrainer;

import Data.TokenizerVocabulary;

namespace Mila::Data
{
    using Mila::Dnn::Data::TokenizerVocabulary;

    /**
     * @brief Abstract interface for training tokenizer vocabularies from text corpora.
     *
     * TokenizerTrainer provides a generic contract for building tokenizer vocabularies
     * by processing text data from various sources (streams, files, or strings) and
     * producing a serialized vocabulary file suitable for later use in tokenization.
     *
     * Typical workflow:
     * 1. Create a concrete trainer instance (e.g., BpeTokenizerTrainer, CharTokenizerTrainer)
     * 2. Add corpus data via addCorpusFromStream(), addCorpusFromFile(), or addCorpus()
     * 3. Call train() to build and save the vocabulary to disk
     * 4. Load the saved vocabulary using the appropriate TokenizerVocabulary implementation
     *
     * Thread safety:
     * - Implementations are NOT required to be thread-safe
     * - Callers must ensure external synchronization if corpus addition or training
     *   occurs concurrently
     *
     * Design rationale:
     * - Training is typically an offline, one-time process; vocabularies are saved to
     *   disk and loaded for inference rather than kept in memory
     * - Stream-based corpus addition enables memory-efficient processing of large datasets
     * - Abstract interface allows for different tokenization algorithms (BPE, character-level,
     *   WordPiece, etc.) with a consistent API
     *
     * @see TokenizerVocabulary
     */
    export class TokenizerTrainer {
    public:
        /**
         * @brief Virtual destructor.
         *
         * Ensures derived trainer implementations are properly destroyed when
         * deleted via base pointers.
         */
        virtual ~TokenizerTrainer() = default;

        /**
         * @brief Add training corpus data from an input stream.
         *
         * This is the primary corpus addition method that all implementations must provide.
         * The stream is read incrementally to support memory-efficient processing of large
         * corpora. Text is expected to be UTF-8 encoded unless documented otherwise by
         * the concrete implementation.
         *
         * Multiple calls to corpus addition methods are cumulative; all provided text
         * contributes to the final trained vocabulary.
         *
         * Implementation requirements:
         * - Must process the stream incrementally without requiring the entire corpus in memory
         * - Should handle stream read failures gracefully (may throw exceptions)
         * - Must support binary and text mode streams
         * - Should document any encoding assumptions beyond UTF-8
         *
         * Behavior on error:
         * - Implementations may throw exceptions (e.g., std::runtime_error, std::ios_base::failure)
         *   on stream read failures or encoding errors
         * - After an exception, the trainer state is implementation-defined; callers should
         *   typically discard the trainer instance
         *
         * @param stream Input stream containing UTF-8 encoded text corpus.
         *               The stream is not closed by this method.
         *
         * @throws std::runtime_error or derived exceptions on stream read errors (implementation-defined)
         *
         * @note The stream position after this call is implementation-defined but typically
         *       will be at end-of-stream if the entire stream was consumed.
         */
        virtual void addCorpusFromStream( std::istream& stream ) = 0;

        /**
         * @brief Add training corpus data from a file.
         *
         * Convenience method that opens the specified file and delegates to
         * addCorpusFromStream(). The file is opened in binary mode to preserve
         * exact byte sequences for UTF-8 text.
         *
         * Multiple calls to corpus addition methods are cumulative; all provided text
         * contributes to the final trained vocabulary.
         *
         * Behavior on error:
         * - Throws std::runtime_error if the file cannot be opened
         * - May throw other exceptions from addCorpusFromStream() during processing
         *
         * @param filepath Path to a UTF-8 encoded text file.
         *
         * @throws std::runtime_error if the file cannot be opened for reading
         * @throws Other exceptions as documented by addCorpusFromStream()
         */
        void addCorpusFromFile( std::string_view filepath ) {
            std::ifstream file( filepath.data(), std::ios::binary );
            
            if ( !file ) {
                throw std::runtime_error(
                    std::format( "TokenizerTrainer: Cannot open file: {}", filepath )
                );
            }
            
            addCorpusFromStream( file );
        }

        /**
         * @brief Add training corpus data from a string.
         *
         * Convenience method for adding small text samples, useful for testing or
         * supplementing file-based corpora. For large corpora, prefer addCorpusFromStream()
         * or addCorpusFromFile() for better memory efficiency.
         *
         * The input text is expected to be UTF-8 encoded.
         *
         * Multiple calls to corpus addition methods are cumulative; all provided text
         * contributes to the final trained vocabulary.
         *
         * Behavior on error:
         * - May throw exceptions from addCorpusFromStream() during processing
         *
         * @param text UTF-8 encoded text to add to the training corpus.
         *
         * @throws Exceptions as documented by addCorpusFromStream()
         *
         * @note This method creates a temporary std::istringstream; for large strings
         *       (>1MB), consider using addCorpusFromStream() with a pre-constructed
         *       stream for better performance.
         */
        void addCorpus( std::string_view text ) {
            std::istringstream text_stream{ std::string( text ) };
            
            addCorpusFromStream( text_stream );
        }

        /**
         * @brief Train the tokenizer and return the resulting vocabulary.
         *
         * Analyzes all previously added corpus data to build a vocabulary according to
         * the concrete implementation's algorithm (e.g., BPE merges, character collection).
         * The returned vocabulary is ready for immediate use in tokenization.
         *
         * The vocabulary can be saved to disk using its save() method for later reuse,
         * and can be shared among multiple Tokenizer instances.
         *
         * @return std::shared_ptr<TokenizerVocabulary> The trained vocabulary.
         *         Ownership is shared; the vocabulary can be used by multiple tokenizers
         *         or saved for later use.
         *
         * @throws std::runtime_error if training fails
         * @throws std::invalid_argument if insufficient corpus data was provided
         *
         * Example usage:
         * @code
         * BpeTokenizerTrainer trainer(vocab_size);
         * trainer.addCorpusFromFile("corpus.txt");
         * auto vocab = trainer.train();
         *
         * // Use immediately
         * BpeTokenizer tokenizer(vocab);
         * auto tokens = tokenizer.encode("hello world");
         *
         * // Save for later
         * vocab->save("vocab.bin");
         * @endcode
         */
        virtual std::shared_ptr<TokenizerVocabulary> train() = 0;
    };
}