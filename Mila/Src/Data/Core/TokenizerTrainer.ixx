/*!
 * \file
 * \brief Abstract trainer interface for building tokenizers' vocabularies.
 *
 * Provides the public API used by tokenizer trainers (BPE, char, etc.)
 * to accumulate text, run training, and produce a TokenizerVocabulary.
 */

module;
#include <string>
#include <memory>

export module Data.TokenizerTrainer;

import Data.TokenizerVocabulary;

namespace Mila::Data
{
    using Mila::Dnn::Data::TokenizerVocabulary;

    /**
     * @brief Abstract interface for tokenizer training implementations.
     *
     * TokenizerTrainer defines the minimal lifecycle required to construct
     * a tokenizer vocabulary from a text corpus:
     *  - accumulate corpus data via addCorpus()
     *  - run training via train()
     *  - create a TokenizerVocabulary via buildVocabulary()
     *
     * Implementations are expected to own any intermediate training state.
     * This class is non-copyable; ownership of the produced vocabulary is
     * transferred to the caller using std::unique_ptr.
     */
    export class TokenizerTrainer {
    public:

        /**
         * @brief Virtual destructor.
         *
         * Ensures derived trainers are cleaned up correctly through base pointers.
         */
        virtual ~TokenizerTrainer() = default;

        /**
         * @brief Add text to the trainer's corpus.
         *
         * The provided text is incorporated into the internal corpus/state
         * used by the trainer. Callers may invoke this multiple times to
         * accumulate data from different sources.
         *
         * @param text UTF-8 encoded text to include in the training corpus.
         */
        virtual void addCorpus( const std::string& text ) = 0;

        /**
         * @brief Execute the training procedure.
         *
         * After corpus accumulation, this method performs the algorithm-specific
         * training steps (e.g., BPE merges, character frequency analysis).
         * Implementations should prepare internal structures so that a subsequent
         * call to buildVocabulary() returns a valid vocabulary.
         */
        virtual void train() = 0;

        /**
         * @brief Construct and return the trained tokenizer vocabulary.
         *
         * Transfers ownership of the resulting TokenizerVocabulary to the caller.
         * The returned vocabulary represents the final token set produced by the
         * trainer after train() has completed.
         *
         * @return std::unique_ptr<TokenizerVocabulary> Owned pointer to the trained vocabulary.
         */
        virtual std::unique_ptr<TokenizerVocabulary> buildVocabulary() = 0;
    };
}