module;
#include <cstddef>
#include <string>
#include <stdexcept>

export module Data.BpeTrainerConfig;

import Data.BpeSpecialTokens;

namespace Mila::Data
{
    /*export struct BpeTrainerConfig {
        std::size_t vocab_size = 10000;
        std::size_t min_frequency = 2;
        bool add_prefix_space = false;
    };*/

    /**
    * @brief Configuration for Byte Pair Encoding (BPE) tokenizer training.
    *
    * BPE learns a vocabulary of subword units by iteratively merging the most
    * frequent adjacent token pairs. This balances vocabulary size against
    * sequence length and handles rare words through subword decomposition.
    *
    * Fluent interface allows chaining:
    * @code
    * auto config = BpeTrainerConfig()
    *     .withVocabSize(32000)
    *     .withMinFrequency(2)
    *     .withByteLevel(true)
    *     .withSpecialTokens(my_tokens);
    * @endcode
    */
    export class BpeTrainerConfig {
    public:
        BpeTrainerConfig() = default;

        /**
         * @brief Set target vocabulary size.
         *
         * The final vocabulary includes base tokens (256 bytes if byte-level,
         * or unique characters otherwise) plus learned merges up to this limit.
         *
         * Typical values:
         * - Small models: 8,000 - 16,000
         * - Medium models: 32,000 - 50,000
         * - Large models (GPT-3/4): 50,000 - 100,000
         *
         * @param size Target vocabulary size. Must be > base vocabulary size.
         * @return Reference to this config for method chaining.
         */
        BpeTrainerConfig& withVocabSize( size_t size ) {
            vocab_size_ = size;
            return *this;
        }

        /**
         * @brief Configure special tokens.
         *
         * @param tokens SpecialTokens configuration.
         * @return Reference to this config for method chaining.
         */
        BpeTrainerConfig& withSpecialTokens( const BpeSpecialTokens& tokens ) {
            special_tokens_ = tokens;
            return *this;
        }

        /**
         * @brief Set minimum frequency threshold for merges.
         *
         * Token pairs occurring fewer than this many times in the corpus
         * will not be merged. Higher values speed up training and can
         * improve generalization by avoiding overfitting to rare patterns.
         *
         * @param frequency Minimum occurrence count (default: 2).
         * @return Reference to this config for method chaining.
         */
        BpeTrainerConfig& withMinFrequency( size_t frequency ) {
            min_frequency_ = frequency;
            return *this;
        }

        /**
         * @brief Set whether to use byte-level BPE.
         *
         * Byte-level BPE operates on UTF-8 bytes rather
         * than Unicode characters. This ensures any text can be represented
         * (no unknown tokens at the byte level) and provides better handling
         * of multilingual text.
         *
         * When false, operates on character level (may have unknown characters).
         *
         * @param byte_level True for byte-level (recommended), false for char-level.
         * @return Reference to this config for method chaining.
         */
        BpeTrainerConfig& withByteLevel( bool byte_level ) {
            byte_level_ = byte_level;
            return *this;
        }

        /**
         * @brief Set pre-tokenization pattern.
         *
         * Pre-tokenization splits text into units before BPE merging.
         * Common patterns:
         * - "" (empty): Simple whitespace splitting (default)
         * - GPT-2: Regex pattern preserving contractions, numbers, etc.
         * - Custom: Domain-specific splitting rules
         *
         * @param pattern Regular expression pattern for pre-tokenization.
         *                Empty string uses simple whitespace splitting.
         * @return Reference to this config for method chaining.
         */
        BpeTrainerConfig& withPreTokenizationPattern( const std::string& pattern ) {
            pre_tokenization_pattern_ = pattern;
            return *this;
        }

        /**
         * @brief Set maximum number of merge iterations.
         *
         * Training stops when either vocab_size is reached or max_merges
         * iterations complete. Useful for limiting training time on large corpora.
         *
         * @param max_merges Maximum merge iterations (0 = no limit, default).
         * @return Reference to this config for method chaining.
         */
        BpeTrainerConfig& withMaxMerges( size_t max_merges ) {
            max_merges_ = max_merges;
            return *this;
        }

        /**
         * @brief Enable or disable merge caching.
         *
         * When enabled, caches merge operations to speed up training on large
         * corpora at the cost of increased memory usage.
         *
         * @param enable True to enable caching (default), false to disable.
         * @return Reference to this config for method chaining.
         */
        BpeTrainerConfig& withMergeCaching( bool enable ) {
            enable_merge_caching_ = enable;
            return *this;
        }

        /**
         * @brief Validate configuration.
         *
         * Throws std::invalid_argument when validation fails.
         *
         * Validation checks:
         * - vocab_size must be > 0
         * - min_frequency must be > 0
         * - vocab_size must be larger than the assumed base vocabulary size
         *   (256 for byte-level, otherwise 128). If special tokens are enabled
         *   an allowance is added for pad/unk/bos/eos.
         *
         * @throws std::invalid_argument If the configuration is invalid.
         */
        void validate() const
        {
            if ( vocab_size_ == 0 ) {
                throw std::invalid_argument( "BpeTrainerConfig: vocab_size must be > 0" );
            }

            if ( min_frequency_ == 0 ) {
                throw std::invalid_argument( "BpeTrainerConfig: min_frequency must be > 0" );
            }

            // For byte-level, validate minimum vocab size
            if ( byte_level_ ) {
                size_t min_base_size = 256 + special_tokens_.count();

                if ( vocab_size_ < min_base_size ) {
                    throw std::invalid_argument(
                        "BpeTrainerConfig: vocab_size (" + std::to_string( vocab_size_ ) +
                        ") must be > base vocabulary size (" + std::to_string( min_base_size ) +
                        ") [256 bytes + " + std::to_string( special_tokens_.count() ) + " special tokens]"
                    );
                }
            }

            // Validate max_merges if set
            if ( max_merges_ > 0 && max_merges_ >= vocab_size_ ) {
                throw std::invalid_argument(
                    "BpeTrainerConfig: max_merges must be < vocab_size (or 0 for unlimited)"
                );
            }

            // Validate special tokens configuration
            if ( special_tokens_.enabled ) {
                special_tokens_.validate();
            }
        }

        const BpeSpecialTokens& getSpecialTokens() const {
            return special_tokens_;
        }

        size_t getVocabSize() const {
            return vocab_size_;
        }
        
        size_t getMinFrequency() const {
            return min_frequency_;
        }
        
        bool isByteLevel() const {
            return byte_level_;
        }
        
        const std::string& getPreTokenizationPattern() const {
            return pre_tokenization_pattern_;
        }
        
        size_t getMaxMerges() const {
            return max_merges_;
        }
        
        bool isMergeCachingEnabled() const {
            return enable_merge_caching_;
        }

    private:
        
        size_t vocab_size_ = 32000;
        size_t min_frequency_ = 2;
        bool byte_level_ = true;
        BpeSpecialTokens special_tokens_ = BpeSpecialTokens::standard();
        std::string pre_tokenization_pattern_ = "";  // Empty = simple whitespace
        size_t max_merges_ = 0;  // 0 = no limit
        bool enable_merge_caching_ = true;
    };
}