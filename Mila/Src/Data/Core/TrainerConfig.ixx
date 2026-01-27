module;
#include <string>
#include <cstdint>
#include <memory>

export module Data.TrainerConfig;

import Data.SpecialTokens;

namespace Mila::Data
{
    /**
     * @brief Base configuration class for tokenizer trainers.
     *
     * Provides common configuration options shared across all tokenizer types.
     * Concrete tokenizer configurations inherit from this class and add
     * type-specific options.
     *
     * This class is abstract and cannot be instantiated directly.
     */
     
     // TJT: Remove as not required
     
    //export class TrainerConfig {
    //public:
    //    virtual ~TrainerConfig() = default;

    //    /**
    //     * @brief Get the configured special tokens.
    //     */
    //    const SpecialTokens& getSpecialTokens() const {
    //        return special_tokens_;
    //    }

    //    /**
    //     * @brief Check if configuration is valid.
    //     *
    //     * Implementations should override to add type-specific validation.
    //     *
    //     * @return true if configuration is valid, false otherwise.
    //     */
    //    virtual bool isValid() const {
    //        return true;  // Base class has no invariants to check
    //    }

    //protected:
    //    SpecialTokens special_tokens_;
    //};
}