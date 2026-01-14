/**
 * @file CrossEntropyConfig.ixx
 * @brief Configuration for the fused SoftmaxCrossEntropy loss module.
 *
 * Minimal configuration used by the fused softmax + cross-entropy kernels.
 */

module;
#include <stdexcept>
#include <cstdint>
#include <utility>
#include <string>
#include <ostream>
#include <sstream>

export module Dnn.Components.SoftmaxCrossEntropy:Config;

import Dnn.Component;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration for fused SoftmaxCrossEntropy loss.
     *
     * Provides a lightweight value object containing the vocabulary size
     * required by the fused softmax + cross-entropy kernels.
     */
    export class CrossEntropyConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Default constructor.
         *
         * Leaves `vocab_size_` at 0; callers should set it before using the
         * configuration. `validate()` will reject a zero vocab size.
         */
        CrossEntropyConfig() = default;

        /**
         * @brief Constructor with required vocabulary size parameter.
         *
         * @param vocab_size The size of the vocabulary (number of classes).
         *                   Must be > 0. Kernels validate: 0 <= target < vocab_size.
         */
        explicit CrossEntropyConfig( int64_t vocab_size )
            : vocab_size_( vocab_size )
        {
        }

        /**
         * @brief C++23-style fluent setter for vocabulary size.
         *
         * @param vocab_size Vocabulary size (number of classes)
         * @return Self&& for method chaining
         */
        template <typename Self>
        decltype(auto) withVocabSize( this Self&& self, int64_t vocab_size )
        {
            self.vocab_size_ = vocab_size;
            return std::forward<Self>( self );
        }

        /**
         * @brief Get the vocabulary size.
         *
         * Used by kernels to validate target indices.
         *
         * @return int64_t The vocabulary size
         */
        int64_t getVocabSize() const
        {
            return vocab_size_;
        }

        /**
         * @brief Validate configuration parameters.
         *
         * Checks that vocabulary size is positive.
         *
         * @throws std::invalid_argument If vocab_size <= 0
         */
        void validate() const override
        {
            if (vocab_size_ <= 0)
            {
                throw std::invalid_argument(
                    "CrossEntropyConfig: vocabulary size must be greater than zero" );
            }
        }

        /**
         * @brief Convert configuration into SerializationMetadata.
         *
         * Produces keys:
         * - "precision" : integer (underlying value of ComputePrecision::Policy)
         * - "vocab_size" : integer
         *
         * @return SerializationMetadata Metadata representing this configuration.
         */
        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>( precision_ ) )
                .set( "vocab_size", static_cast<int64_t>( vocab_size_ ) );

            return meta;
        }

        /**
         * @brief Populate configuration from provided metadata.
         *
         * Missing keys are ignored, leaving defaults intact. Type-safe try-get
         * helpers are used to avoid throwing on absent fields.
         *
         * @param meta Metadata to read configuration values from.
         */
        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto p = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_ )>( *p );
            }

            if ( auto v = meta.tryGetInt( "vocab_size" ) )
            {
                vocab_size_ = *v;
            }
        }

        /**
         * @brief String representation of the configuration.
         *
         * @return std::string Human-readable description of the configuration.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "CrossEntropyConfig(vocab_size=" << vocab_size_ << ")";

            return oss.str();
        }

    private:
        int64_t vocab_size_ = 0;  ///< Number of classes in the vocabulary
    };
}