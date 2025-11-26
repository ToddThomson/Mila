/**
 * @file CrossEntropyConfig.ixx
 * @brief Configuration interface for the fused SoftmaxCrossEntropy module.
 *
 * Provides minimal configuration matching the actual CPU/CUDA kernel implementations.
 * This is the configuration for the FUSED softmax + cross-entropy loss operation.
 *
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
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;

    /**
     * @brief Configuration for fused SoftmaxCrossEntropy loss.
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
         *
         * Example:
         *   CrossEntropyConfig(50257)  // GPT-2 vocab size
         */
        explicit CrossEntropyConfig( int64_t vocab_size )
			: vocab_size_( vocab_size ), ComponentConfig( "cross_entropy" )
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
         * Targets outside [0, vocab_size) are automatically ignored (loss/grad = 0).
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
            //ComponentConfig::validate();

            if (vocab_size_ <= 0)
            {
                throw std::invalid_argument(
                    "CrossEntropyConfig: vocabulary size must be greater than zero" );
            }
        }

        /**
         * @brief Serialize configuration to JSON (ModuleConfig interface).
         *
         * Produces keys:
         * - "name" : string
         * - "precision" : integer (underlying value of ComputePrecision::Policy)
         * - "vocab_size" : integer
         */
        json toJson() const
        {
            json j;
            //j["name"] = name_;
            //j["precision"] = static_cast<int>( precision_ );
            //j["vocab_size"] = vocab_size_;

            return j;
        }

        /**
         * @brief Deserialize configuration from JSON (ModuleConfig interface).
         *
         * Missing keys leave fields at their current values.
         */
        void fromJson( const json& j )
        {
            if ( j.contains( "name" ) )
            {
                name_ = j.at( "name" ).get<std::string>();
            }
            /*
            if ( j.contains( "precision" ) )
            {
                precision_ = static_cast<decltype(precision_)>( j.at( "precision" ).get<int>() );
            }

            if ( j.contains( "vocab_size" ) )
            {
                vocab_size_ = j.at( "vocab_size" ).get<int64_t>();
            }*/
        }

        /**
         * @brief String representation of the configuration (ModuleConfig interface).
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