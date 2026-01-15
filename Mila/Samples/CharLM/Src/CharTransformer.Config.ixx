/**
 * @file CharTransformer.Config.ixx
 * @brief Configuration partition for CharTransformer.
 */

module;
#include <cstdint>
#include <stdexcept>
#include <string>
#include <sstream>

export module CharLM.Transformer:Config;

import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::CharLM
{
    using Mila::Dnn::ComponentConfig;
    using Mila::Dnn::Serialization::SerializationMetadata;

    /**
     * @brief Construction-time configuration for CharTransformer.
     *
     * Lightweight value object containing the model dimensions and limits.
     * Call `validate()` before using the configuration to construct a network.
     *
     * Serialization keys:
     *  - "precision" : integer (ComputePrecision::Policy)
     *  - "vocab_size" : integer
     *  - "max_seq_length" : integer
     *  - "embedding_dim" : integer
     *  - "num_heads" : integer
     *  - "num_layers" : integer
     *  - "mlp_hidden_dim" : integer
     */
    export class CharTransformerConfig : public ComponentConfig
    {
    public:
        CharTransformerConfig() = default;

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument if any constraint is violated.
         */
        void validate() const override
        {
            if (vocab_size <= 0)
            {
                throw std::invalid_argument("CharTransformerConfig: vocab_size must be positive");
            }

            if (max_seq_length <= 0)
            {
                throw std::invalid_argument("CharTransformerConfig: max_seq_length must be positive");
            }

            if (embedding_dim <= 0)
            {
                throw std::invalid_argument("CharTransformerConfig: embedding_dim must be positive");
            }

            if (num_heads <= 0)
            {
                throw std::invalid_argument("CharTransformerConfig: num_heads must be positive");
            }

            if (embedding_dim % num_heads != 0)
            {
                throw std::invalid_argument("CharTransformerConfig: embedding_dim must be divisible by num_heads");
            }

            if (num_layers <= 0)
            {
                throw std::invalid_argument("CharTransformerConfig: num_layers must be positive");
            }

            if (mlp_hidden_dim <= 0)
            {
                throw std::invalid_argument("CharTransformerConfig: mlp_hidden_dim must be positive");
            }
        }

        /**
         * @brief Convert configuration to SerializationMetadata.
         *
         * @return SerializationMetadata metadata describing this configuration.
         */
        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;
            meta.set("precision", static_cast<int64_t>(precision_))
                .set("vocab_size", static_cast<int64_t>(vocab_size))
                .set("max_seq_length", static_cast<int64_t>(max_seq_length))
                .set("embedding_dim", static_cast<int64_t>(embedding_dim))
                .set("num_heads", static_cast<int64_t>(num_heads))
                .set("num_layers", static_cast<int64_t>(num_layers))
                .set("mlp_hidden_dim", static_cast<int64_t>(mlp_hidden_dim));

            return meta;
        }

        /**
         * @brief Populate configuration from provided metadata.
         *
         * Missing keys are ignored so defaults remain intact.
         *
         * @param meta Metadata to read configuration values from.
         */
        void fromMetadata(const SerializationMetadata& meta) override
        {
            if (auto p = meta.tryGetInt("precision"))
            {
                precision_ = static_cast<decltype(precision_)>(*p);
            }

            if (auto v = meta.tryGetInt("vocab_size"))
            {
                vocab_size = *v;
            }

            if (auto m = meta.tryGetInt("max_seq_length"))
            {
                max_seq_length = *m;
            }

            if (auto e = meta.tryGetInt("embedding_dim"))
            {
                embedding_dim = *e;
            }

            if (auto h = meta.tryGetInt("num_heads"))
            {
                num_heads = *h;
            }

            if (auto nl = meta.tryGetInt("num_layers"))
            {
                num_layers = *nl;
            }

            if (auto mh = meta.tryGetInt("mlp_hidden_dim"))
            {
                mlp_hidden_dim = *mh;
            }
        }

        /**
         * @brief Human-readable summary for logging.
         *
         * @return std::string Compact description of the configuration.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "CharTransformerConfig(vocab_size=" << vocab_size
                << ", max_seq_length=" << max_seq_length
                << ", embedding_dim=" << embedding_dim
                << ", num_heads=" << num_heads
                << ", num_layers=" << num_layers
                << ", mlp_hidden_dim=" << mlp_hidden_dim
                << ", precision=" << static_cast<int>(precision_) << ")";

            return oss.str();
        }

        // Public fields (value-type style)
        int64_t vocab_size = 256;
        int64_t max_seq_length = 256;
        int64_t embedding_dim = 256;
        int64_t num_heads = 4;
        int64_t num_layers = 4;
        int64_t mlp_hidden_dim = 1024;
    };
}