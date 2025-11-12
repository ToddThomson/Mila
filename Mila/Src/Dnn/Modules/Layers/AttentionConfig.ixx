/**
 * @file AttentionConfig.ixx
 * @brief Configuration interface for the Attention module in the Mila DNN framework.
 *
 * Defines the AttentionConfig class, providing a type-safe fluent interface for configuring
 * Attention modules. Inherits from ConfigurationBase CRTP base and adds attention-specific options
 * such as embedding dimension, number of heads, and dropout rates.
 */

module;
#include <stdexcept>

export module Dnn.Modules.Attention:Config;

import Dnn.Module;
import Dnn.ConfigurationBase;
import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for Attention module.
     *
     * Note: Some configuration options are currently disabled and marked for future implementation.
     * The base implementation provides core multi-head attention functionality with:
     * - Fixed causal masking (enabled by default for autoregressive models)
     * - Automatic scale factor (1/sqrt(head_dim))
     * - No dropout (to be added in future versions)
     * - Unified Q/K/V input (separate projections to be added in future versions)
     */
    export class AttentionConfig : public ConfigurationBase
    {
    public:
        /**
         * @brief Constructor with required parameters.
         *
         * @param embedding_dim The embedding dimension size
         * @param num_heads The number of attention heads
         */
        AttentionConfig( dim_t embedding_dim, dim_t num_heads )
            : embedding_dim_( embedding_dim ), num_heads_( num_heads )
        {
        }

        /**
         * @brief Get the embedding dimension.
         */
        dim_t getEmbeddingDim() const
        {
            return embedding_dim_;
        }

        /**
         * @brief Get the number of attention heads.
         */
        dim_t getNumHeads() const
        {
            return num_heads_;
        }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails
         */
        void validate() const override
        {
            ConfigurationBase::validate();

            if (embedding_dim_ <= 0)
            {
                throw std::invalid_argument( "Embedding dimension must be greater than zero" );
            }

            if (num_heads_ <= 0)
            {
                throw std::invalid_argument( "Number of attention heads must be greater than zero" );
            }

            if (embedding_dim_ % num_heads_ != 0)
            {
                throw std::invalid_argument( "Embedding dimension must be divisible by number of heads" );
            }
        }

    private:

        dim_t embedding_dim_;
        dim_t num_heads_;
    };
}