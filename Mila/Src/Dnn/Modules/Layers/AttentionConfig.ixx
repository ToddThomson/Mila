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
#include <vector>

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
        AttentionConfig( int64_t embedding_dim, int64_t num_heads )
            : embedding_dim_( embedding_dim ), num_heads_( num_heads )
        {
        }

        // ====================================================================
        // Future Configuration Options (Currently Disabled)
        // ====================================================================

        // TODO: Uncomment when dropout is implemented in attention kernels
        /**
         * @brief Set the dropout rate for attention weights.
         *
         * @param dropout Dropout probability (0.0 to 1.0)
         * @return AttentionConfig& Reference to this for method chaining
         */
         // AttentionConfig& withDropout( float dropout ) {
         //     dropout_ = dropout;
         //     return *this;
         // }

         // TODO: Uncomment when configurable masking is implemented
         /**
          * @brief Configure whether to use causal attention mask.
          *
          * @param causal True to use causal masking (for decoder/autoregressive models)
          * @return AttentionConfig& Reference to this for method chaining
          */
          // AttentionConfig& withCausalMask( bool causal ) {
          //     use_causal_mask_ = causal;
          //     return *this;
          // }

          // TODO: Uncomment when custom scale factors are supported
          /**
           * @brief Set the scaling factor for attention logits.
           *
           * @param scale_factor Scaling factor (typically 1/sqrt(head_dim))
           * @return AttentionConfig& Reference to this for method chaining
           */
           // AttentionConfig& withScaleFactor( float scale_factor ) {
           //     scale_factor_ = scale_factor;
           //     return *this;
           // }

           // TODO: Uncomment when separate Q/K/V projections are implemented
           /**
            * @brief Configure whether to use separate projection matrices for query, key, and value.
            *
            * @param separate_projections True to use separate projections, false to use a single matrix
            * @return AttentionConfig& Reference to this for method chaining
            */
            // AttentionConfig& withSeparateProjections( bool separate_projections ) {
            //     separate_projections_ = separate_projections;
            //     return *this;
            // }

            // ====================================================================
            // Active Configuration Accessors
            // ====================================================================

            /**
             * @brief Get the embedding dimension.
             */
        int64_t getEmbeddingDim() const
        {
            return embedding_dim_;
        }

        /**
         * @brief Get the number of attention heads.
         */
        int64_t getNumHeads() const
        {
            return num_heads_;
        }

        // ====================================================================
        // Future Configuration Accessors (Currently Disabled)
        // ====================================================================

        // TODO: Uncomment when dropout is implemented
        /**
         * @brief Get the dropout rate.
         */
         // float getDropout() const { return dropout_; }

         // TODO: Uncomment when configurable masking is implemented
         /**
          * @brief Check if causal masking is enabled.
          */
          // bool useCausalMask() const { return use_causal_mask_; }

          // TODO: Uncomment when custom scale factors are supported
          /**
           * @brief Get the attention scaling factor.
           */
           // float getScaleFactor() const { return scale_factor_; }

           // TODO: Uncomment when separate projections are implemented
           /**
            * @brief Check if using separate projection matrices.
            */
            // bool useSeparateProjections() const { return separate_projections_; }

            // ====================================================================
            // Validation
            // ====================================================================

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

            // TODO: Uncomment when dropout is implemented
            // if ( dropout_ < 0.0f || dropout_ >= 1.0f ) {
            //     throw std::invalid_argument( "Dropout probability must be in range [0, 1)" );
            // }

            // TODO: Uncomment when custom scale factors are supported
            // if ( scale_factor_ <= 0.0f ) {
            //     throw std::invalid_argument( "Scale factor must be positive" );
            // }
        }

    private:

        int64_t embedding_dim_;
        int64_t num_heads_;

        // FUTURE: configuration options (currently unused)
        // TODO: Uncomment when dropout is implemented in attention kernels
        // float dropout_ = 0.0f;

        // TODO: Uncomment when configurable masking is implemented
        // Currently: causal masking is always enabled in the kernels
        // bool use_causal_mask_ = true;

        // TODO: Uncomment when custom scale factors are supported
        // Currently: scale factor is computed as 1/sqrt(head_dim) in the kernel
        // float scale_factor_ = 1.0f;

        // TODO: Uncomment when separate Q/K/V projections are implemented
        // Currently: expects pre-concatenated Q/K/V input [B, T, 3*C]
        // bool separate_projections_ = false;
    };
}