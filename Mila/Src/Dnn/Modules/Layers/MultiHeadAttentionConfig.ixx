/**
 * @file MultiHeadAttentionConfig.ixx
 * @brief Configuration interface for the MultiHeadAttention module in the Mila DNN framework.
 *
 * Defines the MultiHeadAttentionConfig class, providing a type-safe fluent interface for configuring
 * MultiHeadAttention modules. Inherits from ConfigurationBase CRTP base and adds attention-specific options
 * such as embedding dimension, number of heads, and dropout rates.
 */

module;
#include <stdexcept>
#include <vector>

export module Dnn.Modules.Attention:Config;

import Dnn.Module;
import Dnn.ConfigurationBase;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for MultiHeadAttention module.
     */
    export class MultiHeadAttentionConfig : public ConfigurationBase {
    public:
        /**
         * @brief Constructor with required parameters.
         *
         * @param embedding_dim The embedding dimension size
         * @param num_heads The number of attention heads
         */
        MultiHeadAttentionConfig( size_t embedding_dim, size_t num_heads )
            : embedding_dim_( embedding_dim ), num_heads_( num_heads ) {}

        /**
         * @brief Set the input shape for the attention module.
         *
         * @param input_shape Vector containing input tensor dimensions [batch_size, seq_len, embedding_dim]
         * @return MultiHeadAttentionConfig& Reference to this for method chaining
         */
        MultiHeadAttentionConfig& withInputShape( const shape_t& input_shape ) {
            input_shape_ = input_shape;
            return *this;
        }

        /**
         * @brief Set the dropout rate for attention weights.
         *
         * @param dropout Dropout probability (0.0 to 1.0)
         * @return MultiHeadAttentionConfig& Reference to this for method chaining
         */
        MultiHeadAttentionConfig& withDropout( float dropout ) {
            dropout_ = dropout;
            return *this;
        }

        /**
         * @brief Configure whether to use causal attention mask.
         *
         * @param causal True to use causal masking (for decoder/autoregressive models)
         * @return MultiHeadAttentionConfig& Reference to this for method chaining
         */
        MultiHeadAttentionConfig& withCausalMask( bool causal ) {
            use_causal_mask_ = causal;
            return *this;
        }

        /**
         * @brief Set the scaling factor for attention logits.
         *
         * @param scale_factor Scaling factor (typically 1/sqrt(head_dim))
         * @return MultiHeadAttentionConfig& Reference to this for method chaining
         */
        MultiHeadAttentionConfig& withScaleFactor( float scale_factor ) {
            scale_factor_ = scale_factor;
            return *this;
        }

        /**
         * @brief Configure whether to use separate projection matrices for query, key, and value.
         *
         * @param separate_projections True to use separate projections, false to use a single matrix
         * @return MultiHeadAttentionConfig& Reference to this for method chaining
         */
        MultiHeadAttentionConfig& withSeparateProjections( bool separate_projections ) {
            separate_projections_ = separate_projections;
            return *this;
        }

        /**
         * @brief Get the embedding dimension.
         */
        dim_t getEmbeddingDim() const { return embedding_dim_; }

        /**
         * @brief Get the number of attention heads.
         */
        dim_t getNumHeads() const { return num_heads_; }

        /**
         * @brief Get the input shape.
         */
        const shape_t& getInputShape() const { return input_shape_; }

        /**
         * @brief Get the dropout rate.
         */
        float getDropout() const { return dropout_; }

        /**
         * @brief Check if causal masking is enabled.
         */
        bool useCausalMask() const { return use_causal_mask_; }

        /**
         * @brief Get the attention scaling factor.
         */
        float getScaleFactor() const { return scale_factor_; }

        /**
         * @brief Check if using separate projection matrices.
         */
        bool useSeparateProjections() const { return separate_projections_; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails
         */
        void validate() const {
            ConfigurationBase::validate();

            if ( embedding_dim_ == 0 ) {
                throw std::invalid_argument( "Embedding dimension must be greater than zero" );
            }

            if ( num_heads_ == 0 ) {
                throw std::invalid_argument( "Number of attention heads must be greater than zero" );
            }

            if ( embedding_dim_ % num_heads_ != 0 ) {
                throw std::invalid_argument( "Embedding dimension must be divisible by number of heads" );
            }

            if ( dropout_ < 0.0f || dropout_ >= 1.0f ) {
                throw std::invalid_argument( "Dropout probability must be in range [0, 1)" );
            }

            if ( scale_factor_ <= 0.0f ) {
                throw std::invalid_argument( "Scale factor must be positive" );
            }

            if ( !input_shape_.empty() ) {
                if ( input_shape_.size() < 3 ) {
                    throw std::invalid_argument( "Input shape must have at least 3 dimensions [batch, seq_len, embedding_dim]" );
                }

                if ( input_shape_.back() != embedding_dim_ ) {
                    throw std::invalid_argument( "Last dimension of input shape must match embedding dimension" );
                }
            }
        }

    private:
        dim_t embedding_dim_;
        size_t num_heads_;
        shape_t input_shape_;
        float dropout_ = 0.0f;
        bool use_causal_mask_ = false;
        float scale_factor_ = 1.0f;
        bool separate_projections_ = true;
    };
}