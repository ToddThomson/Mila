/**
 * @file MultiHeadAttentionConfig.ixx
 * @brief Configuration interface for the Attention module.
 */

module;
#include <stdexcept>
#include <string>
#include <utility>
#include <sstream>

export module Dnn.Components.MultiHeadAttention:Config;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.TensorTypes;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

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
    export class MultiHeadAttentionConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Constructor with required parameters.
         *
         * @param model_dim The model dimension size
         * @param num_heads The number of attention heads
         */
        MultiHeadAttentionConfig( dim_t model_dim, dim_t num_heads )
            : model_dim_( model_dim ), num_heads_( num_heads )
        {}

        /**
         * @brief C++23-style fluent setter for model dimension.
         */
        template <typename Self>
        decltype(auto) withModelDim( this Self&& self, dim_t model_dim )
        {
            self.model_dim_ = model_dim;

            return std::forward<Self>( self );
        }

        /**
         * @brief C++23-style fluent setter for number of heads.
         */
        template <typename Self>
        decltype(auto) withNumHeads( this Self&& self, dim_t num_heads )
        {
            self.num_heads_ = num_heads;
            return std::forward<Self>( self );
        }

        /**
         * @brief Get the model dimension.
         */
        dim_t getModelDim() const noexcept
        {
            return model_dim_;
        }

        /**
         * @brief Get the number of attention heads.
         */
        dim_t getNumHeads() const noexcept
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
            if ( model_dim_ <= 0 )
            {
                throw std::invalid_argument( "MultiHeadAttentionConfig: model_dim must be > 0" );
            }

            if ( num_heads_ < 2 )
            {
                throw std::invalid_argument( "MultiHeadAttentionConfig: num_heads must be >= 2" );
            }

            if ( model_dim_ % num_heads_ != 0 )
            {
                throw std::invalid_argument( "MultiHeadAttentionConfig: model_dim must be divisible by num_heads" );
            }
        }


        /**
         * @brief Convert configuration to serialization metadata.
         *
         * Produces a SerializationMetadata object containing the configuration
         * fields suitable for writing into an archive by the caller.
         */
        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>(precision_) )
                .set( "model_dim", static_cast<int64_t>(model_dim_) )
                .set( "num_heads", static_cast<int64_t>(num_heads_) );

            return meta;
        }

        /**
         * @brief Populate configuration from serialization metadata.
         *
         * Reads available fields from the provided metadata and updates the
         * configuration object accordingly.
         */
        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto prec = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype(precision_)>(*prec);
            }

            if ( auto md = meta.tryGetInt( "model_dim" ) )
            {
                model_dim_ = static_cast<dim_t>(*md);
            }

            if ( auto nh = meta.tryGetInt( "num_heads" ) )
            {
                num_heads_ = static_cast<dim_t>(*nh);
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
            oss << "MultiHeadAttentionConfig: { ";
            oss << "precision=" << static_cast<int>(precision_) << ", ";
            oss << "model_dim=" << model_dim_ << ", ";
            oss << "num_heads=" << num_heads_ << " }";

            return oss.str();
        }

    private:

        dim_t model_dim_;
        dim_t num_heads_;
    };
}