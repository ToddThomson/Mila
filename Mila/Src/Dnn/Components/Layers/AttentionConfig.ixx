/**
 * @file AttentionConfig.ixx
 * @brief Configuration interface for the Attention module.
 */

module;
#include <stdexcept>
#include <string>
#include <utility>
#include <sstream>

export module Dnn.Components.Attention:Config;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.TensorTypes;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;

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
    export class AttentionConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Constructor with required parameters.
         *
         * @param embedding_dim The embedding dimension size
         * @param num_heads The number of attention heads
         */
        AttentionConfig( dim_t embedding_dim, dim_t num_heads )
			: embedding_dim_( embedding_dim ), num_heads_( num_heads ), ComponentConfig( "attention" )
        {
        }

        /**
         * @brief C++23-style fluent setter for embedding dimension.
         */
        template <typename Self>
        decltype(auto) withEmbeddingDim( this Self&& self, dim_t embedding_dim )
        {
            self.embedding_dim_ = embedding_dim;
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
         * @brief Get the embedding dimension.
         */
        dim_t getEmbeddingDim() const noexcept
        {
            return embedding_dim_;
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

        /**
         * @brief Serialize configuration to JSON.
         *
         * Keys:
         * - "name" : string
         * - "precision" : integer (underlying value of ComputePrecision::Policy)
         * - "embedding_dim" : integer
         * - "num_heads" : integer
         */
        //json toJson() const
        //{
        //    json j;
        //    /*j["name"] = name_;
        //    j["precision"] = static_cast<int>( precision_ );
        //    j["embedding_dim"] = static_cast<int64_t>( embedding_dim_ );
        //    j["num_heads"] = static_cast<int64_t>( num_heads_ );*/

        //    return j;
        //}

        /**
         * @brief Deserialize configuration from JSON (ModuleConfig interface).
         *
         * Missing keys leave fields at their current values.
         */
        //void fromJson( const json& j )
        //{
        //    /*if ( j.contains( "name" ) )
        //    {
        //        name_ = j.at( "name" ).get<std::string>();
        //    }

        //    if ( j.contains( "precision" ) )
        //    {
        //        precision_ = static_cast<decltype( precision_)>( j.at( "precision" ).get<int>() );
        //    }

        //    if ( j.contains( "embedding_dim" ) )
        //    {
        //        embedding_dim_ = static_cast<dim_t>( j.at( "embedding_dim" ).get<int64_t>() );
        //    }

        //    if ( j.contains( "num_heads" ) )
        //    {
        //        num_heads_ = static_cast<dim_t>( j.at( "num_heads" ).get<int64_t>() );
        //    }*/
        //}

        /**
         * @brief String representation of the configuration (ModuleConfig interface).
         *
		 * @return std::string Human-readable description of the configuration.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "AttentionConfig: { ";
            oss << "name=" << name_ << ", ";
            oss << "precision=" << static_cast<int>(precision_) << ", ";
            oss << " }";
            return oss.str();
        }

    private:

        dim_t embedding_dim_;
        dim_t num_heads_;
    };
}