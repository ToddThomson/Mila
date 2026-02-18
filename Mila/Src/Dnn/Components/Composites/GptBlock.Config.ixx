/**
 * @file GptBlock.Config.ixx
 * @brief Configuration for GPT-style transformer block (block-level).
 *
 * Provides fluent setters and serialization helpers used by block factories.
 */

module;
#include <stdexcept>
#include <vector>
#include <utility>
#include <string>
#include <sstream>
#include <optional>

export module Dnn.Components.GptBlock:Config;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Dnn.ActivationType;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration class for GPT transformer blocks.
     *
     * Holds the model dimension, attention head count, MLP/attention options,
     * and basic activation/residual settings. LLaMA-specific settings (RoPE,
     * KV-head sharing, RMSNorm selection/epsilon, etc.) are intentionally
     * omitted from this block-level GPT config.
     */
    export class GptBlockConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Construct a GPT block configuration.
         *
         * @param model_dim Model dimension. Must be > 0.
         * @param num_heads Number of query attention heads. Must be > 0 and must divide model_dim evenly.
         */
        GptBlockConfig( dim_t model_dim, dim_t num_heads )
            : model_dim_( model_dim ), num_heads_( num_heads )
        {
            if ( model_dim <= 0 )
            {
                throw std::invalid_argument( "GptBlockConfig: model_dim must be > 0" );
            }

            if ( num_heads <= 0 )
            {
                throw std::invalid_argument( "GptBlockConfig: num_heads must be > 0" );
            }

            if ( model_dim % num_heads != 0 )
            {
                throw std::invalid_argument( "GptBlockConfig: model_dim must be divisible by num_heads" );
            }
        }

        // Fluent setters for dimensions and basic config

        template <typename Self>
        decltype(auto) withHiddenSize( this Self&& self, dim_t hidden_dim )
        {
            self.hidden_dim_ = hidden_dim;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withBias( this Self&& self, bool use_bias )
        {
            self.use_bias_ = use_bias;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withActivation( this Self&& self, ActivationType activation_type )
        {
            self.activation_type_ = activation_type;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withResidualScale( this Self&& self, float scale )
        {
            self.residual_scale_ = scale;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set maximum sequence length for block-level positional handling.
         *
         * @param max_seq_len Maximum sequence length (must be > 0).
         */
        template <typename Self>
        decltype(auto) withMaxSequenceLength( this Self&& self, dim_t max_seq_len )
        {
            if ( max_seq_len <= 0 )
            {
                throw std::invalid_argument( "max_seq_len must be > 0" );
            }

            self.max_seq_len_ = max_seq_len;
            return std::forward<Self>( self );
        }

        // Getters

        dim_t getModelDim() const noexcept { return model_dim_; }
        dim_t getNumHeads() const noexcept { return num_heads_; }
        dim_t getHiddenSize() const noexcept { return hidden_dim_; }
        bool useBias() const noexcept { return use_bias_; }
        ActivationType getActivationType() const noexcept { return activation_type_; }
        float getResidualScale() const noexcept { return residual_scale_; }

        /**
         * @brief Get effective hidden dimension with default fallback.
         *
         * @return Hidden dimension, defaulting to 4x model_dim if not set.
         */
        dim_t getEffectiveHiddenDimension() const noexcept
        {
            return hidden_dim_ > 0 ? hidden_dim_ : (model_dim_ * 4);
        }

        void validate() const override
        {
            if ( model_dim_ <= 0 )
            {
                throw std::invalid_argument( "Model dimension must be greater than zero" );
            }

            if ( num_heads_ == 0 )
            {
                throw std::invalid_argument( "Number of attention heads must be greater than zero" );
            }

            if ( model_dim_ % num_heads_ != 0 )
            {
                throw std::invalid_argument( "Model dimension must be divisible by number of heads" );
            }
        }

        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>(precision_) )
                .set( "model_dim", static_cast<int64_t>(model_dim_) )
                .set( "num_heads", static_cast<int64_t>(num_heads_) )
                .set( "hidden_dim", static_cast<int64_t>(hidden_dim_) )
                .set( "use_bias", use_bias_ )
                .set( "activation", static_cast<int64_t>(activation_type_) )
                .set( "residual_scale", static_cast<double>(residual_scale_) )
                .set( "max_seq_len", static_cast<int64_t>(max_seq_len_) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto p = meta.tryGetInt( "precision" ) )
                precision_ = static_cast<decltype(precision_)>(*p);
            if ( auto md = meta.tryGetInt( "model_dim" ) )
                model_dim_ = static_cast<dim_t>(*md);
            if ( auto nh = meta.tryGetInt( "num_heads" ) )
                num_heads_ = static_cast<dim_t>(*nh);
            if ( auto hd = meta.tryGetInt( "hidden_dim" ) )
                hidden_dim_ = static_cast<dim_t>(*hd);
            if ( auto ub = meta.tryGetBool( "use_bias" ) )
                use_bias_ = *ub;
            if ( auto act = meta.tryGetInt( "activation" ) )
                activation_type_ = static_cast<ActivationType>(*act);
            if ( auto rscale = meta.tryGetFloat( "residual_scale" ) )
                residual_scale_ = static_cast<float>(*rscale);
            if ( auto msl = meta.tryGetInt( "max_seq_len" ) )
                max_seq_len_ = static_cast<dim_t>(*msl);
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "GptBlock Configuration:\n";
            oss << "  Model Dim: " << model_dim_ << "\n";
            oss << "  Num Heads: " << num_heads_ << "\n";
            oss << "  Hidden Dim: " << getEffectiveHiddenDimension() << "\n";
            oss << "  Use Bias: " << (use_bias_ ? "Yes" : "No") << "\n";
            oss << "  Activation: " << static_cast<int>(activation_type_) << "\n";
            oss << "  Residual Scale: " << residual_scale_ << "\n";
            oss << "  Max Seq Len: " << max_seq_len_ << "\n";

            return oss.str();
        }

    private:
        dim_t model_dim_;
        dim_t num_heads_;
        dim_t hidden_dim_ = 0;
        bool use_bias_ = false;
        ActivationType activation_type_ = ActivationType::Gelu;
        float residual_scale_ = 1.0f;
        dim_t max_seq_len_ = 2048;
    };
}