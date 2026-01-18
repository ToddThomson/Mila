/**
 * @file TransformerConfig.ixx (Extended Version)
 * @brief Enhanced configuration supporting component selection for presets.
 */

module;
#include <stdexcept>
#include <vector>
#include <utility>
#include <string>
#include <sstream>
#include <optional>

export module Dnn.Blocks.Transformer:Config;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Dnn.ActivationType;
import Dnn.NormType;
import Dnn.AttentionType;
import Dnn.EncodingType;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration class for Transformer modules.
     *
     * Holds the embedding dimension, attention head count, MLP/attention options,
     * and component type selections for architectural flexibility.
     */
    export class TransformerConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Construct a Transformer configuration.
         *
         * @param embedding_dim Model embedding dimension. Must be > 0.
         * @param num_heads Number of query attention heads. Must be > 0 and must divide embedding_dim evenly.
         */
        TransformerConfig( dim_t embedding_dim, dim_t num_heads )
            : embedding_dim_( embedding_dim ), num_heads_( num_heads )
        {
            if ( embedding_dim <= 0 )
            {
                throw std::invalid_argument( "TransformerConfig: embedding_dim must be > 0" );
            }

            if ( num_heads <= 0 )
            {
                throw std::invalid_argument( "TransformerConfig: num_heads must be > 0" );
            }

            if ( embedding_dim % num_heads != 0 )
            {
                throw std::invalid_argument( "TransformerConfig: embedding_dim must be divisible by num_heads" );
            }
        }

        // ====================================================================
        // Fluent setters for dimensions and basic config
        // ====================================================================

        template <typename Self>
        decltype(auto) withHiddenDimension( this Self&& self, dim_t hidden_dim )
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

        // ====================================================================
        // Fluent setters for component selection (Llama-specific features)
        // ====================================================================

        /**
         * @brief Set normalization layer type.
         *
         * @param norm_type LayerNorm (GPT-2) or RMSNorm (LLaMA)
         */
        template <typename Self>
        decltype(auto) withNormType( this Self&& self, NormType norm_type )
        {
            self.norm_type_ = norm_type;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set attention mechanism type.
         *
         * @param attention_type Standard (MHA), GroupedQuery (GQA), or MultiQuery (MQA)
         */
        template <typename Self>
        decltype(auto) withAttentionType( this Self&& self, AttentionType attention_type )
        {
            self.attention_type_ = attention_type;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set number of key-value heads for GQA/MQA.
         *
         * @param num_kv_heads Number of KV heads. For MQA, use 1. For GQA, typically 4-8.
         *                     Must divide num_heads evenly.
         */
        template <typename Self>
        decltype(auto) withKVHeads( this Self&& self, dim_t num_kv_heads )
        {
            if ( num_kv_heads > 0 && self.num_heads_ % num_kv_heads != 0 )
            {
                throw std::invalid_argument( "num_heads must be divisible by num_kv_heads" );
            }
            self.num_kv_heads_ = num_kv_heads;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set positional encoding type.
         *
         * @param pos_type Learned (GPT-2), RoPE (LLaMA), or ALiBi (MPT/BLOOM)
         */
        template <typename Self>
        decltype(auto) withEncoding( this Self&& self, EncodingType pos_type )
        {
            self.encoding_type_ = pos_type;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set RoPE theta base frequency.
         *
         * @param theta Base frequency for RoPE. LLaMA 3 uses 500000.0f.
         */
        template <typename Self>
        decltype(auto) withRoPETheta( this Self&& self, float theta )
        {
            self.rope_theta_ = theta;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set RoPE scaling factor for extended context.
         *
         * @param scale_factor Scaling factor. LLaMA 3.1 uses 8.0f for 128K context.
         */
        template <typename Self>
        decltype(auto) withRoPEScaling( this Self&& self, float scale_factor )
        {
            self.rope_scaling_factor_ = scale_factor;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set maximum sequence length for positional encodings.
         *
         * @param max_seq_len Maximum sequence length. LLaMA 3: 8192, LLaMA 3.1: 131072
         */
        template <typename Self>
        decltype(auto) withMaxSequenceLength( this Self&& self, dim_t max_seq_len )
        {
            self.max_seq_len_ = max_seq_len;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set RMSNorm epsilon for numerical stability.
         *
         * @param eps Epsilon value. LLaMA uses 1e-5f.
         */
        template <typename Self>
        decltype(auto) withNormEpsilon( this Self&& self, float eps )
        {
            self.norm_epsilon_ = eps;
            return std::forward<Self>( self );
        }

        // ====================================================================
        // Getters
        // ====================================================================

        dim_t getEmbeddingDim() const noexcept {
            return embedding_dim_;
        }
        dim_t getNumHeads() const noexcept {
            return num_heads_;
        }
        dim_t getHiddenDimension() const noexcept {
            return hidden_dim_;
        }
        bool useBias() const noexcept {
            return use_bias_;
        }
        ActivationType getActivationType() const noexcept {
            return activation_type_;
        }
        float getResidualScale() const noexcept {
            return residual_scale_;
        }

        // Component selection getters
        NormType getNormType() const noexcept {
            return norm_type_;
        }
        AttentionType getAttentionType() const noexcept {
            return attention_type_;
        }
        dim_t getNumKVHeads() const noexcept
        {
            // Default to num_heads if not set (standard MHA)
            return num_kv_heads_ > 0 ? num_kv_heads_ : num_heads_;
        }
        EncodingType getEncodingType() const noexcept {
            return encoding_type_;
        }
        float getRoPETheta() const noexcept {
            return rope_theta_;
        }
        float getRoPEScaling() const noexcept {
            return rope_scaling_factor_;
        }
        dim_t getMaxSequenceLength() const noexcept {
            return max_seq_len_;
        }
        float getNormEpsilon() const noexcept {
            return norm_epsilon_;
        }

        /**
         * @brief Get effective hidden dimension with default fallback.
         *
         * @return Hidden dimension, defaulting to 4x embedding_dim if not set.
         */
        dim_t getEffectiveHiddenDimension() const noexcept
        {
            return hidden_dim_ > 0 ? hidden_dim_ : (embedding_dim_ * 4);
        }

        void validate() const override
        {
            if ( embedding_dim_ <= 0 )
            {
                throw std::invalid_argument( "Embedding dimension must be greater than zero" );
            }

            if ( num_heads_ == 0 )
            {
                throw std::invalid_argument( "Number of attention heads must be greater than zero" );
            }

            if ( embedding_dim_ % num_heads_ != 0 )
            {
                throw std::invalid_argument( "Embedding dimension must be divisible by number of heads" );
            }

            // Validate KV heads for GQA/MQA
            if ( num_kv_heads_ > 0 )
            {
                if ( num_heads_ % num_kv_heads_ != 0 )
                {
                    throw std::invalid_argument( "Number of query heads must be divisible by number of KV heads" );
                }

                if ( attention_type_ == AttentionType::Standard && num_kv_heads_ != num_heads_ )
                {
                    throw std::invalid_argument( "Standard attention requires num_kv_heads == num_heads" );
                }
            }

            // Validate RoPE settings
            if ( encoding_type_ == EncodingType::RoPE )
            {
                if ( rope_theta_ <= 0.0f )
                {
                    throw std::invalid_argument( "RoPE theta must be positive" );
                }
            }
        }

        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>(precision_) )
                .set( "embedding_dim", static_cast<int64_t>(embedding_dim_) )
                .set( "num_heads", static_cast<int64_t>(num_heads_) )
                .set( "num_kv_heads", static_cast<int64_t>(num_kv_heads_) )
                .set( "hidden_dim", static_cast<int64_t>(hidden_dim_) )
                .set( "use_bias", use_bias_ )
                .set( "activation", static_cast<int64_t>(activation_type_) )
                .set( "norm_type", static_cast<int64_t>(norm_type_) )
                .set( "attention_type", static_cast<int64_t>(attention_type_) )
                .set( "encoding_type", static_cast<int64_t>(encoding_type_) )
                .set( "rope_theta", static_cast<double>(rope_theta_) )
                .set( "rope_scaling", static_cast<double>(rope_scaling_factor_) )
                .set( "max_seq_len", static_cast<int64_t>(max_seq_len_) )
                .set( "norm_epsilon", static_cast<double>(norm_epsilon_) )
                .set( "residual_scale", static_cast<double>(residual_scale_) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto p = meta.tryGetInt( "precision" ) )
                precision_ = static_cast<decltype(precision_)>(*p);
            if ( auto ed = meta.tryGetInt( "embedding_dim" ) )
                embedding_dim_ = static_cast<dim_t>(*ed);
            if ( auto nh = meta.tryGetInt( "num_heads" ) )
                num_heads_ = static_cast<dim_t>(*nh);
            if ( auto nkv = meta.tryGetInt( "num_kv_heads" ) )
                num_kv_heads_ = static_cast<dim_t>(*nkv);
            if ( auto hd = meta.tryGetInt( "hidden_dim" ) )
                hidden_dim_ = static_cast<dim_t>(*hd);
            if ( auto ub = meta.tryGetBool( "use_bias" ) )
                use_bias_ = *ub;
            if ( auto act = meta.tryGetInt( "activation" ) )
                activation_type_ = static_cast<ActivationType>(*act);
            if ( auto nt = meta.tryGetInt( "norm_type" ) )
                norm_type_ = static_cast<NormType>(*nt);
            if ( auto at = meta.tryGetInt( "attention_type" ) )
                attention_type_ = static_cast<AttentionType>(*at);
            if ( auto pet = meta.tryGetInt( "pos_encoding_type" ) )
                encoding_type_ = static_cast<EncodingType>(*pet);
            if ( auto rt = meta.tryGetFloat( "rope_theta" ) )
                rope_theta_ = static_cast<float>(*rt);
            if ( auto rs = meta.tryGetFloat( "rope_scaling" ) )
                rope_scaling_factor_ = static_cast<float>(*rs);
            if ( auto msl = meta.tryGetInt( "max_seq_len" ) )
                max_seq_len_ = static_cast<dim_t>(*msl);
            if ( auto ne = meta.tryGetFloat( "norm_epsilon" ) )
                norm_epsilon_ = static_cast<float>(*ne);
            if ( auto rscale = meta.tryGetFloat( "residual_scale" ) )
                residual_scale_ = static_cast<float>(*rscale);
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Transformer Configuration:\n";
            oss << "  Embedding Dim: " << embedding_dim_ << "\n";
            oss << "  Num Heads: " << num_heads_ << "\n";
            oss << "  Num KV Heads: " << getNumKVHeads() << "\n";
            oss << "  Hidden Dim: " << getEffectiveHiddenDimension() << "\n";
            oss << "  Use Bias: " << (use_bias_ ? "Yes" : "No") << "\n";
            oss << "  Activation: " << static_cast<int>(activation_type_) << "\n";
            oss << "  Norm Type: " << (norm_type_ == NormType::RMSNorm ? "RMSNorm" : "LayerNorm") << "\n";
            oss << "  Attention Type: ";
            switch ( attention_type_ )
            {
                case AttentionType::Standard: oss << "Standard MHA\n"; break;
                case AttentionType::GroupedQuery: oss << "Grouped Query (GQA)\n"; break;
                case AttentionType::MultiQuery: oss << "Multi Query (MQA)\n"; break;
            }
            oss << "  Positional Encoding: ";
            switch ( encoding_type_ )
            {
                case EncodingType::Learned: oss << "Learned\n"; break;
                case EncodingType::RoPE:
                    oss << "RoPE (theta=" << rope_theta_ << ", scaling=" << rope_scaling_factor_ << ")\n";
                    break;
                case EncodingType::ALiBi: oss << "ALiBi\n"; break;
            }
            return oss.str();
        }

    private:
        // Original fields
        dim_t embedding_dim_;
        dim_t num_heads_;
        dim_t hidden_dim_ = 0;
        bool use_bias_ = false;
        ActivationType activation_type_ = ActivationType::Gelu;
        float residual_scale_ = 1.0f;

        // Component selection fields
        NormType norm_type_ = NormType::LayerNorm;
        AttentionType attention_type_ = AttentionType::Standard;
        dim_t num_kv_heads_ = 0;  // 0 means use num_heads (standard MHA)
        EncodingType encoding_type_ = EncodingType::Learned;
        float rope_theta_ = 10000.0f;
        float rope_scaling_factor_ = 1.0f;
        dim_t max_seq_len_ = 2048;
        float norm_epsilon_ = 1e-5f;
    };
}

/**
 * Usage Example with Presets:
 *
 * // Simple GPT-2 (defaults work)
 * auto config = Presets::GPT2_Small();
 * auto model = Transformer<DeviceType::CUDA, float32>(config);
 *
 * // Llama 3 8B (explicitly configure components)
 * auto config = Presets::Llama3_8B();
 * auto model = Transformer<DeviceType::CUDA, bfloat16>(config);
 * // Your createGraph() will inspect config to build RMSNorm, GQA, SwiGLU, RoPE
 *
 * // Research: Llama architecture with LayerNorm instead of RMSNorm
 * auto config = Presets::Llama3_8B()
 *     .withNormType(NormType::LayerNorm);
 *
 * // Research: GPT-2 but with RoPE instead of learned positions
 * auto config = Presets::GPT2_Small()
 *     .withPositionalEncoding(EncodingType::RoPE)
 *     .withRoPETheta(10000.0f);
 */