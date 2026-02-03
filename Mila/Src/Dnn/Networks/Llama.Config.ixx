/**
 * @file Llama.Config.ixx
 * @brief LLaMA network-level configuration
 *
 * Provides only the network-level settings required for LLaMA networks.
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>

export module Dnn.Networks.Llama:Config;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn::Networks
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Network-level configuration for LLaMA-style transformer networks.
     *
     * Exposes only the settings needed at network scope:
     * vocabulary, number of layers, embedding dimension, and max sequence length.
     */
    export class LlamaConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Construct a LLaMA network configuration.
         *
         * @param embedding_dim Model embedding dimension. Must be > 0.
         * @param num_layers Number of transformer layers. Must be > 0.
         */
        LlamaConfig( dim_t embedding_dim, dim_t num_layers )
            : embedding_dim_( embedding_dim ), num_layers_( num_layers )
        {
            if ( embedding_dim <= 0 )
            {
                throw std::invalid_argument( "LlamaConfig: embedding_dim must be > 0" );
            }

            if ( num_layers <= 0 )
            {
                throw std::invalid_argument( "LlamaConfig: num_layers must be > 0" );
            }
        }

        template <typename Self>
        decltype(auto) withVocabSize( this Self&& self, dim_t vocab_size )
        {
            if ( vocab_size <= 0 )
            {
                throw std::invalid_argument( "vocab_size must be > 0" );
            }

            self.vocab_size_ = vocab_size;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withNumHeads( this Self&& self, dim_t num_heads )
        {
            if ( num_heads <= 0 )
            {
                throw std::invalid_argument( "num_heads must be > 0" );
            }

            if ( self.embedding_dim_ > 0 && (self.embedding_dim_ % num_heads != 0) )
            {
                throw std::invalid_argument( "num_heads must divide embedding_dim" );
            }

            self.num_heads_ = num_heads;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withNumKVHeads( this Self&& self, dim_t num_kv_heads )
        {
            if ( num_kv_heads > 0 && self.num_heads_ > 0 && (self.num_heads_ % num_kv_heads != 0) )
            {
                throw std::invalid_argument( "num_heads must be divisible by num_kv_heads" );
            }

            self.num_kv_heads_ = num_kv_heads;
            return std::forward<Self>( self );
        }

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

        template <typename Self>
        decltype(auto) withHiddenDimension( this Self&& self, dim_t hidden_dim )
        {
            self.hidden_dim_ = hidden_dim;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withRoPETheta( this Self&& self, float theta )
        {
            if ( theta <= 0.0f )
            {
                throw std::invalid_argument( "rope_theta must be > 0" );
            }

            self.rope_theta_ = theta;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withRoPEScalingFactor( this Self&& self, float scale_factor )
        {
            if ( scale_factor <= 0.0f )
            {
                throw std::invalid_argument( "rope_scaling_factor must be > 0" );
            }

            self.rope_scaling_factor_ = scale_factor;
            return std::forward<Self>( self );
        }

        // New: control bias on linear projections (network-level default for LLaMA is false)
        template <typename Self>
        decltype(auto) withBias( this Self&& self, bool use_bias )
        {
            self.use_bias_ = use_bias;
            return std::forward<Self>( self );
        }

        // --- Getters (network-level and commonly used block-related values) ---

        dim_t getVocabSize() const noexcept { return vocab_size_; }
        dim_t getNumLayers() const noexcept { return num_layers_; }
        dim_t getEmbeddingSize() const noexcept { return embedding_dim_; }
        dim_t getMaxSequenceLength() const noexcept { return max_seq_len_; }

        dim_t getNumHeads() const noexcept { return num_heads_; }

        dim_t getNumKVHeads() const noexcept
        {
            // If num_kv_heads not set, fallback to num_heads
            if ( num_kv_heads_ > 0 )
            {
                return num_kv_heads_;
            }

            return num_heads_;
        }

        dim_t getHiddenDimension() const noexcept { return hidden_dim_; }

        float getRoPETheta() const noexcept { return rope_theta_; }

        float getRoPEScalingFactor() const noexcept { return rope_scaling_factor_; }

        float getRMSNormEpsilon() const noexcept { return rms_norm_eps_; }

        // New: expose bias flag
        bool useBias() const noexcept { return use_bias_; }

        // Validation ensures network-level constraints
        void validate() const override
        {
            if ( vocab_size_ <= 0 )
            {
                throw std::invalid_argument( "vocab_size must be greater than zero" );
            }

            if ( num_layers_ <= 0 )
            {
                throw std::invalid_argument( "num_layers must be greater than zero" );
            }

            if ( embedding_dim_ <= 0 )
            {
                throw std::invalid_argument( "embedding_dim must be greater than zero" );
            }

            if ( num_heads_ > 0 )
            {
                if ( embedding_dim_ % num_heads_ != 0 )
                {
                    throw std::invalid_argument( "embedding_dim must be divisible by num_heads" );
                }
            }

            if ( num_kv_heads_ > 0 && num_heads_ > 0 )
            {
                if ( num_heads_ % num_kv_heads_ != 0 )
                {
                    throw std::invalid_argument( "num_heads must be divisible by num_kv_heads" );
                }
            }

            if ( rope_theta_ <= 0.0f )
            {
                throw std::invalid_argument( "roPE theta must be positive" );
            }

            if ( rope_scaling_factor_ <= 0.0f )
            {
                throw std::invalid_argument( "roPE scaling factor must be positive" );
            }
        }

        // Serialization helpers

        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>(precision_) )
                .set( "vocab_size", static_cast<int64_t>(vocab_size_) )
                .set( "num_layers", static_cast<int64_t>(num_layers_) )
                .set( "embedding_dim", static_cast<int64_t>(embedding_dim_) )
                .set( "max_seq_len", static_cast<int64_t>(max_seq_len_) )
                .set( "num_heads", static_cast<int64_t>(num_heads_) )
                .set( "num_kv_heads", static_cast<int64_t>(num_kv_heads_) )
                .set( "hidden_dim", static_cast<int64_t>(hidden_dim_) )
                .set( "rope_theta", static_cast<double>(rope_theta_) )
                .set( "rope_scaling", static_cast<double>(rope_scaling_factor_) )
                .set( "rms_norm_eps", static_cast<double>(rms_norm_eps_) )
                .set( "use_bias", use_bias_ );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto p = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype(precision_)>(*p);
            }

            if ( auto vs = meta.tryGetInt( "vocab_size" ) )
            {
                vocab_size_ = static_cast<dim_t>(*vs);
            }

            if ( auto nl = meta.tryGetInt( "num_layers" ) )
            {
                num_layers_ = static_cast<dim_t>(*nl);
            }

            if ( auto ed = meta.tryGetInt( "embedding_dim" ) )
            {
                embedding_dim_ = static_cast<dim_t>(*ed);
            }

            if ( auto msl = meta.tryGetInt( "max_seq_len" ) )
            {
                max_seq_len_ = static_cast<dim_t>(*msl);
            }

            if ( auto nh = meta.tryGetInt( "num_heads" ) )
            {
                num_heads_ = static_cast<dim_t>(*nh);
            }

            if ( auto nkv = meta.tryGetInt( "num_kv_heads" ) )
            {
                num_kv_heads_ = static_cast<dim_t>(*nkv);
            }

            if ( auto hd = meta.tryGetInt( "hidden_dim" ) )
            {
                hidden_dim_ = static_cast<dim_t>(*hd);
            }

            if ( auto rt = meta.tryGetFloat( "rope_theta" ) )
            {
                rope_theta_ = static_cast<float>(*rt);
            }

            if ( auto rs = meta.tryGetFloat( "rope_scaling" ) )
            {
                rope_scaling_factor_ = static_cast<float>(*rs);
            }

            if ( auto re = meta.tryGetFloat( "rms_norm_eps" ) )
            {
                rms_norm_eps_ = static_cast<float>(*re);
            }

            if ( auto ub = meta.tryGetBool( "use_bias" ) )
            {
                use_bias_ = *ub;
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "LLaMA Network Configuration:\n";
            oss << "  Vocab Size: " << vocab_size_ << "\n";
            oss << "  Num Layers: " << num_layers_ << "\n";
            oss << "  Embedding Dim: " << embedding_dim_ << "\n";
            oss << "  Num Heads: " << num_heads_ << "\n";
            oss << "  Num KV Heads: " << getNumKVHeads() << "\n";
            oss << "  Hidden Dim: " << hidden_dim_ << "\n";
            oss << "  Max Seq Len: " << max_seq_len_ << "\n";
            oss << "  RoPE: theta=" << rope_theta_ << ", scaling=" << rope_scaling_factor_ << "\n";
            oss << "  RMSNorm eps: " << rms_norm_eps_ << "\n";
            oss << "  Use Bias: " << ( use_bias_ ? "Yes" : "No" ) << "\n";

            return oss.str();
        }

    private:
        dim_t vocab_size_ = 128256;        // Llama 3/3.1 default (was 32000 for Llama 2)
        dim_t embedding_dim_ = 4096;       // Llama 3 8B default
        dim_t num_layers_ = 32;            // Llama 3 8B default
        dim_t num_heads_ = 32;             // Llama 3 8B default
        dim_t num_kv_heads_ = 8;           // Llama 3 8B default (GQA)
        dim_t max_seq_len_ = 8192;         // Llama 3 8B default
        dim_t hidden_dim_ = 14336;         // Llama 3 8B default (SwiGLU)
        float rope_theta_ = 500000.0f;
        float rope_scaling_factor_ = 1.0f;
        float rms_norm_eps_ = 1e-5f;
        bool use_bias_ = false;
    };
}

