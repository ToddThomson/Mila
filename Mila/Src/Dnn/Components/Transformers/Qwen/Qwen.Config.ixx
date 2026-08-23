/**
 * @file Qwen.Config.ixx
 * @brief Network-level configuration for Qwen 3.8 transformer networks.
 *
 * Carries the published `Qwen/Qwen3.8-27B` geometry (Specifications/Qwen3.8.md section 1).
 * Two things separate it from GemmaConfig, and both are the reason a distinct family exists:
 *
 *  - The interleave is over MIXERS, not geometries. Gemma alternates two attention layers
 *    that differ in head width and window; Qwen alternates full attention with a Gated
 *    DeltaNet recurrence, which shares no projection shape with it at all. So the DeltaNet
 *    geometry is its own set of fields rather than a "global" variant of the attention ones.
 *  - The query projection is DOUBLE WIDTH. `attn_output_gate` makes it emit [query | gate],
 *    which changes the packed QKV width and the split the block performs.
 *
 * What is deliberately NOT here:
 *  - The gate function. It selects a type, so it rides the AttentionOutputGate template
 *    parameter, per the discriminating principle in Qwen3.8.md's overview: template axes are
 *    for types and layouts, runtime config is for arithmetic. Note the published
 *    `output_gate_type: "swish"` is a DEAD key -- the reference implementation reads it
 *    nowhere and applies sigmoid -- so it would be the wrong thing to carry here anyway.
 *  - The mrope section split [11, 11, 10]. It is a multimodal positional layout, and the
 *    vision tower is out of scope for this chassis; text-only mrope degenerates to standard
 *    RoPE because all three sections then carry the same position ids.
 *  - The MTP draft head and the vision tower (Qwen3.8.md section 1, "out of scope").
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>

export module Dnn.Components.QwenConfig;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Network-level configuration for Qwen 3.8 transformer networks.
     *
     * Fluent setters follow the C++23 explicit-object-parameter pattern used throughout the
     * codebase. Defaults are the published 27B values, so a default-constructed config is
     * the real model rather than a placeholder.
     */
    export class QwenConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Construct a Qwen network configuration.
         *
         * @param embedding_dim Residual-stream dimension (HuggingFace hidden_size). Must be > 0.
         * @param num_layers    Total transformer layers, full-attention and DeltaNet. Must be > 0.
         */
        QwenConfig( dim_t embedding_dim, dim_t num_layers )
            : embedding_dim_( embedding_dim ), num_layers_( num_layers )
        {
            if ( embedding_dim <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: embedding_dim must be > 0" );
            }

            if ( num_layers <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: num_layers must be > 0" );
            }
        }

        // ====================================================================
        // Fluent setters
        // ====================================================================

        template <typename Self>
        decltype(auto) withVocabularyLength( this Self&& self, dim_t vocab_size )
        {
            if ( vocab_size <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: vocab_size must be > 0" );
            }

            self.vocab_size_ = vocab_size;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withNumHeads( this Self&& self, dim_t num_heads )
        {
            if ( num_heads <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: num_heads must be > 0" );
            }

            self.num_heads_ = num_heads;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withNumKVHeads( this Self&& self, dim_t num_kv_heads )
        {
            if ( num_kv_heads > 0 && self.num_heads_ > 0 && (self.num_heads_ % num_kv_heads != 0) )
            {
                throw std::invalid_argument( "QwenConfig: num_heads must be divisible by num_kv_heads" );
            }

            self.num_kv_heads_ = num_kv_heads;
            return std::forward<Self>( self );
        }

        /**
         * @brief Per-head dimension, decoupled from the residual stream (5120 / 24 != 256).
         */
        template <typename Self>
        decltype(auto) withHeadDim( this Self&& self, dim_t head_dim )
        {
            if ( head_dim <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: head_dim must be > 0" );
            }

            self.head_dim_ = head_dim;
            return std::forward<Self>( self );
        }

        /**
         * @brief Whether the query projection is double width and emits [query | gate].
         *
         * The published 27B sets this true. False is the plain-GQA geometry and exists so a
         * test can pin the packed width against the ungated case.
         */
        template <typename Self>
        decltype(auto) withAttentionOutputGate( this Self&& self, bool attention_output_gate )
        {
            self.attention_output_gate_ = attention_output_gate;
            return std::forward<Self>( self );
        }

        /**
         * @brief SwiGLU FFN intermediate width, uniform across both layer kinds.
         */
        template <typename Self>
        decltype(auto) withHiddenDimension( this Self&& self, dim_t hidden_dim )
        {
            if ( hidden_dim <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: hidden_dim must be > 0" );
            }

            self.hidden_dim_ = hidden_dim;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withMaxSequenceLength( this Self&& self, dim_t max_seq_len )
        {
            if ( max_seq_len <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: max_seq_len must be > 0" );
            }

            self.max_seq_len_ = max_seq_len;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withRMSNormEpsilon( this Self&& self, float eps )
        {
            if ( eps <= 0.0f )
            {
                throw std::invalid_argument( "QwenConfig: rms_norm_eps must be > 0" );
            }

            self.rms_norm_eps_ = eps;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withRoPETheta( this Self&& self, float theta )
        {
            if ( theta <= 0.0f )
            {
                throw std::invalid_argument( "QwenConfig: rope_theta must be > 0" );
            }

            self.rope_theta_ = theta;
            return std::forward<Self>( self );
        }

        /**
         * @brief Fraction of head_dim that rotates (0.25 on the 27B: 64 of 256).
         *
         * Stored as the published fraction rather than the resolved width so the config
         * reads as the checkpoint does; getRotaryDim() resolves it.
         */
        template <typename Self>
        decltype(auto) withPartialRotaryFactor( this Self&& self, float factor )
        {
            if ( factor <= 0.0f || factor > 1.0f )
            {
                throw std::invalid_argument( "QwenConfig: partial_rotary_factor must be in (0, 1]" );
            }

            self.partial_rotary_factor_ = factor;
            return std::forward<Self>( self );
        }

        /**
         * @brief Layer period at which a full-attention layer appears (27B: 4).
         *
         * Period N gives (N-1) DeltaNet layers per full-attention layer; the 27B's 4 yields
         * the 3:1 interleave and 16 cache-holding layers of 64.
         */
        template <typename Self>
        decltype(auto) withFullAttentionInterval( this Self&& self, dim_t interval )
        {
            if ( interval <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: full_attention_interval must be > 0" );
            }

            self.full_attention_interval_ = interval;
            return std::forward<Self>( self );
        }

        /**
         * @brief Whether lm_head shares the token embedding table. The 27B is UNTIED.
         */
        template <typename Self>
        decltype(auto) withTieWordEmbeddings( this Self&& self, bool tie )
        {
            self.tie_word_embeddings_ = tie;
            return std::forward<Self>( self );
        }

        // ---- Gated DeltaNet geometry (consumed at Phase 3) -------------------

        template <typename Self>
        decltype(auto) withLinearNumKeyHeads( this Self&& self, dim_t num_key_heads )
        {
            if ( num_key_heads <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: linear_num_key_heads must be > 0" );
            }

            self.linear_num_key_heads_ = num_key_heads;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withLinearNumValueHeads( this Self&& self, dim_t num_value_heads )
        {
            if ( num_value_heads <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: linear_num_value_heads must be > 0" );
            }

            self.linear_num_value_heads_ = num_value_heads;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withLinearHeadDim( this Self&& self, dim_t head_dim )
        {
            if ( head_dim <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: linear_head_dim must be > 0" );
            }

            self.linear_head_dim_ = head_dim;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withLinearConvKernelDim( this Self&& self, dim_t kernel_dim )
        {
            if ( kernel_dim <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: linear_conv_kernel_dim must be > 0" );
            }

            self.linear_conv_kernel_dim_ = kernel_dim;
            return std::forward<Self>( self );
        }

        // ====================================================================
        // Primary accessors
        // ====================================================================

        dim_t getVocabSize() const noexcept { return vocab_size_; }
        dim_t getNumLayers() const noexcept { return num_layers_; }

        /**
         * @brief Residual-stream dimension. NOT num_heads * head_dim (5120 vs 6144).
         */
        dim_t getModelDim() const noexcept { return embedding_dim_; }

        dim_t getNumHeads() const noexcept { return num_heads_; }
        dim_t getNumKVHeads() const noexcept { return num_kv_heads_ > 0 ? num_kv_heads_ : num_heads_; }
        dim_t getHeadDim() const noexcept { return head_dim_; }
        dim_t getHiddenDimension() const noexcept { return hidden_dim_; }
        dim_t getMaxSequenceLength() const noexcept { return max_seq_len_; }
        float getRMSNormEpsilon() const noexcept { return rms_norm_eps_; }
        float getRoPETheta() const noexcept { return rope_theta_; }
        float getPartialRotaryFactor() const noexcept { return partial_rotary_factor_; }
        bool hasAttentionOutputGate() const noexcept { return attention_output_gate_; }
        dim_t getFullAttentionInterval() const noexcept { return full_attention_interval_; }
        bool getTieWordEmbeddings() const noexcept { return tie_word_embeddings_; }

        dim_t getLinearNumKeyHeads() const noexcept { return linear_num_key_heads_; }
        dim_t getLinearNumValueHeads() const noexcept { return linear_num_value_heads_; }
        dim_t getLinearHeadDim() const noexcept { return linear_head_dim_; }
        dim_t getLinearConvKernelDim() const noexcept { return linear_conv_kernel_dim_; }

        // ====================================================================
        // Derived geometry
        // ====================================================================

        /**
         * @brief Rotated dimension count = partial_rotary_factor * head_dim (27B: 64).
         *
         * Truncated to an even width because RoPE pairs dimensions.
         */
        dim_t getRotaryDim() const noexcept
        {
            const dim_t rotary = static_cast<dim_t>( partial_rotary_factor_ * static_cast<float>( head_dim_ ) );

            return rotary - (rotary % 2);
        }

        /// Attention output width = num_heads * head_dim. What o_proj reads.
        dim_t getQProjectionWidth() const noexcept { return num_heads_ * head_dim_; }

        /**
         * @brief Query-projection OUTPUT width, doubled when the output gate is present.
         *
         * The distinction from getQProjectionWidth() is the whole of `attn_output_gate`:
         * the projection emits this many columns, the attention consumes half of them, and
         * the other half is the gate.
         */
        dim_t getGatedQProjectionWidth() const noexcept
        {
            return (attention_output_gate_ ? 2 : 1) * getQProjectionWidth();
        }

        /// K and V projection width each = num_kv_heads * head_dim.
        dim_t getKVProjectionWidth() const noexcept { return getNumKVHeads() * head_dim_; }

        /**
         * @brief Fused QKV projection width = gated Q width + 2 * KV width.
         *
         * The block splits this into [query|gate], key, value. The GQA operation's own
         * packed convention is the ungated (num_heads + 2 * num_kv_heads) * head_dim, which
         * is what getAttentionPackedQKVWidth() reports -- the two differ exactly by the
         * gate half, and confusing them is how the gate goes missing.
         */
        dim_t getPackedQKVWidth() const noexcept
        {
            return getGatedQProjectionWidth() + 2 * getKVProjectionWidth();
        }

        /// The width the GQA operation validates its build context against (no gate half).
        dim_t getAttentionPackedQKVWidth() const noexcept
        {
            return getQProjectionWidth() + 2 * getKVProjectionWidth();
        }

        /// DeltaNet key (and query) stream width = linear_num_key_heads * linear_head_dim.
        dim_t getDeltaNetKeyWidth() const noexcept
        {
            return linear_num_key_heads_ * linear_head_dim_;
        }

        /// Fused DeltaNet [query|key] projection width -- twice the key width.
        dim_t getDeltaNetQueryKeyWidth() const noexcept
        {
            return 2 * getDeltaNetKeyWidth();
        }

        /// DeltaNet value stream width = linear_num_value_heads * linear_head_dim. Also the
        /// width of z, of the mixer output, and of the gated result.
        dim_t getDeltaNetValueWidth() const noexcept
        {
            return linear_num_value_heads_ * linear_head_dim_;
        }

        /// DeltaNet gating width -- one scalar per value head, for each of a and b.
        dim_t getDeltaNetGatingWidth() const noexcept
        {
            return linear_num_value_heads_;
        }

        // ====================================================================
        // Per-layer kind (the 3 DeltaNet : 1 full-attention interleave)
        // ====================================================================

        /**
         * @brief True when layer_index holds full attention (and therefore a KV cache).
         *
         * Period N makes every Nth layer full attention; the 27B's 4 places them at layers
         * 3, 7, ... 63, so the final layer is full attention and 16 of 64 hold a cache --
         * the property Qwen3.8.md section 3 turns into a 12 GiB budget.
         */
        bool isFullAttentionLayer( dim_t layer_index ) const noexcept
        {
            return ((layer_index + 1) % full_attention_interval_) == 0;
        }

        /// Count of full-attention layers, i.e. of KV-cache-holding layers.
        dim_t getNumFullAttentionLayers() const noexcept
        {
            return num_layers_ / full_attention_interval_;
        }

        /// Count of Gated DeltaNet layers, i.e. of recurrent-state-holding layers.
        dim_t getNumDeltaNetLayers() const noexcept
        {
            return num_layers_ - getNumFullAttentionLayers();
        }

        // ====================================================================
        // Validation
        // ====================================================================

        void validate() const override
        {
            if ( vocab_size_ <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: vocab_size must be > 0" );
            }

            if ( num_layers_ <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: num_layers must be > 0" );
            }

            if ( embedding_dim_ <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: embedding_dim must be > 0" );
            }

            if ( num_heads_ <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: num_heads must be > 0" );
            }

            // head_dim is decoupled from embedding_dim, so it is validated directly rather
            // than through an embedding_dim % num_heads check that would bake in coupling
            // this model does not have.
            if ( head_dim_ <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: head_dim must be > 0" );
            }

            if ( head_dim_ % 2 != 0 )
            {
                throw std::invalid_argument( "QwenConfig: head_dim must be even (RoPE pairs dimensions)" );
            }

            if ( num_kv_heads_ > 0 && (num_heads_ % num_kv_heads_ != 0) )
            {
                throw std::invalid_argument( "QwenConfig: num_heads must be divisible by num_kv_heads" );
            }

            if ( hidden_dim_ <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: hidden_dim must be > 0" );
            }

            if ( rms_norm_eps_ <= 0.0f )
            {
                throw std::invalid_argument( "QwenConfig: rms_norm_eps must be > 0" );
            }

            if ( rope_theta_ <= 0.0f )
            {
                throw std::invalid_argument( "QwenConfig: rope_theta must be > 0" );
            }

            if ( partial_rotary_factor_ <= 0.0f || partial_rotary_factor_ > 1.0f )
            {
                throw std::invalid_argument( "QwenConfig: partial_rotary_factor must be in (0, 1]" );
            }

            if ( getRotaryDim() <= 0 )
            {
                throw std::invalid_argument(
                    "QwenConfig: partial_rotary_factor * head_dim resolves to no rotated dimensions" );
            }

            if ( full_attention_interval_ <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: full_attention_interval must be > 0" );
            }

            if ( linear_num_key_heads_ <= 0 || linear_num_value_heads_ <= 0 || linear_head_dim_ <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: DeltaNet head counts and head_dim must be > 0" );
            }

            if ( linear_num_value_heads_ % linear_num_key_heads_ != 0 )
            {
                throw std::invalid_argument(
                    "QwenConfig: linear_num_value_heads must be divisible by linear_num_key_heads" );
            }

            if ( linear_conv_kernel_dim_ <= 0 )
            {
                throw std::invalid_argument( "QwenConfig: linear_conv_kernel_dim must be > 0" );
            }
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "vocab_size", static_cast<int64_t>(vocab_size_) )
                .set( "num_layers", static_cast<int64_t>(num_layers_) )
                .set( "embedding_dim", static_cast<int64_t>(embedding_dim_) )
                .set( "num_heads", static_cast<int64_t>(num_heads_) )
                .set( "num_kv_heads", static_cast<int64_t>(num_kv_heads_) )
                .set( "head_dim", static_cast<int64_t>(head_dim_) )
                .set( "attention_output_gate", attention_output_gate_ )
                .set( "hidden_dim", static_cast<int64_t>(hidden_dim_) )
                .set( "max_seq_len", static_cast<int64_t>(max_seq_len_) )
                .set( "rms_norm_eps", static_cast<double>(rms_norm_eps_) )
                .set( "rope_theta", static_cast<double>(rope_theta_) )
                .set( "partial_rotary_factor", static_cast<double>(partial_rotary_factor_) )
                .set( "full_attention_interval", static_cast<int64_t>(full_attention_interval_) )
                .set( "tie_word_embeddings", tie_word_embeddings_ )
                .set( "linear_num_key_heads", static_cast<int64_t>(linear_num_key_heads_) )
                .set( "linear_num_value_heads", static_cast<int64_t>(linear_num_value_heads_) )
                .set( "linear_head_dim", static_cast<int64_t>(linear_head_dim_) )
                .set( "linear_conv_kernel_dim", static_cast<int64_t>(linear_conv_kernel_dim_) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "vocab_size" ) )
            {
                vocab_size_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "num_layers" ) )
            {
                num_layers_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "embedding_dim" ) )
            {
                embedding_dim_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "num_heads" ) )
            {
                num_heads_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "num_kv_heads" ) )
            {
                num_kv_heads_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "head_dim" ) )
            {
                head_dim_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetBool( "attention_output_gate" ) )
            {
                attention_output_gate_ = *v;
            }

            if ( auto v = meta.tryGetInt( "hidden_dim" ) )
            {
                hidden_dim_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "max_seq_len" ) )
            {
                max_seq_len_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetFloat( "rms_norm_eps" ) )
            {
                rms_norm_eps_ = static_cast<float>(*v);
            }

            if ( auto v = meta.tryGetFloat( "rope_theta" ) )
            {
                rope_theta_ = static_cast<float>(*v);
            }

            if ( auto v = meta.tryGetFloat( "partial_rotary_factor" ) )
            {
                partial_rotary_factor_ = static_cast<float>(*v);
            }

            if ( auto v = meta.tryGetInt( "full_attention_interval" ) )
            {
                full_attention_interval_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetBool( "tie_word_embeddings" ) )
            {
                tie_word_embeddings_ = *v;
            }

            if ( auto v = meta.tryGetInt( "linear_num_key_heads" ) )
            {
                linear_num_key_heads_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "linear_num_value_heads" ) )
            {
                linear_num_value_heads_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "linear_head_dim" ) )
            {
                linear_head_dim_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "linear_conv_kernel_dim" ) )
            {
                linear_conv_kernel_dim_ = static_cast<dim_t>(*v);
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Qwen Network Configuration:\n";
            oss << "  Vocab Size: " << vocab_size_ << "\n";
            oss << "  Num Layers: " << num_layers_ << " ("
                << getNumFullAttentionLayers() << " full attention, "
                << getNumDeltaNetLayers() << " Gated DeltaNet)\n";
            oss << "  Embedding Dim (residual): " << embedding_dim_ << "\n";
            oss << "  Num Heads: " << num_heads_ << "\n";
            oss << "  Num KV Heads: " << getNumKVHeads() << "\n";
            oss << "  Head Dim: " << head_dim_ << "\n";
            oss << "  Attention Output Gate: " << (attention_output_gate_ ? "Yes" : "No") << "\n";
            oss << "  Q Projection Width: " << getQProjectionWidth()
                << " (projected " << getGatedQProjectionWidth() << ")\n";
            oss << "  Hidden Dim (FFN): " << hidden_dim_ << "\n";
            oss << "  Max Seq Len: " << max_seq_len_ << "\n";
            oss << "  RMSNorm eps: " << rms_norm_eps_ << "\n";
            oss << "  RoPE theta: " << rope_theta_ << "\n";
            oss << "  Rotary dim: " << getRotaryDim() << " of " << head_dim_ << "\n";
            oss << "  Full attention interval: " << full_attention_interval_ << "\n";
            oss << "  Tie Word Embeddings: " << (tie_word_embeddings_ ? "Yes" : "No") << "\n";
            oss << "  DeltaNet: " << linear_num_key_heads_ << " key heads, "
                << linear_num_value_heads_ << " value heads, head_dim " << linear_head_dim_
                << ", conv kernel " << linear_conv_kernel_dim_ << "\n";

            return oss.str();
        }

    private:
        dim_t vocab_size_ = 248320;        // Qwen 3.8 27B
        dim_t embedding_dim_ = 5120;       // residual stream
        dim_t num_layers_ = 64;
        dim_t num_heads_ = 24;             // full-attention query heads
        dim_t num_kv_heads_ = 4;           // full-attention KV heads (GQA group 6)
        dim_t head_dim_ = 256;             // decoupled: 5120 / 24 = 213, not 256
        bool  attention_output_gate_ = true;  // q projection emits [query | gate]
        dim_t hidden_dim_ = 17408;         // SwiGLU intermediate, both layer kinds
        dim_t max_seq_len_ = 262144;       // extensible to 1M via YaRN, not modeled here
        float rms_norm_eps_ = 1e-6f;
        float rope_theta_ = 1e7f;
        float partial_rotary_factor_ = 0.25f;  // 64 of head_dim 256
        dim_t full_attention_interval_ = 4;    // 3 DeltaNet : 1 full attention
        bool  tie_word_embeddings_ = false;    // UNTIED: two full-size tables

        // Gated DeltaNet geometry. Present from the start because it is part of the
        // published config and the transformer must know which layers are which; the
        // component that consumes it arrives at Phase 3.
        dim_t linear_num_key_heads_ = 16;
        dim_t linear_num_value_heads_ = 48;
        dim_t linear_head_dim_ = 128;
        dim_t linear_conv_kernel_dim_ = 4;     // short causal depthwise convolution over q/k/v
    };
}
