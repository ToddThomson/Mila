/**
 * @file Gemma.Config.ixx
 * @brief Network-level configuration for Gemma 4 transformer networks.
 *
 * Gemma 4 decouples per-head width from the residual stream: head_dim is NOT
 * embedding_dim / num_heads (Gemma 4 12B: 3840 / 16 = 240, but head_dim is 256).
 * This config therefore carries head_dim as an explicit, first-class field
 * separate from embedding_dim, where LlamaConfig derives it from the residual
 * dimension. See Specifications/Gemma.md sections 2-3.
 *
 * Step 0 of the Gemma foundation sequence (Gemma.md section 9) is exactly this
 * decoupling. Gemma-specific axes that arrive in later steps are intentionally
 * absent here and noted at their fields:
 *   - global-layer geometry (global_head_dim, num_global_kv_heads, k_eq_v) - Step 1
 *   - dual-RoPE bases / partial-rotary                                      - Step 3
 *   - per-layer sliding/global type, window, logit softcap                 - Steps 2/5
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <cmath>

export module Dnn.Components.GemmaConfig;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Network-level configuration for Gemma 4 transformer networks.
     *
     * Carries the network geometry with head_dim decoupled from the residual
     * stream. The Q-projection width (num_heads * head_dim), the KV-projection
     * width (num_kv_heads * head_dim), and the packed QKV width are derived on
     * demand so they always stay consistent with the primary fields and remain
     * correct when num_heads * head_dim != embedding_dim (the Gemma case).
     *
     * Fluent setters follow the C++23 explicit-object-parameter pattern used
     * throughout the codebase.
     */
    export class GemmaConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Construct a Gemma network configuration.
         *
         * @param embedding_dim Residual-stream dimension (HuggingFace hidden_size). Must be > 0.
         * @param num_layers    Number of transformer layers. Must be > 0.
         */
        GemmaConfig( dim_t embedding_dim, dim_t num_layers )
            : embedding_dim_( embedding_dim ), num_layers_( num_layers )
        {
            if ( embedding_dim <= 0 )
            {
                throw std::invalid_argument( "GemmaConfig: embedding_dim must be > 0" );
            }

            if ( num_layers <= 0 )
            {
                throw std::invalid_argument( "GemmaConfig: num_layers must be > 0" );
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
                throw std::invalid_argument( "GemmaConfig: vocab_size must be > 0" );
            }

            self.vocab_size_ = vocab_size;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withNumHeads( this Self&& self, dim_t num_heads )
        {
            if ( num_heads <= 0 )
            {
                throw std::invalid_argument( "GemmaConfig: num_heads must be > 0" );
            }

            self.num_heads_ = num_heads;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withNumKVHeads( this Self&& self, dim_t num_kv_heads )
        {
            if ( num_kv_heads > 0 && self.num_heads_ > 0 && (self.num_heads_ % num_kv_heads != 0) )
            {
                throw std::invalid_argument( "GemmaConfig: num_heads must be divisible by num_kv_heads" );
            }

            self.num_kv_heads_ = num_kv_heads;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set the per-head dimension, decoupled from the residual stream.
         *
         * This is the Gemma 4 departure from Llama: head_dim is an independent
         * field, not embedding_dim / num_heads. Default 0 means "derive from
         * embedding_dim / num_heads" (the Llama-compatible fallback); Gemma sets
         * it explicitly (256 for the sliding/base geometry).
         */
        template <typename Self>
        decltype(auto) withHeadDim( this Self&& self, dim_t head_dim )
        {
            if ( head_dim < 0 )
            {
                throw std::invalid_argument( "GemmaConfig: head_dim must be >= 0" );
            }

            self.head_dim_ = head_dim;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set the global-layer per-head dimension (Gemma 4 12B: 512).
         *
         * Applies to the 1-in-6 full-attention layers. Default 0 means "use the
         * sliding head_dim" (no distinct global width).
         */
        template <typename Self>
        decltype(auto) withGlobalHeadDim( this Self&& self, dim_t global_head_dim )
        {
            if ( global_head_dim < 0 )
            {
                throw std::invalid_argument( "GemmaConfig: global_head_dim must be >= 0" );
            }

            self.global_head_dim_ = global_head_dim;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set the global-layer KV-head count (Gemma 4 12B: 1, MQA).
         *
         * Default 0 means "use num_kv_heads".
         */
        template <typename Self>
        decltype(auto) withNumGlobalKVHeads( this Self&& self, dim_t num_global_kv_heads )
        {
            if ( num_global_kv_heads > 0 && self.num_heads_ > 0 && (self.num_heads_ % num_global_kv_heads != 0) )
            {
                throw std::invalid_argument( "GemmaConfig: num_heads must be divisible by num_global_kv_heads" );
            }

            self.num_global_kv_heads_ = num_global_kv_heads;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set whether global layers share K=V (no separate v_proj).
         *
         * When true, the global-layer packed QKV width drops to
         * (num_heads + num_global_kv_heads) * global_head_dim; the block aliases
         * the value projection to the key projection.
         */
        template <typename Self>
        decltype(auto) withKeyEqualsValue( this Self&& self, bool key_equals_value )
        {
            self.key_equals_value_ = key_equals_value;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set the FFN intermediate dimension (GeGLU hidden width).
         */
        template <typename Self>
        decltype(auto) withHiddenDimension( this Self&& self, dim_t hidden_dim )
        {
            self.hidden_dim_ = hidden_dim;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set the trained maximum sequence length (HuggingFace max_position_embeddings).
         */
        template <typename Self>
        decltype(auto) withMaxSequenceLength( this Self&& self, dim_t max_seq_len )
        {
            if ( max_seq_len <= 0 )
            {
                throw std::invalid_argument( "GemmaConfig: max_seq_len must be > 0" );
            }

            self.max_seq_len_ = max_seq_len;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withRMSNormEpsilon( this Self&& self, float eps )
        {
            if ( eps <= 0.0f )
            {
                throw std::invalid_argument( "GemmaConfig: rms_norm_eps must be > 0" );
            }

            self.rms_norm_eps_ = eps;
            return std::forward<Self>( self );
        }

        /**
         * @brief Sliding-window size for local (sliding) layers (Gemma 4 12B: 1024).
         */
        template <typename Self>
        decltype(auto) withWindow( this Self&& self, dim_t window )
        {
            if ( window < 0 )
            {
                throw std::invalid_argument( "GemmaConfig: window must be >= 0" );
            }

            self.window_ = window;
            return std::forward<Self>( self );
        }

        /**
         * @brief Whether lm_head shares the token embedding table (weight tying). Set from
         * checkpoint metadata; when true the transformer installs the shared table into
         * lm_head BEFORE build so the head never allocates its own weight (WeightTying.md).
         */
        template <typename Self>
        decltype(auto) withTieWordEmbeddings( this Self&& self, bool tie )
        {
            self.tie_word_embeddings_ = tie;
            return std::forward<Self>( self );
        }

        /**
         * @brief Sliding/global interleave period. Pattern N means every Nth layer is
         * global (full attention) and the rest are sliding; Gemma 4 uses 6 (5 sliding :
         * 1 global), which also makes the final layer global for layer counts that are
         * multiples of the period.
         */
        template <typename Self>
        decltype(auto) withSlidingWindowPattern( this Self&& self, dim_t pattern )
        {
            if ( pattern <= 0 )
            {
                throw std::invalid_argument( "GemmaConfig: sliding_window_pattern must be > 0" );
            }

            self.sliding_window_pattern_ = pattern;
            return std::forward<Self>( self );
        }

        /**
         * @brief Rotated dimension count for global layers (proportional partial-rotary).
         * Gemma 4: 128 of global_head_dim 512. 0 = full rotation.
         */
        template <typename Self>
        decltype(auto) withGlobalRotaryDim( this Self&& self, dim_t global_rotary_dim )
        {
            if ( global_rotary_dim < 0 )
            {
                throw std::invalid_argument( "GemmaConfig: global_rotary_dim must be >= 0" );
            }

            self.global_rotary_dim_ = global_rotary_dim;
            return std::forward<Self>( self );
        }

        /**
         * @brief RoPE base for local (sliding) layers (Gemma 4: 10000).
         */
        template <typename Self>
        decltype(auto) withRoPETheta( this Self&& self, float theta )
        {
            if ( theta <= 0.0f )
            {
                throw std::invalid_argument( "GemmaConfig: rope_theta must be > 0" );
            }

            self.rope_theta_local_ = theta;
            return std::forward<Self>( self );
        }

        /**
         * @brief RoPE base for global (full) layers (Gemma 4: 1e6).
         */
        template <typename Self>
        decltype(auto) withGlobalRoPETheta( this Self&& self, float theta )
        {
            if ( theta <= 0.0f )
            {
                throw std::invalid_argument( "GemmaConfig: global rope_theta must be > 0" );
            }

            self.rope_theta_global_ = theta;
            return std::forward<Self>( self );
        }

        /**
         * @brief Final logit soft-cap value (Gemma 4: 30.0; 0 disables).
         */
        template <typename Self>
        decltype(auto) withFinalLogitSoftcapping( this Self&& self, float cap )
        {
            if ( cap < 0.0f )
            {
                throw std::invalid_argument( "GemmaConfig: final_logit_softcapping must be >= 0" );
            }

            self.final_logit_softcapping_ = cap;
            return std::forward<Self>( self );
        }

        // ====================================================================
        // Primary accessors
        // ====================================================================

        dim_t getVocabSize() const noexcept { return vocab_size_; }
        dim_t getNumLayers() const noexcept { return num_layers_; }

        /**
         * @brief Residual-stream dimension (HuggingFace hidden_size).
         *
         * For Gemma this is NOT num_heads * head_dim. The attention output
         * (num_heads * head_dim) projects back down to this width via a
         * non-square output projection.
         */
        dim_t getModelDim() const noexcept { return embedding_dim_; }

        dim_t getNumHeads() const noexcept { return num_heads_; }

        dim_t getNumKVHeads() const noexcept
        {
            if ( num_kv_heads_ > 0 )
            {
                return num_kv_heads_;
            }

            return num_heads_;
        }

        /**
         * @brief Per-head dimension, decoupled from the residual stream.
         *
         * Returns the explicitly-set head_dim, or falls back to
         * embedding_dim / num_heads when unset (head_dim_ == 0).
         */
        dim_t getHeadDim() const noexcept
        {
            if ( head_dim_ > 0 )
            {
                return head_dim_;
            }

            return (num_heads_ > 0) ? (embedding_dim_ / num_heads_) : 0;
        }

        dim_t getHiddenDimension() const noexcept { return hidden_dim_; }
        dim_t getMaxSequenceLength() const noexcept { return max_seq_len_; }
        float getRMSNormEpsilon() const noexcept { return rms_norm_eps_; }

        /**
         * @brief Global-layer per-head dimension (falls back to head_dim when unset).
         */
        dim_t getGlobalHeadDim() const noexcept
        {
            if ( global_head_dim_ > 0 )
            {
                return global_head_dim_;
            }

            return getHeadDim();
        }

        /**
         * @brief Global-layer KV-head count (falls back to num_kv_heads when unset).
         */
        dim_t getNumGlobalKVHeads() const noexcept
        {
            if ( num_global_kv_heads_ > 0 )
            {
                return num_global_kv_heads_;
            }

            return getNumKVHeads();
        }

        bool keyEqualsValue() const noexcept { return key_equals_value_; }

        // ====================================================================
        // Derived geometry (always consistent with the primary fields, correct
        // when num_heads * head_dim != embedding_dim)
        // ====================================================================

        /**
         * @brief Q-projection output width = num_heads * head_dim.
         *
         * This is the value to feed as GqaConfig/RopeConfig's first constructor
         * argument (both derive head_dim from it). For Gemma it differs from the
         * residual stream (4096 vs 3840 for the sliding geometry).
         */
        dim_t getQProjectionWidth() const noexcept
        {
            return num_heads_ * getHeadDim();
        }

        /**
         * @brief KV-projection output width = num_kv_heads * head_dim.
         */
        dim_t getKVProjectionWidth() const noexcept
        {
            return getNumKVHeads() * getHeadDim();
        }

        /**
         * @brief Packed QKV trailing dimension = (num_heads + 2 * num_kv_heads) * head_dim.
         *
         * The fused QKV projection output width for a standard (non-K=V) layer.
         * The global-layer K=V variant (Step 1) uses (num_heads + num_kv_heads)
         * * head_dim instead.
         */
        dim_t getPackedQKVWidth() const noexcept
        {
            return (num_heads_ + 2 * getNumKVHeads()) * getHeadDim();
        }

        // ---- Global-layer derived widths ------------------------------------

        /**
         * @brief Global-layer Q-projection width = num_heads * global_head_dim.
         */
        dim_t getGlobalQProjectionWidth() const noexcept
        {
            return num_heads_ * getGlobalHeadDim();
        }

        /**
         * @brief Global-layer KV-projection width = num_global_kv_heads * global_head_dim.
         */
        dim_t getGlobalKVProjectionWidth() const noexcept
        {
            return getNumGlobalKVHeads() * getGlobalHeadDim();
        }

        /**
         * @brief Global-layer packed QKV trailing dimension.
         *
         * With K=V (the Gemma global case) the value section is absent, so the
         * width is (num_heads + num_global_kv_heads) * global_head_dim; otherwise
         * it is (num_heads + 2 * num_global_kv_heads) * global_head_dim.
         */
        dim_t getGlobalPackedQKVWidth() const noexcept
        {
            const dim_t kv_sections = key_equals_value_ ? 1 : 2;

            return (num_heads_ + kv_sections * getNumGlobalKVHeads()) * getGlobalHeadDim();
        }

        // ====================================================================
        // Chassis accessors
        // ====================================================================

        dim_t getWindow() const noexcept { return window_; }
        bool getTieWordEmbeddings() const noexcept { return tie_word_embeddings_; }
        dim_t getSlidingWindowPattern() const noexcept { return sliding_window_pattern_; }
        dim_t getGlobalRotaryDim() const noexcept { return global_rotary_dim_; }
        float getRoPEThetaLocal() const noexcept { return rope_theta_local_; }
        float getRoPEThetaGlobal() const noexcept { return rope_theta_global_; }
        float getFinalLogitSoftcapping() const noexcept { return final_logit_softcapping_; }

        /**
         * @brief Embedding scale applied after token lookup: sqrt(embedding_dim).
         *
         * Gemma multiplies the embedded hidden states by this factor; Llama does not.
         */
        float getEmbeddingScale() const noexcept
        {
            return std::sqrt( static_cast<float>( embedding_dim_ ) );
        }

        // ====================================================================
        // Per-layer geometry (the 5:1 sliding/global interleave)
        //
        // Gemma interleaves sliding and global layers; the global layers also change
        // head_dim, KV-head count, K=V sharing, window, and RoPE base/rotary-dim. These
        // helpers resolve the per-layer values the block wires for each instantiation.
        // ====================================================================

        /**
         * @brief True when layer_index is a global (full-attention) layer.
         *
         * Pattern N (sliding_window_pattern) makes every Nth layer global; Gemma 4's
         * period of 6 yields the 5 sliding : 1 global interleave with a global final layer.
         */
        bool isGlobalLayer( dim_t layer_index ) const noexcept
        {
            return ((layer_index + 1) % sliding_window_pattern_) == 0;
        }

        dim_t getHeadDimForLayer( dim_t layer_index ) const noexcept
        {
            return isGlobalLayer( layer_index ) ? getGlobalHeadDim() : getHeadDim();
        }

        dim_t getNumKVHeadsForLayer( dim_t layer_index ) const noexcept
        {
            return isGlobalLayer( layer_index ) ? getNumGlobalKVHeads() : getNumKVHeads();
        }

        bool keyEqualsValueForLayer( dim_t layer_index ) const noexcept
        {
            return isGlobalLayer( layer_index ) ? key_equals_value_ : false;
        }

        /// Sliding layers carry the window; global layers are unbounded (window 0).
        dim_t getWindowForLayer( dim_t layer_index ) const noexcept
        {
            return isGlobalLayer( layer_index ) ? dim_t{ 0 } : window_;
        }

        float getRoPEThetaForLayer( dim_t layer_index ) const noexcept
        {
            return isGlobalLayer( layer_index ) ? rope_theta_global_ : rope_theta_local_;
        }

        /// Global layers use proportional partial-rotary; sliding layers rotate fully (0).
        dim_t getRotaryDimForLayer( dim_t layer_index ) const noexcept
        {
            return isGlobalLayer( layer_index ) ? global_rotary_dim_ : dim_t{ 0 };
        }

        /// Q-projection width for this layer = num_heads * head_dim(layer).
        dim_t getQProjectionWidthForLayer( dim_t layer_index ) const noexcept
        {
            return num_heads_ * getHeadDimForLayer( layer_index );
        }

        /// Packed fused-QKV width for this layer (K=V global layers drop the V section).
        dim_t getPackedQKVWidthForLayer( dim_t layer_index ) const noexcept
        {
            return isGlobalLayer( layer_index ) ? getGlobalPackedQKVWidth() : getPackedQKVWidth();
        }

        // ====================================================================
        // Validation
        // ====================================================================

        void validate() const override
        {
            if ( vocab_size_ <= 0 )
            {
                throw std::invalid_argument( "GemmaConfig: vocab_size must be > 0" );
            }

            if ( num_layers_ <= 0 )
            {
                throw std::invalid_argument( "GemmaConfig: num_layers must be > 0" );
            }

            if ( embedding_dim_ <= 0 )
            {
                throw std::invalid_argument( "GemmaConfig: embedding_dim must be > 0" );
            }

            if ( num_heads_ <= 0 )
            {
                throw std::invalid_argument( "GemmaConfig: num_heads must be > 0" );
            }

            // head_dim is decoupled from embedding_dim - validate it directly
            // rather than asserting embedding_dim % num_heads (the Llama check
            // that bakes in head_dim == embedding_dim / num_heads).
            const dim_t head_dim = getHeadDim();

            if ( head_dim <= 0 )
            {
                throw std::invalid_argument( "GemmaConfig: head_dim must be > 0" );
            }

            if ( head_dim % 2 != 0 )
            {
                throw std::invalid_argument( "GemmaConfig: head_dim must be even (RoPE pairs dimensions)" );
            }

            if ( num_kv_heads_ > 0 && (num_heads_ % num_kv_heads_ != 0) )
            {
                throw std::invalid_argument( "GemmaConfig: num_heads must be divisible by num_kv_heads" );
            }

            const dim_t global_head_dim = getGlobalHeadDim();

            if ( global_head_dim <= 0 )
            {
                throw std::invalid_argument( "GemmaConfig: global_head_dim must be > 0" );
            }

            if ( global_head_dim % 2 != 0 )
            {
                throw std::invalid_argument( "GemmaConfig: global_head_dim must be even (RoPE pairs dimensions)" );
            }

            if ( num_global_kv_heads_ > 0 && (num_heads_ % num_global_kv_heads_ != 0) )
            {
                throw std::invalid_argument( "GemmaConfig: num_heads must be divisible by num_global_kv_heads" );
            }

            if ( rms_norm_eps_ <= 0.0f )
            {
                throw std::invalid_argument( "GemmaConfig: rms_norm_eps must be > 0" );
            }

            if ( window_ < 0 )
            {
                throw std::invalid_argument( "GemmaConfig: window must be >= 0" );
            }

            if ( sliding_window_pattern_ <= 0 )
            {
                throw std::invalid_argument( "GemmaConfig: sliding_window_pattern must be > 0" );
            }

            if ( global_rotary_dim_ < 0 || global_rotary_dim_ > getGlobalHeadDim() )
            {
                throw std::invalid_argument( "GemmaConfig: global_rotary_dim must be in [0, global_head_dim]" );
            }

            if ( rope_theta_local_ <= 0.0f || rope_theta_global_ <= 0.0f )
            {
                throw std::invalid_argument( "GemmaConfig: rope_theta (local and global) must be > 0" );
            }

            if ( final_logit_softcapping_ < 0.0f )
            {
                throw std::invalid_argument( "GemmaConfig: final_logit_softcapping must be >= 0" );
            }
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;

            meta.set( "vocab_size", static_cast<int64_t>(vocab_size_) )
                .set( "num_layers", static_cast<int64_t>(num_layers_) )
                .set( "embedding_dim", static_cast<int64_t>(embedding_dim_) )
                .set( "num_heads", static_cast<int64_t>(num_heads_) )
                .set( "num_kv_heads", static_cast<int64_t>(num_kv_heads_) )
                .set( "head_dim", static_cast<int64_t>(head_dim_) )
                .set( "global_head_dim", static_cast<int64_t>(global_head_dim_) )
                .set( "num_global_kv_heads", static_cast<int64_t>(num_global_kv_heads_) )
                .set( "key_equals_value", key_equals_value_ )
                .set( "hidden_dim", static_cast<int64_t>(hidden_dim_) )
                .set( "max_seq_len", static_cast<int64_t>(max_seq_len_) )
                .set( "rms_norm_eps", static_cast<double>(rms_norm_eps_) )
                .set( "window", static_cast<int64_t>(window_) )
                .set( "sliding_window_pattern", static_cast<int64_t>(sliding_window_pattern_) )
                .set( "global_rotary_dim", static_cast<int64_t>(global_rotary_dim_) )
                .set( "rope_theta_local", static_cast<double>(rope_theta_local_) )
                .set( "rope_theta_global", static_cast<double>(rope_theta_global_) )
                .set( "final_logit_softcapping", static_cast<double>(final_logit_softcapping_) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta )
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

            if ( auto v = meta.tryGetInt( "global_head_dim" ) )
            {
                global_head_dim_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "num_global_kv_heads" ) )
            {
                num_global_kv_heads_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetBool( "key_equals_value" ) )
            {
                key_equals_value_ = *v;
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

            if ( auto v = meta.tryGetInt( "window" ) )
            {
                window_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "sliding_window_pattern" ) )
            {
                sliding_window_pattern_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "global_rotary_dim" ) )
            {
                global_rotary_dim_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetFloat( "rope_theta_local" ) )
            {
                rope_theta_local_ = static_cast<float>(*v);
            }

            if ( auto v = meta.tryGetFloat( "rope_theta_global" ) )
            {
                rope_theta_global_ = static_cast<float>(*v);
            }

            if ( auto v = meta.tryGetFloat( "final_logit_softcapping" ) )
            {
                final_logit_softcapping_ = static_cast<float>(*v);
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Gemma Network Configuration:\n";
            oss << "  Vocab Size: " << vocab_size_ << "\n";
            oss << "  Num Layers: " << num_layers_ << "\n";
            oss << "  Embedding Dim (residual): " << embedding_dim_ << "\n";
            oss << "  Num Heads: " << num_heads_ << "\n";
            oss << "  Num KV Heads: " << getNumKVHeads() << "\n";
            oss << "  Head Dim: " << getHeadDim() << "\n";
            oss << "  Q Projection Width: " << getQProjectionWidth() << "\n";
            oss << "  Global Head Dim: " << getGlobalHeadDim() << "\n";
            oss << "  Num Global KV Heads: " << getNumGlobalKVHeads() << "\n";
            oss << "  Key Equals Value: " << (key_equals_value_ ? "Yes" : "No") << "\n";
            oss << "  Hidden Dim (FFN): " << hidden_dim_ << "\n";
            oss << "  Max Seq Len: " << max_seq_len_ << "\n";
            oss << "  RMSNorm eps: " << rms_norm_eps_ << "\n";
            oss << "  Window: " << window_ << " (pattern " << sliding_window_pattern_ << ", 1 global per N)\n";
            oss << "  RoPE theta (local/global): " << rope_theta_local_ << " / " << rope_theta_global_ << "\n";
            oss << "  Global rotary dim: " << global_rotary_dim_ << "\n";
            oss << "  Final logit softcap: " << final_logit_softcapping_ << "\n";

            return oss.str();
        }

    private:
        dim_t vocab_size_ = 262144;        // Gemma 4 default
        dim_t embedding_dim_ = 3840;       // Gemma 4 12B residual stream
        dim_t num_layers_ = 48;            // Gemma 4 12B
        dim_t num_heads_ = 16;             // Gemma 4 12B query heads
        dim_t num_kv_heads_ = 8;           // Gemma 4 12B sliding-layer KV heads
        dim_t head_dim_ = 256;             // Gemma 4 12B sliding head_dim; 0 = derive embedding_dim / num_heads
        dim_t hidden_dim_ = 15360;         // Gemma 4 12B GeGLU intermediate
        dim_t max_seq_len_ = 262144;       // Gemma 4 256K context
        float rms_norm_eps_ = 1e-6f;       // Gemma 4

        // Global-layer geometry (the 1-in-6 full-attention layers). The global
        // geometry rides the existing GQA op at num_global_kv_heads / global_head_dim;
        // K=V is realized by the block aliasing the V projection to K (Step 1, Gemma.md section 5).
        dim_t global_head_dim_ = 512;      // Gemma 4 12B global head_dim; 0 = use head_dim
        dim_t num_global_kv_heads_ = 1;    // Gemma 4 12B global KV heads (MQA); 0 = use num_kv_heads
        bool  key_equals_value_ = true;    // global layers share K=V (no separate v_proj)
        bool  tie_word_embeddings_ = false; // lm_head shares the embedding table; set from checkpoint metadata

        // Chassis: sliding/global interleave, dual RoPE, logit softcap.
        dim_t window_ = 1024;                       // Gemma 4 12B sliding window
        dim_t sliding_window_pattern_ = 6;          // every 6th layer global (5 sliding : 1 global)
        dim_t global_rotary_dim_ = 128;             // global proportional partial-rotary (of global_head_dim 512)
        float rope_theta_local_ = 10000.0f;         // sliding-layer RoPE base
        float rope_theta_global_ = 1000000.0f;      // global-layer RoPE base (proportional)
        float final_logit_softcapping_ = 30.0f;     // tanh logit soft-cap; 0 disables
    };
}
