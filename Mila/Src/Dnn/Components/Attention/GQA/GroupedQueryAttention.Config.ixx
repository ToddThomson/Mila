/**
 * @file GroupedQueryAttention.Config.ixx
 * @brief Configuration interface for the Grouped-Query Attention component.
 *
 * Grouped-Query Attention (GQA) extends Multi-Head Attention by decoupling
 * the number of Q heads from the number of K/V heads.  Each K/V head is
 * shared by a contiguous group of Q heads, reducing KV cache size and memory
 * bandwidth during inference proportionally to (num_heads / num_kv_heads).
 *
 * Special cases:
 *   num_kv_heads == num_heads  ->  standard Multi-Head Attention
 *   num_kv_heads == 1          ->  Multi-Query Attention (MQA)
 */

module;
#include <stdexcept>
#include <string>
#include <utility>
#include <sstream>
#include <cmath>

export module Dnn.Components.GqaConfig;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.TensorTypes;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration class for the Grouped-Query Attention module.
     *
     * Carries the three parameters that uniquely define a GQA layer:
     *   - model_dim   : total Q-projection output width  (= num_heads * head_dim)
     *   - num_heads   : number of Q attention heads
     *   - num_kv_heads: number of K/V attention heads    (must divide num_heads)
     *
     * The derived head_dim = model_dim / num_heads and group_size =
     * num_heads / num_kv_heads are computed on demand rather than stored, so
     * they always stay consistent with the three primary fields.
     *
     * Fluent setters follow the C++23 explicit-object-parameter pattern used
     * throughout the codebase, enabling value-category-preserving method
     * chaining on both lvalue and rvalue configs.
     */
    export class GqaConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Constructor with all required parameters.
         *
         * @param model_dim    Total Q-projection output width.
         * @param num_heads    Number of Q attention heads.
         * @param num_kv_heads Number of K/V attention heads.
         */
        GqaConfig( dim_t model_dim, dim_t num_heads, dim_t num_kv_heads )
            : model_dim_( model_dim ), num_heads_( num_heads ), num_kv_heads_( num_kv_heads )
        {}

        // ====================================================================
        // Fluent setters (C++23 explicit object parameter)
        // ====================================================================

        /**
         * @brief Fluent setter for model dimension.
         */
        template <typename Self>
        decltype(auto) withModelDim( this Self&& self, dim_t model_dim )
        {
            self.model_dim_ = model_dim;
            return std::forward<Self>( self );
        }

        /**
         * @brief Fluent setter for number of Q heads.
         */
        template <typename Self>
        decltype(auto) withNumHeads( this Self&& self, dim_t num_heads )
        {
            self.num_heads_ = num_heads;
            return std::forward<Self>( self );
        }

        /**
         * @brief Fluent setter for number of K/V heads.
         */
        template <typename Self>
        decltype(auto) withNumKvHeads( this Self&& self, dim_t num_kv_heads )
        {
            self.num_kv_heads_ = num_kv_heads;
            return std::forward<Self>( self );
        }

        /**
         * @brief Fluent setter for the sliding-attention window size.
         *
         * The number of most-recent keys a query may attend to. 0 (the default)
         * means unbounded/global causal attention - every query attends over
         * [0, abs_t], which is the behavior for Llama / Qwen / Gemma global layers.
         * A positive value bounds the lower edge to [abs_t - window + 1, abs_t]
         * (Gemma sliding layers: 1024).
         */
        template <typename Self>
        decltype(auto) withWindow( this Self&& self, dim_t window )
        {
            self.window_ = window;
            return std::forward<Self>( self );
        }

        /**
         * @brief Fluent setter for the attention logit scale (QK softmax multiplier).
         *
         * 0 (the default) means "derive 1/sqrt(head_dim)" -- the standard scaling for
         * Llama/Qwen/MHA. A positive value overrides it; Gemma uses 1.0 because its
         * QK-norm already controls the dot-product magnitude (HF self.scaling = 1.0).
         */
        template <typename Self>
        decltype(auto) withAttentionScale( this Self&& self, float scale )
        {
            self.attention_scale_ = scale;
            return std::forward<Self>( self );
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        /**
         * @brief Total Q-projection output width (= num_heads * head_dim).
         */
        dim_t getModelDim() const noexcept
        {
            return model_dim_;
        }

        /**
         * @brief Number of Q attention heads.
         */
        dim_t getNumHeads() const noexcept
        {
            return num_heads_;
        }

        /**
         * @brief Number of K/V attention heads.
         *
         * The KV cache is sized proportionally to this value, giving GQA its
         * memory bandwidth advantage over full MHA.
         */
        dim_t getNumKvHeads() const noexcept
        {
            return num_kv_heads_;
        }

        /**
         * @brief Per-head feature dimension (model_dim / num_heads).
         *
         * Derived quantity -- always consistent with model_dim and num_heads.
         */
        dim_t getHeadDim() const noexcept
        {
            return model_dim_ / num_heads_;
        }

        /**
         * @brief Number of Q heads sharing each K/V head (num_heads / num_kv_heads).
         *
         * Equivalent to the GQA "group size".  A value of 1 recovers standard MHA;
         * a value equal to num_heads recovers Multi-Query Attention.
         */
        dim_t getGroupSize() const noexcept
        {
            return num_heads_ / num_kv_heads_;
        }

        /**
         * @brief Sliding-attention window size (0 = unbounded/global causal).
         */
        dim_t getWindow() const noexcept
        {
            return window_;
        }

        /**
         * @brief Attention logit scale: the explicit value, or 1/sqrt(head_dim) when unset.
         */
        float getAttentionScale() const noexcept
        {
            if ( attention_scale_ > 0.0f )
            {
                return attention_scale_;
            }

            const dim_t head_dim = model_dim_ / num_heads_;

            return (head_dim > 0) ? (1.0f / std::sqrt( static_cast<float>(head_dim) )) : 1.0f;
        }

        // ====================================================================
        // Validation
        // ====================================================================

        /**
         * @brief Validate all configuration parameters.
         *
         * Checks:
         *   1. model_dim > 0
         *   2. num_heads >= 2
         *   3. model_dim % num_heads == 0      (integer head_dim)
         *   4. num_kv_heads >= 1
         *   5. num_kv_heads <= num_heads        (KV heads cannot exceed Q heads)
         *   6. num_heads % num_kv_heads == 0    (integer group size)
         *
         * @throws std::invalid_argument If any constraint is violated.
         */
        void validate() const override
        {
            if ( model_dim_ <= 0 )
            {
                throw std::invalid_argument(
                    "GroupedQueryAttentionConfig: model_dim must be > 0" );
            }

            if ( num_heads_ < 2 )
            {
                throw std::invalid_argument(
                    "GroupedQueryAttentionConfig: num_heads must be >= 2" );
            }

            if ( model_dim_ % num_heads_ != 0 )
            {
                throw std::invalid_argument(
                    "GroupedQueryAttentionConfig: model_dim must be divisible by num_heads" );
            }

            if ( num_kv_heads_ < 1 )
            {
                throw std::invalid_argument(
                    "GroupedQueryAttentionConfig: num_kv_heads must be >= 1" );
            }

            if ( num_kv_heads_ > num_heads_ )
            {
                std::ostringstream oss;
                oss << "GroupedQueryAttentionConfig: num_kv_heads (" << num_kv_heads_
                    << ") must be <= num_heads (" << num_heads_ << ")";
                throw std::invalid_argument( oss.str() );
            }

            if ( num_heads_ % num_kv_heads_ != 0 )
            {
                std::ostringstream oss;
                oss << "GroupedQueryAttentionConfig: num_heads (" << num_heads_
                    << ") must be divisible by num_kv_heads (" << num_kv_heads_ << ")";
                throw std::invalid_argument( oss.str() );
            }

            if ( window_ < 0 )
            {
                throw std::invalid_argument(
                    "GroupedQueryAttentionConfig: window must be >= 0 (0 = global)" );
            }

            if ( attention_scale_ < 0.0f )
            {
                throw std::invalid_argument(
                    "GroupedQueryAttentionConfig: attention_scale must be >= 0 (0 = derive 1/sqrt(head_dim))" );
            }
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Convert configuration to serialization metadata.
         *
         * Produces a SerializationMetadata object containing all configuration
         * fields suitable for writing into an archive by the caller.
         */
        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "model_dim", static_cast<int64_t>(model_dim_) )
                .set( "num_heads", static_cast<int64_t>(num_heads_) )
                .set( "num_kv_heads", static_cast<int64_t>(num_kv_heads_) )
                .set( "window", static_cast<int64_t>(window_) )
                .set( "attention_scale", attention_scale_ );

            return meta;
        }

        /**
         * @brief Populate configuration from serialization metadata.
         *
         * Reads available fields from the provided metadata and updates the
         * configuration object in place.  Missing keys are silently ignored so
         * that older checkpoints without num_kv_heads fall back to the
         * constructor-supplied default.
         */
        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto md = meta.tryGetInt( "model_dim" ) )
            {
                model_dim_ = static_cast<dim_t>(*md);
            }

            if ( auto nh = meta.tryGetInt( "num_heads" ) )
            {
                num_heads_ = static_cast<dim_t>(*nh);
            }

            if ( auto nkv = meta.tryGetInt( "num_kv_heads" ) )
            {
                num_kv_heads_ = static_cast<dim_t>(*nkv);
            }

            if ( auto w = meta.tryGetInt( "window" ) )
            {
                window_ = static_cast<dim_t>(*w);
            }

            if ( auto s = meta.tryGetFloat( "attention_scale" ) )
            {
                attention_scale_ = *s;
            }
        }

        // ====================================================================
        // Diagnostics
        // ====================================================================

        /**
         * @brief Human-readable description of the configuration.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "GQA Config: { ";
            oss << "model_dim=" << model_dim_ << ", ";
            oss << "num_heads=" << num_heads_ << ", ";
            oss << "num_kv_heads=" << num_kv_heads_ << ", ";
            oss << "head_dim=" << getHeadDim() << ", ";
            oss << "group_size=" << getGroupSize() << ", ";
            oss << "window=" << window_ << ", ";
            oss << "attention_scale=" << getAttentionScale() << " }";

            return oss.str();
        }

    private:
        dim_t model_dim_;
        dim_t num_heads_;
        dim_t num_kv_heads_;
        dim_t window_{ 0 };           ///< 0 = unbounded/global causal; > 0 = sliding window
        float attention_scale_{ 0.0f }; ///< 0 = derive 1/sqrt(head_dim); > 0 = explicit (Gemma uses 1.0)
    };
}
