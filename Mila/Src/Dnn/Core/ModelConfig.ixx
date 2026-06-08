/**
 * @file ModelConfig.ixx
 * @brief Base configuration for all deployable Mila models.
 *
 * ModelConfig carries model-wide deployment concerns that are common to all
 * models and independent of network architecture. It is intentionally decoupled
 * from ComponentConfig, which is purely structural (dimensions, features, flags).
 *
 * Deployment concerns owned here:
 *
 *  1. ComputePrecision::Policy  — cuBLASLt algorithm selection heuristic applied
 *                                 uniformly to all compute components (Linear, GQA, MHA).
 *                                 Replaces the precision_ field previously on ComponentConfig.
 *
 *  2. QuantizationConfig        — weight storage dtype and scale allocation policy.
 *                                 Currently consumed by Linear only.
 *                                 Defaults to QuantizationConfig::none().
 *
 *  3. context_length            — maximum sequence length the model is built for.
 *                                 Universal across all sequence models.
 *
 *  4. strict                    — whether unrecognized parameter names throw on load.
 *                                 Universal across all pretrained model loading.
 *
 * Construction is via fluent setters on the concrete subclass.
 * context_length is required and has no default — subclasses must enforce this.
 *
 * ## Relationship to BuildContext
 *
 * ModelConfig is the public API surface for deployment configuration.
 * BuildContext is the internal carrier through the component tree.
 * fromPretrained() projects ModelConfig into BuildContext once — they
 * are never the same object.
 */

module;
#include <string>
#include <cstdint>
#include <stdexcept>

export module Dnn.ModelConfig;

import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Abstract base configuration for all deployable Mila models.
     *
     * Subclasses add architecture-specific deployment concerns
     * (e.g. LlamaModelConfig adds nothing beyond what ModelConfig already owns —
     * all Llama architectural parameters come from checkpoint metadata).
     *
     * Non-copyable by design — model configs are constructed once and passed
     * by const reference into fromPretrained().
     */
    export class ModelConfig
    {
    public:

        virtual ~ModelConfig() = default;

        ModelConfig( const ModelConfig& ) = delete;
        ModelConfig& operator=( const ModelConfig& ) = delete;

        // ====================================================================
        // Fluent setters
        // ====================================================================

        /**
         * @brief Set the weight quantization policy.
         *
         * Determines weight storage dtype and scale allocation for all
         * quantizable components (currently Linear only).
         *
         * @param quantization  Quantization policy to apply.
         * @return              Reference to the concrete config for chaining.
         */
        /*template<typename Self>
        Self& withQuantization( this Self& self, QuantizationConfig quantization )
        {
            self.quantization_ = quantization;
            return self;
        }*/

        /**
         * @brief Set the maximum sequence length.
         *
         * Required. The model is built at this context length so that RoPE
         * embeddings and KV cache buffers cover the full range.
         *
         * @param context_length  Maximum sequence length in tokens.
         * @return                Reference to the concrete config for chaining.
         */
        template<typename Self>
        Self& withContextLength( this Self& self, dim_t context_length )
        {
            if ( context_length == 0 )
            {
                throw std::invalid_argument(
                    "ModelConfig: context_length must be greater than zero" );
            }

            self.context_length_ = context_length;
            return self;
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        /*const QuantizationConfig& getQuantization() const noexcept
        {
            return quantization_;
        }*/

        dim_t getContextLength() const noexcept
        {
            return context_length_;
        }

        // ====================================================================
        // Diagnostics
        // ====================================================================

        /**
         * @brief Produce a human-readable summary of the model configuration.
         *
         * Implementations should include base fields by calling baseToString()
         * and appending subclass-specific fields.
         */
        virtual std::string toString() const = 0;

    protected:

        /**
         * @brief Construct with required context_length.
         *
         * Protected — construction is via concrete subclass only.
         *
         * @param context_length  Maximum sequence length. Must be > 0.
         */
        explicit ModelConfig( dim_t context_length )
            : context_length_( context_length )
        {
            if ( context_length == 0 )
            {
                throw std::invalid_argument(
                    "ModelConfig: context_length must be greater than zero" );
            }
        }

        /**
         * @brief Default constructor for subclasses that set context_length
         *        via withContextLength().
         *
         * context_length_ is initialised to zero. Subclasses or fromPretrained()
         * must call withContextLength() before passing the config to build().
         */
        ModelConfig() = default;

        /**
         * @brief Produce the base fields portion of toString().
         *
         * Subclasses call this and append their own fields.
         */
        std::string baseToString() const
        {
            std::string result;
            result += "  context_length:   " + std::to_string( context_length_ ) + "\n";

            return result;
        }

        // ====================================================================
        // Data members — accessible to subclasses
        // ====================================================================

        dim_t context_length_{ 0 };
    };
}