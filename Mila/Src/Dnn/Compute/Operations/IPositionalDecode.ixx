/**
 * @file IPositionalUnaryOp.ixx
 * @brief Interface for unary operations whose output depends on absolute token position.
 */

module;

export module Compute.IPositionalDecode;

import Dnn.ITensor;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Capability interface for position-dependent unary operations.
     *
     * Implemented by operations whose mathematical output changes based on
     * the token's absolute position in the sequence — e.g. positional embedding
     * lookups (Lpe) and attention operations that index into a KV cache (GQA, MHA).
     *
     * Operations that are position-agnostic (Linear, RmsNorm, SwiGLU, Residual)
     * do not implement this interface — they use forward() for all modes.
     */
    export struct IPositionalDecode
    {
        /**
         * @brief Process a single token at an explicit sequence position.
         *
         * @param input    Single-token input [B, 1, ...].
         * @param output   Single-token output [B, 1, ...].
         * @param position Zero-based absolute sequence position.
         */
        virtual void decode( const ITensor& input, ITensor& output, int position ) = 0;

        virtual ~IPositionalDecode() = default;
    };
}