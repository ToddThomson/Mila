/**
 * @file IPositionalPairedOp.ixx
 * @brief Interface for paired operations whose output depends on absolute token position.
 */

module;

export module Compute.IPositionalPairedOp;

import Dnn.ITensor;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Capability interface for position-dependent paired operations.
     *
     * Implemented by operations that accept two input/output tensor pairs and
     * whose mathematical output changes based on absolute token position --
     * e.g. RoPE, which rotates both Q and K embeddings by position-dependent
     * angles.
     *
     * Parameter order follows the PairedOperation convention: all inputs first,
     * then all outputs, followed by the position argument.
     *
     * Operations that are position-agnostic do not implement this interface --
     * they use forward() for all modes.
     */
    export struct IPositionalPairedOp
    {
        /**
         * @brief Process a chunk of tokens starting at a given position.
         *
         * @param inputA          First input tensor (e.g. Q) for this chunk.
         * @param inputB          Second input tensor (e.g. K) for this chunk.
         * @param outputA         First output tensor for this chunk.
         * @param outputB         Second output tensor for this chunk.
         * @param position_offset Absolute position of the first token in this chunk.
         */
        virtual void prefill(
            const ITensor& inputA, const ITensor& inputB,
            ITensor& outputA, ITensor& outputB,
            int position_offset ) = 0;

        /**
         * @brief Process a single token at an explicit sequence position.
         *
         * @param inputA   Single-token first input [B, 1, ...].
         * @param inputB   Single-token second input [B, 1, ...].
         * @param outputA  Single-token first output [B, 1, ...].
         * @param outputB  Single-token second output [B, 1, ...].
         * @param position Zero-based absolute sequence position.
         */
        virtual void decode(
            const ITensor& inputA, const ITensor& inputB,
            ITensor& outputA, ITensor& outputB,
            int position ) = 0;

        virtual ~IPositionalPairedOp() = default;
    };
}