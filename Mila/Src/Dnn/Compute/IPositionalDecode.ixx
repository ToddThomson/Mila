/**
 * @file IPositionalDecode.ixx
 * @brief Interface for position-aware single-token inference decode.
 */

module;

export module Compute.IPositionalDecode;

import Dnn.ITensor;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Interface for operations that require an explicit sequence position during decode.
     *
     * Implemented by operations whose output depends on the token's absolute position
     * in the sequence — primarily the positional embedding lookup (Lpe). Unlike
     * IDecode, which handles position-agnostic single-token ops (e.g. Linear,
     * Attention with KV cache), this interface carries a `position` argument so the
     * backend can select the correct positional embedding row (wpe[position]) rather
     * than the token's index within the local input tensor.
     *
     * Precondition: the implementing operation must have been built and have its
     * parameters bound before decode() is called.
     */
    export struct IPositionalDecode
    {
        /**
         * @brief Single-token decode with explicit sequence position.
         *
         * @param input    Single-token index tensor [B, 1] (INT32).
         * @param output   Pre-allocated output tensor [B, 1, C] to receive the embedding.
         * @param position Zero-based absolute position of the token in the full sequence.
         *                 Used to index into the positional embedding table (wpe[position]).
         */
        virtual void decode( const ITensor& input, ITensor& output, int position ) const = 0;

        virtual ~IPositionalDecode() = default;
    };
}