/**
 * @file IDecode.ixx
 * @brief Interface for position-agnostic single-token inference decode.
 */

module;

export module Compute.IDecode;

import Dnn.ITensor;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Interface for operations that support an optimized single-token decode path.
     *
     * Implemented by operations whose decode output does not depend on an explicit
     * sequence position — primarily Linear projections and Attention (which manages
     * position internally via the KV cache). Operations that require an absolute
     * sequence position for correct output (e.g. positional embedding lookup)
     * implement IPositionalDecode instead.
     *
     * Precondition: the implementing operation must have been built and have its
     * parameters bound before decode() is called.
     */
    export struct IDecode
    {
        virtual ~IDecode() = default;

        /**
         * @brief Single-token decode pass.
         *
         * Executes an inference-only forward pass optimized for a single token.
         * Attention implementations use this to dispatch to their internal KV cache
         * path; Linear implementations may use a matrix-vector kernel rather than
         * a full matrix-matrix GEMM.
         *
         * @param input  Single-token input tensor [B, 1, features].
         * @param output Pre-allocated output tensor [B, 1, out_features] to receive results.
         */
        virtual void decode( const ITensor& input, ITensor& output ) const = 0;
    };
}