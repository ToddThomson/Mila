/**
 * @file Rope.RotaryLayout.ixx
 * @brief Which dimension pairs a partial rotary embedding rotates.
 */

module;

export module Dnn.Components.RotaryLayout;

namespace Mila::Dnn
{
    /**
     * @brief Which dimension pairs a partial rotary rotates. Families genuinely disagree.
     *
     * Only meaningful when `rotary_dim` is below `head_dim`; the two coincide at full rotary.
     * Getting it wrong is silent: both layouts rotate the same NUMBER of dimensions and
     * produce a plausible result, and at a short prompt the angles are small enough that
     * generation still looks correct. It diverges with context.
     */
    export enum class RotaryLayout
    {
        /**
         * Rotation spans the whole head, pairing channel `i` with `i + head_dim/2`, and the
         * partial width is expressed by zero frequencies in the cos/sin cache. Gemma:
         * `(x * cos) + (rotate_half(x) * sin)` over the unsliced head.
         */
        WholeHead,

        /**
         * Rotation is confined to the leading `rotary_dim` channels, pairing `i` with
         * `i + rotary_dim/2`; channels at or above `rotary_dim` pass through untouched.
         * Qwen: `q[..., :rotary_dim]` then `rotate_half` inside that slice.
         */
        RotaryPrefix
    };
}
