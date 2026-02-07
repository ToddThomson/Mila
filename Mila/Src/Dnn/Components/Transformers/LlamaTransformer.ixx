/**
 * @file Llama.ixx
 * @brief LLaMA Network component.
 *
 * Provides the public network-level configuration and preset factories for
 * LLaMA-style transformer networks. This module re-exports the `Config`
 * and `Presets` partitions used by network builders and tests.
 */

export module Dnn.Components.LlamaTransformer;

export import :Config;
export import :Presets;

