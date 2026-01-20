/**
 * @file Llama.ixx
 * @brief LLaMA network module exports.
 *
 * Provides the public network-level configuration and preset factories for
 * LLaMA-style transformer networks. This module re-exports the `Config`
 * and `Presets` partitions used by network builders and tests.
 */

export module Dnn.Networks.Llama;

/// @defgroup DnnNetworksLlama LLaMA Networks
/// @brief Public network API and presets for LLaMA models.
/// @{

export import :Config;
export import :Presets;

/// @}
