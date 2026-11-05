/**
 * @file OperationTraits.ixx
 * @brief Aggregator for the unified operation traits dispatch table.
 *
 * Import this module to access the primary OperationTraits template and every
 * registered backend specialization.
 */
export module Compute.OperationTraits;

export import Compute.OperationTraits.Template;

// A backend's partition is only compiled when that backend is enabled (MILA_HAS_CUDA
// and MILA_HAS_METAL are PUBLIC compile definitions set by its block in
// Lib/CMakeLists.txt). Guard the re-export so this aggregator -- which lives in
// the always-compiled core module set -- stays valid in a CPU-only build.
#ifdef MILA_HAS_CUDA
export import :Cuda;
#endif

#ifdef MILA_HAS_METAL
export import :Metal;
#endif
