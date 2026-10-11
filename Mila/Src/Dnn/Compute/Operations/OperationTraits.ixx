/**
 * @file OperationTraits.ixx
 * @brief Aggregator for the unified operation traits dispatch table.
 *
 * Import this module to access the primary OperationTraits template and every
 * registered backend specialization. CUDA is the only compute backend.
 */
export module Compute.OperationTraits;

export import Compute.OperationTraits.Template;

// The :Cuda partition is only compiled when the CUDA backend is enabled
// (MILA_HAS_CUDA is a PUBLIC compile definition set by the CUDA block in
// Mila/CMakeLists.txt). Guard the re-export so this aggregator -- which lives in
// the always-compiled core module set -- stays valid in a CPU-only build.
#ifdef MILA_HAS_CUDA
export import :Cuda;
#endif
