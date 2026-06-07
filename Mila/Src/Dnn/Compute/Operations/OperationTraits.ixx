/**
 * @file OperationTraits.ixx
 * @brief Aggregator for the unified operation traits dispatch table.
 *
 * Import this module to access both the primary OperationTraits template and all
 * registered backend specializations. Partition modules are added here as each
 * component is migrated from its legacy *OpTypeMap to OperationTraits.
 *
 * Migration status:
 *   :Cuda  -- LinearOp complete; policy-free ops complete; SamplingOp pending
 *   :Cpu   -- policy-free ops complete; LinearOp pending; SamplingOp pending
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
export import :Cpu;
