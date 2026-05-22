/**
 * @file OperationTraits.ixx
 * @brief Aggregator for the unified operation traits dispatch table.
 *
 * Import this module to access both the primary OperationTraits template and all
 * registered backend specializations. Partition modules are added here as each
 * component is migrated from its legacy *OpTypeMap to OperationTraits.
 *
 * Migration status:
 *   :Cuda  -- LinearOp complete; SamplingOp pending; policy-free ops pending
 *   :Cpu   -- pending
 */
export module Compute.OperationTraits;

export import Compute.OperationTraits.Template;

export import :Cuda;
// export import :Cpu;  -- pending
