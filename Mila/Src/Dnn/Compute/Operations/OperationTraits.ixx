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

export import :Cuda;
export import :Cpu;
