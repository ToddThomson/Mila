/**
 * @file CpuOperations.ixx
 * @brief Aggregated CPU operation module exports.
 *
 * This module re-exports CPU-specific operation implementations so callers
 * can import a single module that exposes the available CPU operation
 * modules and their factory registrations.
 *
 * Exported modules:
 *  - Compute.CpuGeluOp
 *  - Compute.CpuLayerNormOp
 *  - Compute.CpuLinearOp
 *  - Compute.CpuResidualOp
 *  - Compute.CpuSoftmaxOp
 *
 * Notes:
 *  - Each exported module provides device- and precision-specific operation
 *    implementations and registers factory creators with the OperationRegistry.
 *  - These implementations target CPU devices and use the project's CPU
 *    device abstractions and memory resources.
 *
 */
export module Compute.CpuOperations;

//export import Compute.CpuAttention;
//export import Compute.CpuCrossEntropyOp;
export import Compute.CpuEncoderOp;
export import Compute.CpuGeluOp;
export import Compute.CpuLayerNormOp;
export import Compute.CpuLinearOp;
export import Compute.CpuResidualOp;
export import Compute.CpuSoftmaxOp;
export import Compute.CpuSoftmaxCrossEntropyOp;