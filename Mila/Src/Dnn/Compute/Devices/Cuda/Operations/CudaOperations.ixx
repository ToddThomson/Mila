/**
 * @file CudaOperations.ixx
 * @brief Aggregated CUDA operation module exports.
 *
 * This module re-exports CUDA-specific operation implementations so callers
 * can import a single module that exposes the available CUDA operation
 * modules and their factory registrations.
 *
 * Exported modules:
 *  - Compute.CudaGeluOp
 *  - Compute.CudaLayerNormOp
 *  - Compute.CudaLinearOp
 *  - Compute.CudaResidualOp
 *  - Compute.CudaSoftmaxOp
 *
 * Notes:
 *  - Each exported module provides device and precision specific operation
 *    implementations and registers factory creators with the OperationRegistry.
 *  - These implementations target NVIDIA CUDA devices and depend on the
 *    project's CUDA kernels and device context abstractions.
 *  - For CPU equivalents see the Compute.CpuOperations module.
 *
 * @since Alpha
 */
export module Compute.CudaOperations;

//export import Compute.CudaEncoderOp;
export import Compute.CudaGeluOp;
export import Compute.CudaLayerNormOp;
export import Compute.CudaLinearOp;
//export import Compute.CudaMatMulBiasGeluOp;
//export import Compute.CudaAttentionOp;
//export import Compute.CudaResidualOp;
export import Compute.CudaSoftmaxOp;
