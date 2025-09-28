/**
 * @file CudaTensorOps.ixx
 * @brief Primary CUDA tensor operations module
 *
 * This module provides the unified interface for all CUDA tensor operations,
 * organizing functionality into logical partitions for clean separation of concerns.
 * Serves as the primary module interface that re-exports all tensor operation
 * partitions for a cohesive API.
 *
 * Architecture:
 * - Transfer: Memory transfer operations (host?device, type conversion)
 * - Fill: Tensor initialization and fill operations (future)
 * - Transform: Shape and layout transformation operations (future)
 * - Math: Mathematical operations and reductions (future)
 */

export module Compute.CudaTensorOps;

//export import :Fill;        // Tensor initialization and fill operations
//export import :Math;        // Arithmetic operations, reductions, activations