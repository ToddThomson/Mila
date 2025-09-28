/**
 * @file TensorOps.ixx
 * @brief Module that re-exports tensor operation partitions and declares the primary TensorOps template.
 *
 * This module collects and re-exports device-agnostic operation partitions (Math, Initializers, ...)
 * and declares the primary `TensorOps<TComputeDeviceTag>` template that device-specific specializations
 * must provide (for example `TensorOps<Compute::CpuComputeDeviceTag>`).
 *
 * The primary template is intentionally left as a declaration here; concrete device implementations
 * live in per-device partitions (e.g., `:Math.Cpu`, `:Initializers.Cpu`).  High-level free-function
 * helpers forward to the appropriate `TensorOps<Tag>::...` implementation based on the tensor's
 * memory resource `ComputeDeviceTag`.
 *
 * Note: Add new operation partitions as submodules and re-export them here so callers can import
 * `Dnn.TensorOps` to access the high-level operation entry points.
 */

module;
#include <type_traits>

export module Dnn.TensorOps;
export import :Math;
export import :Initializers;

namespace Mila::Dnn
{
	export template<typename TComputeDeviceTag> struct TensorOps;
}