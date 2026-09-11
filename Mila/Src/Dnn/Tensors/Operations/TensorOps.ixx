export module Dnn.TensorOps;

export import Dnn.TensorOps.Base;

// The traits modules export no entities -- an explicit specialization cannot carry
// `export` -- so re-exporting them publishes nothing while still satisfying MSVC 14.51,
// which will not complete a merely reachable explicit specialization when the
// dereference is dependent. The backends stay on plain imports, which is what keeps
// ZeroOps, FillOps, MathOps, TransferOps, StructuralOps and RandomOps out of
// `import Mila;`.
export import Dnn.TensorOpsTraits.Cpu;
import Compute.CpuTensorOps;
#ifdef MILA_HAS_CUDA
export import Dnn.TensorOpsTraits.Cuda;
import Compute.CudaTensorOps;
#endif

// REVIEW: Zero, Random and Fill are all related to initialization and should be grouped together under a TensorOps.Init
export import :Zero;
export import :Random;
export import :Fill;

export import :Math;
export import :Transfer;
export import :Structural;

