export module Dnn.TensorOps;

export import Dnn.TensorOps.Base;
export import Compute.CpuTensorOps;
#ifdef MILA_HAS_CUDA
export import Compute.CudaTensorOps;
#endif

// REVIEW: Zero, Random and Fill are all related to initialization and should be grouped together under a TensorOps.Init
export import :Zero;
export import :Random;
export import :Fill;

export import :Math;
export import :Transfer;
export import :Structural;

