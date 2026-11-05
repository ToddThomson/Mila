/**
 * @file MetalTensorOps.ixx
 * @brief The tensor operations of the Metal backend.
 */

export module Compute.MetalTensorOps;

export import :Fill;
export import :Random;
export import :Scale;
export import :Structural;
export import :Transfer;
export import :Zero;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute::Metal;

    export struct MetalTensorOps : FillOps, RandomOps, ScaleOps, StructuralOps, TransferOps, ZeroOps {};
}
