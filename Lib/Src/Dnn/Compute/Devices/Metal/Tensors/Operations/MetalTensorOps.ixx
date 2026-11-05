/**
 * @file MetalTensorOps.ixx
 * @brief The tensor operations of the Metal backend.
 */

export module Compute.MetalTensorOps;

export import :Fill;
export import :Transfer;
export import :Zero;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute::Metal;

    export struct MetalTensorOps : FillOps, TransferOps, ZeroOps {};
}
