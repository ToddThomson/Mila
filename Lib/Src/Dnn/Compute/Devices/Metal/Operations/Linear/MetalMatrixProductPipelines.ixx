/**
 * @file MetalMatrixProductPipelines.ixx
 * @brief The kernels a matrix product on Metal may dispatch, resolved once by the operation that encodes it.
 */

export module Compute.MetalMatrixProductPipelines;

import Compute.MetalKernelLibrary;

namespace Mila::Dnn::Compute::Metal
{
    /// One precision's kernels, from `matrixProductPipelines` in Compute.MetalMatrixProduct.
    export struct MatrixProductPipelines
    {
        /// TensorOps over tiles of 128 rows, and of 8.
        MetalPipeline large_tiles{ nullptr };
        MetalPipeline small_tiles{ nullptr };

        /// The own kernel, for inputs a multiple of 4 and for any others.
        MetalPipeline lanes{ nullptr };
        MetalPipeline elements{ nullptr };
    };
}
