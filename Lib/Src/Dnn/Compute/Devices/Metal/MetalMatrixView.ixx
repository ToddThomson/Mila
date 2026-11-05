/**
 * @file MetalMatrixView.ixx
 * @brief A matrix inside a Metal allocation, as a kernel's TensorOps argument sees it.
 */

export module Compute.MetalMatrixView;

import Dnn.TensorDataType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Densely stored elements in `rows` rows of `columns`, at an address inside a live Metal allocation.
     *
     * `row_stride` is the distance between rows, in elements. It equals `columns` for a whole matrix.
     */
    export struct MetalMatrixView
    {
        const void* address{ nullptr };
        TensorDataType data_type{ TensorDataType::FP16 };
        int rows{ 0 };
        int columns{ 0 };
        int row_stride{ 0 };
    };
}
