module;
#include <vector>;
#include <cstdint>

export module Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Integer type used for tensor dimensions and indices.
     *
     */
    export using dim_t = int64_t;

    /**
     * @brief Row-major shape vector describing tensor dimensional sizes.
     *
     * Interpretation:
     * - {}    : scalar (rank 0)
     * - {n}   : 1D tensor of length n
     * - {m,n} : 2D tensor with m rows and n columns
     *
     * Elements must be non-negative; a zero in any position indicates an empty tensor.
     */
    export using shape_t = std::vector<dim_t>;

    /**
     * @brief Stride vector (in elements) for each tensor dimension using row-major layout.
     *
     * stride_t[i] gives the number of elements to skip to advance one position along dimension i.
     * Length equals shape.size(); empty for scalars.
     */
    export using stride_t = std::vector<dim_t>;

    /**
     * @brief Index vector used for multi-dimensional element access.
     *
     * Must contain one index per tensor dimension. Valid indices satisfy: 0 <= index[i] < shape[i].
     */
    export using index_t = std::vector<dim_t>;
}