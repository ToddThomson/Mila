module;
#include <vector>;
#include <stdexcept>;
#include <string>;
#include <cstdint>;
#include <utility>

export module Dnn.TensorPartitioning;

import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Information about axis partitioning of a tensor.
     */
    export struct AxisPartition
    {
        int64_t normalized_axis;  ///< Axis normalized to [0, ndim)
        int64_t outer_size;       ///< Product of dimensions before axis
        int64_t axis_size;        ///< Size of the axis dimension
        int64_t inner_size;       ///< Product of dimensions after axis
        int64_t num_slices;       ///< outer_size * inner_size (for statistics buffers)
    };

    /**
     * @brief Normalize and validate an axis, then compute partition sizes.
     *
     * @param shape Tensor shape
     * @param axis Axis to normalize (supports negative indexing)
     * @param op_name Operation name for error messages
     * @return AxisPartition Structure containing normalized axis and sizes
     * @throws std::runtime_error If axis is out of range
     */
    export AxisPartition computeAxisPartition(
        const shape_t& shape,
        dim_t axis,
        const char* op_name = "Operation"
    )
    {
        const int64_t ndim = static_cast<int64_t>(shape.size());

        if (ndim == 0)
        {
            throw std::runtime_error(
                std::string( op_name ) + " - cannot operate on scalar (0-dimensional) tensor"
            );
        }

        // Normalize negative axis
        int64_t normalized_axis = axis;
        if (normalized_axis < 0)
        {
            normalized_axis += ndim;
        }

        // Validate range
        if (normalized_axis < 0 || normalized_axis >= ndim)
        {
            throw std::runtime_error(
                std::string( op_name ) + " - axis " + std::to_string( axis ) +
                " out of range for tensor with " + std::to_string( ndim ) + " dimensions"
            );
        }

        // Compute outer size (product of all dimensions before axis)
        int64_t outer_size = 1;
        for (int64_t i = 0; i < normalized_axis; ++i)
        {
            outer_size *= shape[i];
        }

        // Get axis dimension size
        const int64_t axis_size = shape[normalized_axis];

        // Compute inner size (product of all dimensions after axis)
        int64_t inner_size = 1;
        for (int64_t i = normalized_axis + 1; i < ndim; ++i)
        {
            inner_size *= shape[i];
        }

        return AxisPartition{
            .normalized_axis = normalized_axis,
            .outer_size = outer_size,
            .axis_size = axis_size,
            .inner_size = inner_size,
            .num_slices = outer_size * inner_size
        };
    }

    /**
     * @brief Compute total number of elements in a tensor shape.
     *
     * @param shape Tensor shape
     * @return int64_t Total number of elements
     */
    export int64_t computeNumElements( const std::vector<int64_t>& shape )
    {
        int64_t num_elements = 1;
        for (auto dim : shape)
        {
            num_elements *= dim;
        }

        return num_elements;
    }

    /**
     * @brief Validate that a tensor has the expected number of elements.
     *
     * @param shape Tensor shape to validate
     * @param expected_size Expected number of elements
     * @param tensor_name Name of tensor for error message
     * @param op_name Operation name for error message
     * @throws std::runtime_error If size doesn't match
     */
    export void validateTensorSize(
        const std::vector<int64_t>& shape,
        int64_t expected_size,
        const char* tensor_name = "tensor",
        const char* op_name = "Operation"
    )
    {
        int64_t actual_size = computeNumElements( shape );
        if (actual_size != expected_size)
        {
            throw std::runtime_error(
                std::string( op_name ) + " - " + tensor_name +
                " has unexpected size. Expected " + std::to_string( expected_size ) +
                " elements, got " + std::to_string( actual_size )
            );
        }
    }

    /**
     * @brief Multi-axis partition (for operations like LayerNorm over multiple trailing dims).
     */
    export struct MultiAxisPartition
    {
        int64_t outer_size;          ///< Product of dimensions before normalized axes
        int64_t normalized_size;     ///< Product of the normalized dimensions
        std::vector<int64_t> outer_shape;      ///< Shape of outer dimensions
        std::vector<int64_t> normalized_shape; ///< Shape of normalized dimensions
    };

    /**
     * @brief Compute partition for normalization over trailing dimensions.
     *
     * @param shape Input tensor shape
     * @param normalized_shape Expected trailing dimensions to normalize over
     * @param op_name Operation name for error messages
     * @return MultiAxisPartition Structure containing partition information
     * @throws std::runtime_error If normalized_shape doesn't match trailing dims
     */
    export MultiAxisPartition computeNormalizedShapePartition(
        const std::vector<int64_t>& shape,
        const std::vector<int64_t>& normalized_shape,
        const char* op_name = "Operation"
    )
    {
        const int64_t ndim = static_cast<int64_t>(shape.size());
        const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());

        if (norm_ndim > ndim)
        {
            throw std::runtime_error(
                std::string( op_name ) + " - normalized_shape has more dimensions (" +
                std::to_string( norm_ndim ) + ") than input tensor (" +
                std::to_string( ndim ) + ")"
            );
        }

        // Verify trailing dimensions match
        const int64_t offset = ndim - norm_ndim;
        for (int64_t i = 0; i < norm_ndim; ++i)
        {
            if (shape[offset + i] != normalized_shape[i])
            {
                throw std::runtime_error(
                    std::string( op_name ) + " - input shape trailing dimensions don't match normalized_shape. "
                    "Expected dimension " + std::to_string( i ) + " to be " +
                    std::to_string( normalized_shape[i] ) + ", got " +
                    std::to_string( shape[offset + i] )
                );
            }
        }

        // Compute outer size and shape
        int64_t outer_size = 1;
        std::vector<int64_t> outer_shape;
        for (int64_t i = 0; i < offset; ++i)
        {
            outer_size *= shape[i];
            outer_shape.push_back( shape[i] );
        }

        // Compute normalized size
        int64_t normalized_size = 1;
        for (auto dim : normalized_shape)
        {
            normalized_size *= dim;
        }

        return MultiAxisPartition{
            .outer_size = outer_size,
            .normalized_size = normalized_size,
            .outer_shape = std::move( outer_shape ),
            .normalized_shape = normalized_shape
        };
    }
}