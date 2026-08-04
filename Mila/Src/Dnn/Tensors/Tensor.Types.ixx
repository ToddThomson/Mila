/**
 * @file Tensor.Types.ixx
 * @brief Core shape, stride, and index types for the Mila tensor API.
 *
 * Defines TensorShape -- a fixed-capacity inline shape descriptor -- and the
 * shape_t, stride_t, and index_t aliases used throughout the framework.
 */

module;
#include <array>
#include <string>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <initializer_list>

export module Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Integer type used for tensor dimensions and indices.
     */
    export using dim_t = int64_t;

    /**
     * @brief Narrow a dim_t to the 32-bit index type used inside device kernels.
     *
     * dim_t is the type of every axis extent, position, and element count at the
     * API, component, and operation-interface layers. Kernels index in 32-bit for
     * register pressure and address-arithmetic cost, so exactly one narrowing point
     * exists per call path: the operation that launches the kernel. Routing that
     * narrowing through this function keeps the membrane greppable and gives the
     * out-of-range case a diagnosable failure instead of a silent wrap.
     *
     * The check is host-side and runs once per launch, not per element -- negligible
     * against launch overhead. The margin is not theoretical: a Gemma 4 12B embedding
     * table is 262144 x 3840 ~ 1.0e9 elements, roughly half of INT_MAX.
     *
     * @throws std::out_of_range if the value does not fit in an int.
     */
    export inline int narrowToKernelIndex( dim_t value )
    {
        if ( value < static_cast<dim_t>(std::numeric_limits<int>::min()) ||
             value > static_cast<dim_t>(std::numeric_limits<int>::max()) )
        {
            throw std::out_of_range(
                "narrowToKernelIndex: value " + std::to_string( value ) +
                " exceeds the 32-bit kernel index range" );
        }

        return static_cast<int>( value );
    }

    /**
     * @brief Fixed-capacity inline shape descriptor for N-dimensional tensors.
     *
     * Stores up to MaxRank dimension sizes inline with no heap allocation. The active
     * rank is tracked by ndim; unused slots are zero-initialized and ignored by all
     * operations. An empty shape (ndim == 0) represents a scalar tensor.
     *
     * MaxRank = 6 covers all current Mila architectures (2-4D) and common research
     * extensions including video and spatiotemporal models (5D) with one dimension
     * of safety margin. shape_t, stride_t, and index_t are all aliases of TensorShape
     * since all three are structurally identical fixed-capacity int64 sequences.
     *
     * @par Example
     * @code
     * shape_t s{ 2, 1024, 768 };  // [B, T, C]
     * s[0] == 2; s.size() == 3;
     * for (auto d : s) { ... }
     * @endcode
     */
    export struct TensorShape
    {
        static constexpr uint8_t MaxRank = 6;

        std::array<dim_t, MaxRank> dims{};
        uint8_t ndim{ 0 };

        TensorShape() = default;

        /**
         * @brief Constructs from a brace-enclosed list of dimension sizes.
         *
         * @param il Dimension sizes in row-major order.
         * @throws std::invalid_argument if il.size() > MaxRank.
         */
        TensorShape( std::initializer_list<dim_t> il )
        {
            if ( il.size() > MaxRank )
            {
                throw std::invalid_argument( "TensorShape: rank exceeds MaxRank (6)" );
            }

            ndim = static_cast<uint8_t>( il.size() );
            std::copy( il.begin(), il.end(), dims.begin() );
        }

        /**
         * @brief Constructs from a contiguous pointer range of dimension sizes.
         *
         * Covers serialization boundaries where shapes arrive as contiguous storage
         * (e.g. std::vector<int64_t>::data()). The range [first, last) must not
         * exceed MaxRank elements.
         *
         * @param first Pointer to the first dimension size.
         * @param last  One-past-end pointer.
         * @throws std::invalid_argument if the range length exceeds MaxRank.
         */
        TensorShape( const dim_t* first, const dim_t* last )
        {
            const auto count = static_cast<size_t>( last - first );

            if ( count > MaxRank )
            {
                throw std::invalid_argument( "TensorShape: rank exceeds MaxRank (6)" );
            }

            ndim = static_cast<uint8_t>( count );
            std::copy( first, last, dims.begin() );
        }

        dim_t  operator[]( size_t i ) const { return dims[i]; }
        dim_t& operator[]( size_t i ) { return dims[i]; }

        const dim_t* begin() const { return dims.data(); }
        const dim_t* end() const { return dims.data() + ndim; }
        dim_t* begin() { return dims.data(); }
        dim_t* end() { return dims.data() + ndim; }

        const dim_t* data() const { return dims.data(); }
        dim_t* data() { return dims.data(); }

        dim_t  back() const { return dims[ndim - 1]; }
        dim_t& back() { return dims[ndim - 1]; }

        size_t size() const { return ndim; }
        bool empty() const { return ndim == 0; }

        /**
         * @brief Equality compares only the active ndim dimensions.
         *
         * Unused slots (indices >= ndim) are ignored regardless of their values.
         */
        bool operator==( const TensorShape& other ) const noexcept
        {
            if ( ndim != other.ndim ) 
                return false;

            for ( uint8_t i = 0; i < ndim; ++i )
            {
                if ( dims[i] != other.dims[i] )
                    return false;
            }

            return true;
        }
    };

    /**
     * @brief Row-major shape descriptor for tensor dimensional sizes.
     *
     * - {}           : scalar (rank 0)
     * - {n}          : 1D tensor of length n
     * - {m, n}       : 2D tensor, m rows and n columns
     * - {B, T, C}    : 3D activation (batch, sequence, channels)
     * - {B, H, T, D} : 4D attention tensor
     *
     * A zero in any position indicates an empty tensor.
     */
    export using shape_t = TensorShape;

    /**
     * @brief Product of a shape's extents -- the number of logical elements it describes.
     *
     * Distinct from TensorShape::size(), which is the rank. A rank-0 shape holds one
     * element, so the empty product of 1 is the correct answer rather than a special case.
     */
    export constexpr dim_t elementCount( const shape_t& shape ) noexcept
    {
        dim_t count = 1;

        for ( auto extent : shape )
        {
            count *= extent;
        }

        return count;
    }

    /**
     * @brief Stride descriptor (in elements) for each tensor dimension, row-major layout.
     *
     * stride_t[i] is the element count to advance one step along dimension i.
     * Length equals shape.size(); empty for scalars.
     */
    export using stride_t = TensorShape;

    /**
     * @brief Index descriptor for multi-dimensional element access.
     *
     * One index per tensor dimension. Valid indices satisfy: 0 <= index[i] < shape[i].
     */
    export using index_t = TensorShape;

    export std::string shapeToString( const shape_t& shape )
    {
        if ( shape.empty() ) return "[]";

        std::string result = "[";

        for ( uint8_t i = 0; i < shape.ndim; ++i )
        {
            if ( i > 0 ) result += ", ";
            result += std::to_string( shape[i] );
        }

        result += "]";

        return result;
    }

    export std::string strideToString( const stride_t& stride )
    {
        return shapeToString( stride );
    }

    export std::string indexToString( const index_t& index )
    {
        return shapeToString( index );
    }
}