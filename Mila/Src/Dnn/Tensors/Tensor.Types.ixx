module;
#include <vector>
#include <string>
#include <cstdint>

export module Dnn.TensorTypes;

namespace Mila::Dnn
{
    // REVIEW: Consider strong spacial dimension typing

    //template <typename Tag>
    //struct Dim
    //{
    //    dim_t value;
    //};
    //
    //struct EmbeddingTag;
    //struct ModelTag;
    //struct VocabTag;
    //
    //using EmbeddingDim = Dim<EmbeddingTag>;
    //using ModelDim = Dim<ModelTag>;
    //using VocabularySize = Dim<VocabTag>;

    // REVIEW: Consider adding a compile-time dimension type for model dimension to enable static checks and optimizations.
    // This would require defining a template parameter for dimension types and enforcing that they are positive integers. For example:
    // 
    //template<typename T>
    //concept PositiveDim = std::integral<T>;

    //template <PositiveDim D>
    //auto withModelDimension( D d )
    //{
    //    assert( d > 0 );
    //    model_dim_ = static_cast<dim_t>(d);
    //    return *this;
    //}

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

    /**
     * @brief Convert a vector of dim_t to string representation
     *
     * Generic helper used by shape/stride/index formatters
     */
    std::string vectorToString( const std::vector<dim_t>& vec )
    {
        if ( vec.empty() ) {
            return "[]";
        }

        std::string result = "[";
        
        for ( size_t i = 0; i < vec.size(); ++i ) {
            if ( i > 0 ) result += ", ";
            result += std::to_string( vec[ i ] );
        }
        
        result += "]";
    
        return result;
    }

    export std::string shapeToString( const shape_t& shape )
    {
        return vectorToString( shape );
    }

    export std::string strideToString( const stride_t& stride )
    {
        return vectorToString( stride );
    }

    export std::string indexToString( const index_t& index )
    {
        return vectorToString( index );
    }
}