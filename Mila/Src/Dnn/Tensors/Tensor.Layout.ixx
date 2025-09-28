module;
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

export module Dnn.Tensor:Layout;

import :Interface;

import Dnn.TensorTraits;
import Dnn.TensorData;
import Compute.MemoryResource;

//namespace Mila::Dnn
//{
//    /**
//    * @brief Get the tensor shape (dimensions)
//    *
//    * Returns a reference to the tensor's shape vector, which contains
//    * the size of each dimension. The number of elements in the vector
//    * equals the tensor's rank.
//    *
//    * @return Constant reference to the shape vector
//    */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    const std::vector<size_t>& Tensor<TElementType, TMemoryResource>::shape() const {
//        return shape_;
//    }
//
//    /**
//     * @brief Get the strides for each dimension
//     *
//     * Returns a reference to the tensor's stride vector, which contains
//     * the memory step size for each dimension. Strides are used to calculate
//     * the memory offset when accessing elements with multi-dimensional indices.
//     *
//     * For row-major format (default):
//     * - The last dimension has stride 1
//     * - Each preceding dimension's stride is the product of all following dimensions' sizes
//     *
//     * @return Constant reference to the strides vector
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    const std::vector<size_t>& Tensor<TElementType, TMemoryResource>::strides() const {
//        return strides_;
//    }
//
//    /**
//     * @brief Returns the total number of elements in the tensor.
//     *
//     * This method provides the total count of elements across all dimensions,
//     * which equals the product of all dimension sizes in the shape vector.
//     * For an empty tensor, this returns 0.
//     *
//     * @return The total number of elements in the tensor
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    size_t Tensor<TElementType, TMemoryResource>::size() const {
//        return size_;
//    }
//
//    /**
//     * @brief Returns the number of dimensions (rank) of the tensor.
//     *
//     * The rank represents the number of axes in the tensor, which equals
//     * the number of elements in the shape vector. For example:
//     * - A scalar has rank 0
//     * - A vector has rank 1
//     * - A matrix has rank 2
//     * - An n-dimensional tensor has rank n
//     *
//     * @return The number of dimensions in the tensor
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    size_t Tensor<TElementType, TMemoryResource>::rank() const {
//        return shape_.size();
//    }
//
//    /**
//     * @brief Checks if the tensor is empty (contains no elements).
//     *
//     * @return true if the tensor has no elements, false otherwise
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    const bool Tensor<TElementType, TMemoryResource>::empty() const {
//        return (size_ == 0);
//    }
//
//    /**
//     * @brief Reshapes the tensor to new dimensions while preserving the total number of elements.
//     *
//     * Changes the shape of the tensor without changing the underlying data.
//     * The new shape must have the same total number of elements as the original shape.
//     * If the tensor is empty, the buffer will be resized to match the new shape.
//     *
//     * @param new_shape The new dimensions for the tensor
//     * @throws std::runtime_error if the new shape doesn't match the total element count
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    void Tensor<TElementType, TMemoryResource>::reshape( const std::vector<size_t>& new_shape ) {
//        size_t new_size = computeSize( new_shape );
//        if ( !empty() && (new_size != size_) ) {
//            throw std::runtime_error( "The new shape must match the size of the tensor or the tensor must be empty." );
//        }
//
//        shape_ = new_shape;
//        strides_ = computeStrides( new_shape );
//
//        if ( empty() ) {
//            buffer_->resize( new_size );
//            size_ = new_size;
//        }
//    }
//
//    /**
//     * @brief Flattens the tensor to a 2D tensor by combining all dimensions except the last one.
//     *
//     * This method reshapes the tensor from [D1, D2, ..., Dn-1, Dn] to [D1*D2*...*Dn-1, Dn].
//     * This is particularly useful for operations that treat all but the last dimension as batch dimensions,
//     * such as Fully Connected operations.
//     *
//     * @return A reference to this tensor after flattening.
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    Tensor<TElementType, TMemoryResource>& Tensor<TElementType, TMemoryResource>::flatten() {
//        if ( rank() <= 1 ) {
//            return *this;
//        }
//
//        // Calculate the product of all dimensions except the last
//        size_t flat_dim = 1;
//        for ( size_t i = 0; i < shape().size() - 1; i++ ) {
//            flat_dim *= shape()[ i ];
//        }
//
//        // The new shape is [flat_dim, last_dim]
//        std::vector<size_t> new_shape = { flat_dim, shape().back() };
//        reshape( new_shape );
//
//        return *this;
//    }
//
//    /**
//     * @brief Creates a flattened copy of this tensor.
//     *
//     * Similar to flatten(), but returns a new tensor instead of modifying this one.
//     *
//     * @return A new tensor with flattened shape.
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    Tensor<TElementType, TMemoryResource> Tensor<TElementType, TMemoryResource>::flattened() const {
//        if (rank() <= 1) {
//            return *this;
//        }   
//    
//        Tensor<TElementType, TMemoryResource> result = *this;
//        result.flatten();
//    
//        return result;
//    }
//
//    /**
//     * @brief Create string representation of the layout
//     * @return String describing the layout
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    std::string Tensor<TElementType, TMemoryResource>::outputLayout() const {
//        std::string result = "ITensorData(shape=[";
//
//        for ( size_t i = 0; i < shape_.size(); ++i ) {
//            result += std::to_string( shape_[ i ] );
//            if ( i < shape_.size() - 1 ) result += ",";
//        }
//
//        result += "], strides=[";
//
//        for ( size_t i = 0; i < strides_.size(); ++i ) {
//            result += std::to_string( strides_[ i ] );
//            if ( i < strides_.size() - 1 ) result += ",";
//        }
//
//        result += "], format=RowMajor";
//        result += ", size=" + std::to_string( size_ ) + ")";
//
//        return result;
//    }
//
//    /**
//     * @brief Calculates strides for the provided shape.
//     *
//     * Computes the memory stride for each dimension in row-major format.
//     *
//     * @param shape The tensor shape
//     * @return Vector of strides corresponding to each dimension
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    std::vector<size_t> Tensor<TElementType, TMemoryResource>::computeStrides( const std::vector<size_t>& shape ) {
//        std::vector<size_t> strides( shape.size(), 1 );
//
//        if ( shape.empty() ) {
//            return strides;
//        }
//        
//        // REVIEW:
//
//        /*for ( size_t i = shape.size() - 2; i >= 0; --i ) {
//            strides[ i ] = strides[ i + 1 ] * shape[ i + 1 ];
//        }*/
//
//        for ( size_t i = shape.size() - 1; i > 0; --i ) {
//            strides[ i - 1 ] = strides[ i ] * shape[ i ];
//        }
//
//        return strides;
//    }
//
//    /**
//     * @brief Computes the total number of elements from a shape vector.
//     *
//     * @param shape The tensor shape
//     * @return Total number of elements (product of all dimensions)
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    size_t Tensor<TElementType, TMemoryResource>::computeSize( const std::vector<size_t>& shape ) {
//        if ( shape.empty() ) {
//            return 0;
//        }
//        return std::accumulate( shape.begin(), shape.end(), 1ull, std::multiplies<size_t>() );
//    }
//
//    /**
//     * @brief Computes the linear memory index from multi-dimensional indices.
//     *
//     * @param indices Multi-dimensional indices
//     * @return Linearized memory index
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    size_t Tensor<TElementType, TMemoryResource>::computeIndex( const std::vector<size_t>& indices ) const {
//        size_t index = 0;
//        for ( size_t i = 0; i < indices.size(); ++i ) {
//            index += indices[ i ] * strides_[ i ];
//        }
//        return index;
//    }
//
//    /**
//     * @brief Validates that indices are within the bounds of the tensor shape.
//     *
//     * @param indices Multi-dimensional indices to validate
//     * @param method_name Name of the calling method for error reporting
//     * @throws std::runtime_error if indices count doesn't match tensor rank
//     * @throws std::out_of_range if an index is out of bounds
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    void Tensor<TElementType, TMemoryResource>::validateIndices( const std::vector<size_t>& indices, const std::string& method_name ) const {
//        if ( indices.size() != shape_.size() ) {
//            throw std::runtime_error( method_name + ": Number of indices must match the tensor rank." );
//        }
//
//        for ( size_t i = 0; i < indices.size(); ++i ) {
//            if ( indices[ i ] >= shape_[ i ] ) {
//                throw std::out_of_range( method_name + ": Index " + std::to_string( indices[ i ] ) +
//                    " is out of range for dimension " + std::to_string( i ) +
//                    " with size " + std::to_string( shape_[ i ] ) );
//            }
//        }
//    }
//}