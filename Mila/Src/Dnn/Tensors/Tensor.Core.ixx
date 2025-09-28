/**
 * @file Tensor.Base.ixx
 * @brief Core tensor definition module partition
 *
 * This module partition defines the foundational structure of the Tensor class including:
 * - Class template declaration with template constraints
 * - Primary member fields for tensor state management
 * - Constructor/destructor implementation
 * - Basic type functionality (copy/move semantics)
 * - Core interface implementation from ITensorData
 *
 * This partition serves as the backbone for all other specialized tensor functionality
 * partitions (Memory, Layout, ID, IO) and contains only the essential components
 * needed for tensor instantiation and identity.
 */

module;
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>

export module Dnn.Tensor:Core;

import :Interface;

import Dnn.TensorBuffer;
import Dnn.TensorData;
import Dnn.TensorPtr;
import Dnn.TensorTraits;
import Compute.MemoryResource;
import Compute.CudaMemoryResource;

//namespace Mila::Dnn
//{
//    /**
//     * @brief Constructs a tensor with the given shape and initializes it with the specified value.
//     *
//     * This constructor initializes the tensor with the provided shape and fills it with the given value.
//     * If no value is provided, the tensor is initialized with the default value of the type TElementType.
//     *
//     * @param shape The shape of the tensor.
//     * @param value The value to initialize the tensor with. Defaults to the default value of type TElementType.
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    Tensor<TElementType, TMemoryResource>::Tensor( const std::vector<size_t>& shape, TElementType value )
//        : uid_( setUId() ), shape_( shape ), strides_( computeStrides( shape ) ), size_( computeSize( shape ) ) {
//        allocateBuffer( value );
//    }
//
//    /**
//     * @brief Constructs a tensor with the given shape and a shared pointer to allocated memory.
//     *
//     * This constructor initializes the tensor with the provided shape and uses the given shared pointer to allocated memory.
//     * The tensor does not take ownership of the memory.
//     *
//     * @param shape The shape of the tensor.
//     * @param data_ptr Shared pointer to the allocated memory.
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    Tensor<TElementType, TMemoryResource>::Tensor( const std::vector<size_t>& shape, std::shared_ptr<TElementType> data_ptr )
//        : uid_( setUId() ), shape_( shape ), strides_( computeStrides( shape ) ), size_( computeSize( shape ) ), external_memory_ptr_( data_ptr ) {
//        if ( !external_memory_ptr_ ) {
//            throw std::invalid_argument( "data_ptr cannot be null." );
//        }
//
//        buffer_ = std::make_shared<TensorBuffer<TElementType, TMemoryResource>>( size_, external_memory_ptr_.get() );
//    }
//
//    /**
//     * @brief Default constructor that creates an empty tensor.
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    Tensor<TElementType, TMemoryResource>::Tensor()
//        : uid_( setUId() ), shape_(), strides_( computeStrides( shape_ ) ), size_( 0 ) {
//        allocateBuffer( TElementType{} );
//    }
//
//    /**
//     * @brief Copy constructor (creates a shallow copy).
//     *
//     * This constructor creates a new tensor that shares the underlying data buffer with
//     * the original tensor. Modifications to one tensor's data will affect the other.
//     * For a deep, independent copy, use the clone() method instead.
//     *
//     * @param other The tensor to copy from.
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    Tensor<TElementType, TMemoryResource>::Tensor( const Tensor& other )
//        : uid_( other.uid_ ), name_( other.name_ ), shape_( other.shape_ ), strides_( other.strides_ ),
//        size_( other.size_ ), buffer_( other.buffer_ ) {}
//
//    /**
//     * @brief Move constructor.
//     *
//     * This constructor moves the contents of the given tensor to this tensor.
//     *
//     * @param other The tensor to move from.
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    Tensor<TElementType, TMemoryResource>::Tensor( Tensor&& other ) noexcept
//        : uid_( std::move( other.uid_ ) ),
//        name_( std::move( other.name_ ) ),
//        size_( other.size_ ),
//        shape_( std::move( other.shape_ ) ),
//        strides_( std::move( other.strides_ ) ),
//        buffer_( std::move( other.buffer_ ) ) {
//        other.size_ = 0;
//    }
//
//    /**
//     * @brief Move assignment operator.
//     *
//     * This operator moves the contents of the given tensor to this tensor.
//     *
//     * @param other The tensor to move from.
//     * @return Tensor& A reference to this tensor.
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    Tensor<TElementType, TMemoryResource>& Tensor<TElementType, TMemoryResource>::operator=( Tensor<TElementType, TMemoryResource>&& other ) noexcept {
//        if ( this != &other ) {
//            uid_ = std::move( other.uid_ );
//            name_ = std::move( other.name_ );
//            shape_ = std::move( other.shape_ );
//            strides_ = std::move( other.strides_ );
//            size_ = other.size_;
//            buffer_ = std::move( other.buffer_ );
//
//            other.size_ = 0;
//        }
//        return *this;
//    }
//
//    /**
//     * @brief Copy assignment operator.
//     *
//     * This operator copies the contents of the given tensor to this tensor.
//     *
//     * @param other The tensor to copy from.
//     * @return Tensor& A reference to this tensor.
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    Tensor<TElementType, TMemoryResource>& Tensor<TElementType, TMemoryResource>::operator=( const Tensor<TElementType, TMemoryResource>& other ) {
//        if ( this != &other ) {
//            uid_ = other.uid_;
//            name_ = other.name_;
//            shape_ = other.shape_;
//            strides_ = other.strides_;
//            size_ = other.size_;
//            buffer_ = other.buffer_;
//        }
//        return *this;
//    }
//
//    /**
//     * @brief Get the tensor's data type
//     * @return TensorDataType enumeration representing the element type
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    TensorDataType Tensor<TElementType, TMemoryResource>::getDataType() const {
//        return TensorTrait<TElementType>::data_type;
//    }
//
//    /**
//     * @brief Get string representation of the tensor's data type
//     * @return String name of the tensor's data type
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    std::string Tensor<TElementType, TMemoryResource>::getDataTypeName() const {
//        return std::string( TensorTrait<TElementType>::type_name );
//    }
//
//    /**
//     * @brief Allocates the tensor buffer with a specified initial value
//     * @param value Initial value for all elements
//     */
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    void Tensor<TElementType, TMemoryResource>::allocateBuffer( TElementType value ) {
//        buffer_ = std::make_shared<TensorBuffer<TElementType, TMemoryResource>>( size_, value );
//    }
//
//}


