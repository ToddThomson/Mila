module;
#include <cuda_fp16.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

export module Dnn.Tensor:Io;

import :Interface;

import Dnn.TensorTraits;
import Compute.MemoryResource;

//namespace Mila::Dnn
//{
//    // ----------------------------------------------------------------
//    // Public Tensor I/O operations
//    // ----------------------------------------------------------------
//
//    /**
//    * @brief Converts the tensor to a string representation
//    *
//    * @param showBuffer Whether to include the tensor's contents in the output
//    * @return std::string String representation of the tensor
//    */
//    export template <typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType,TMemoryResource>
//    std::string Tensor<TElementType, TMemoryResource>::toString( bool showBuffer ) const {
//        std::ostringstream oss;
//        oss << "Tensor: " << uid_;
//        if ( !name_.empty() )
//            oss << "::" << name_;
//        oss << ", ";
//        oss << outputLayout();
//
//        oss << " Type: " << TensorTrait<TElementType>::type_name << std::endl;
//
//        if ( showBuffer ) {
//            oss << getBufferString( 0, 0 );
//        }
//
//        return oss.str();
//    }
//
//    /**
//     * @brief Gets a string representation of the tensor buffer
//     *
//     * @param start_index Starting index in the buffer
//     * @param depth Current dimension depth
//     * @return std::string String representation of the tensor buffer
//     */
//    export template <typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    std::string Tensor<TElementType, TMemoryResource>::getBufferString( size_t start_index, size_t depth ) const {
//        if constexpr ( is_host_accessible<TMemoryResource>() ) {
//            return outputBuffer( start_index, depth );
//        }
//        else {
//            // FIXME:
//            //auto host_tensor = to<Compute::HostMemoryResource>();
//            //return host_tensor.getBufferString( start_index, depth );
//            return "Tensor is not host-accessible. Cannot output buffer contents.";
//        }
//    }
//
//    /**
//     * @brief Stream insertion operator for tensor output
//     *
//     * @param os Output stream
//     * @param tensor Tensor to output
//     * @return std::ostream& Reference to the output stream
//     */
//    export template <typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    std::ostream& operator<<( std::ostream& os, const Tensor<TElementType, TMemoryResource>& tensor ) {
//        os << tensor.toString();
//        return os;
//    }
//
//    // ----------------------------------------------------------------
//    // Private methods for internal use
//    // ----------------------------------------------------------------
//
//    /**
//     * @brief Formats the tensor buffer for output
//     *
//     * @param index Current index in the buffer
//     * @param depth Current dimension depth
//     * @return std::string Formatted string representation
//     */
//    export template <typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    std::string Tensor<TElementType, TMemoryResource>::outputBuffer( size_t index, size_t depth ) const {
//        std::ostringstream oss;
//        if ( depth == shape_.size() - 1 ) {
//            for ( size_t i = 0; i < shape_[ depth ]; ++i ) {
//                if ( i < 3 || i >= shape_[ depth ] - 3 ) {
//                    TElementType value = buffer_->data()[ index + i ];
//
//                    if constexpr ( std::is_same_v<TElementType, half> ) {
//                        oss << std::setw( 10 ) << static_cast<float>( __half2float( value ) ) << " ";
//                    }
//                    else {
//                        oss << std::setw( 10 ) << value << " ";
//                    }
//                }
//                else if ( i == 3 ) {
//                    oss << "... ";
//                }
//            }
//        }
//        else {
//            for ( size_t i = 0; i < shape_[ depth ]; ++i ) {
//                if ( i < 3 || i >= shape_[ depth ] - 3 ) {
//                    oss << "[ ";
//                    oss << outputBuffer( index + i * strides_[ depth ], depth + 1 );
//                    oss << "]" << std::endl;
//                }
//                else if ( i == 3 ) {
//                    oss << "[ ... ]" << std::endl;
//                    i = shape_[ depth ] - 4;
//                }
//            }
//        }
//        return oss.str();
//    }
//}