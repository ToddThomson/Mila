/**
 * @file Tensor.Access.ixx
 * @brief Element access implementation for the Tensor class
 *
 * This module partition implements all element access operations including:
 * - Direct element access via indices (at, operator())
 * - Slicing operations
 * - Any future indexing mechanisms
 */

module;
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

export module Dnn.Tensor:Access;

import :Interface;

import Dnn.TensorTraits;
import Compute.MemoryResource;

//namespace Mila::Dnn
//{
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    TElementType Tensor<TElementType,TMemoryResource>::at( const std::vector<size_t>& indices ) const {
//        validateIndices( indices, "at()" );
//
//        if constexpr ( is_host_accessible<TMemoryResource>() ) {
//            size_t index = computeIndex( indices );
//            return buffer_->data()[ index ];
//        }
//        else {
//            // Get a single element directly instead of copying the entire tensor
//            TElementType result;
//            size_t index = computeIndex( indices );
//
//            cudaError_t status = cudaMemcpy( &result, buffer_->data() + index,
//                sizeof( TElementType ), cudaMemcpyDeviceToHost );
//
//            if ( status != cudaSuccess ) {
//                throw std::runtime_error( "CUDA memory transfer failed in at(): " +
//                    std::string( cudaGetErrorString( status ) ) );
//            }
//
//            return result;
//        }
//    }
//
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    void Tensor<TElementType, TMemoryResource>::set( const std::vector<size_t>& indices, TElementType value ) {
//        validateIndices( indices, "set()" );
//
//        if constexpr ( is_host_accessible<TMemoryResource>() ) {
//            size_t index = computeIndex( indices );
//            buffer_->data()[ index ] = value;
//        }
//        else {
//            // Set a single element directly
//            size_t index = computeIndex( indices );
//
//            cudaError_t status = cudaMemcpy( buffer_->data() + index, &value,
//                sizeof( TElementType ), cudaMemcpyHostToDevice );
//
//            if ( status != cudaSuccess ) {
//                throw std::runtime_error( "CUDA memory transfer failed in set(): " +
//                    std::string( cudaGetErrorString( status ) ) );
//            }
//        }
//    }
//
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    TElementType& Tensor<TElementType, TMemoryResource>::operator[]( const std::vector<size_t>& indices ) {
//        if ( indices.size() != shape_.size() ) {
//            throw std::runtime_error( "operator[]: Number of indices must match the tensor rank." );
//        }
//
//        // Validate indices are within bounds
//        for ( size_t i = 0; i < indices.size(); ++i ) {
//            if ( indices[ i ] >= shape_[ i ] ) {
//                throw std::out_of_range( "operator[]: Index " + std::to_string( indices[ i ] ) +
//                    " is out of range for dimension " + std::to_string( i ) +
//                    " with size " + std::to_string( shape_[ i ] ) );
//            }
//        }
//
//        size_t index = computeIndex( indices );
//
//        if constexpr ( is_host_accessible<TMemoryResource>() ) {
//            return buffer_->data()[ index ];
//        }
//        else {
//            throw std::runtime_error( "Direct tensor access requires host-accessible memory. Use to<CpuMemoryResource>() first." );
//        }
//    }
//
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    const TElementType& Tensor<TElementType, TMemoryResource>::operator[]( const std::vector<size_t>& indices ) const {
//        if ( indices.size() != shape_.size() ) {
//            throw std::runtime_error( "operator[] const: Number of indices must match the tensor rank." );
//        }
//
//        // Validate indices are within bounds
//        for ( size_t i = 0; i < indices.size(); ++i ) {
//            if ( indices[ i ] >= shape_[ i ] ) {
//                throw std::out_of_range( "operator[] const: Index " + std::to_string( indices[ i ] ) +
//                    " is out of range for dimension " + std::to_string( i ) +
//                    " with size " + std::to_string( shape_[ i ] ) );
//            }
//        }
//
//        size_t index = computeIndex( indices );
//
//        if constexpr ( is_host_accessible<TMemoryResource>() ) {
//            return buffer_->data()[ index ];
//        }
//        else {
//            throw std::runtime_error( "Direct tensor access requires host-accessible memory. Use to<CpuMemoryResource>() first." );
//        }
//    }
//
//    /*export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    template<typename... Indices>
//    TElementType& Tensor<TElementType, TMemoryResource>::operator[]( Indices... indices ) {
//        std::vector<size_t> idx{ static_cast<size_t>(indices)... };
//        return this->operator[]( idx );
//    }
//
//    export template<typename TElementType, typename TMemoryResource>
//        requires isValidTensor<TElementType, TMemoryResource>
//    template<typename... Indices>
//    const TElementType& Tensor<TElementType, TMemoryResource>::operator[]( Indices... indices ) const {
//        std::vector<size_t> idx{ static_cast<size_t>(indices)... };
//        return this->operator[]( idx );
//    }*/
//}