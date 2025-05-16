/**
 * @file TensorType.ixx
 * @brief Defines tensor data types and related utility functions.
 */

module;
#include <cstdint>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdexcept>

export module Dnn.TensorType;

namespace Mila::Dnn
{
    /**
     * @brief Enumeration of supported tensor data types.
     *
     * This enumeration defines the different numeric data types that can be
     * used for tensor elements in the neural network framework. It supports
     * various precision floating point and integer formats.
     */
    /*export enum class TensorType {
        FP32,   ///< 32-bit floating point (float)
        FP16,   ///< 16-bit floating point (half)
        BF16,   ///< 16-bit brain floating point (nv_bfloat16)
        FP8,    ///< 8-bit floating point (__nv_fp8_e4m3) - 4 exponent bits, 3 mantissa bits
        FP4,    ///< 4-bit floating point (future support)
        INT16,  ///< 16-bit signed integer (int16_t)
        INT32,  ///< 32-bit signed integer (int32_t)
        UINT16, ///< 16-bit unsigned integer (uint16_t)
        UINT32, ///< 32-bit unsigned integer (uint32_t)
    };*/

    /**
     * @brief Returns the size in bytes of a specific tensor data type.
     *
     * Given a tensor type enumeration value, this function returns the number of
     * bytes required to store a single scalar element of that type.
     *
     * @param type The tensor data type
     * @return size_t The size in bytes of the specified data type
     * @throws std::runtime_error If the type is not recognized
     */
    //export size_t sizeof_tensor_type( TensorType type ) {
    //    switch ( type ) {
    //        case TensorType::FP32:
    //            return sizeof( float );
    //        case TensorType::FP16:
    //            return sizeof( half );
    //        case TensorType::BF16:
    //            return sizeof( nv_bfloat16 );
    //        case TensorType::FP8:
    //            // NVIDIA CUDA offers different FP8 formats:
    //            // - __nv_fp8_e4m3: 4 exponent bits, 3 mantissa bits, plus sign bit
    //            // - __nv_fp8_e5m2: 5 exponent bits, 2 mantissa bits, plus sign bit
    //            // Currently using e4m3 format as it has better precision for typical neural network values
    //            return sizeof( __nv_fp8_e4m3 );
    //        case TensorType::FP4:
    //            // Placeholder for future FP4 support
    //            // Actual size may depend on NVIDIA's implementation
    //            return 1; // Assuming 1 byte storage for now
    //        case TensorType::INT16:
    //            return sizeof( int16_t );
    //        case TensorType::INT32:
    //            return sizeof( int );
    //        case TensorType::UINT16:
    //            return sizeof( uint16_t );
    //        case TensorType::UINT32:
    //            return sizeof( uint32_t );
    //        default:
    //            throw std::runtime_error( "Unknown tensor type" );
    //    }
    //};

    
}