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
    export enum class TensorType {
        FP32,   ///< 32-bit floating point (float)
        FP16,   ///< 16-bit floating point (half)
        BF16,   ///< 16-bit brain floating point (nv_bfloat16)
        FP8,    ///< 8-bit floating point (__nv_fp8_e4m3) - 4 exponent bits, 3 mantissa bits
        FP4,    ///< 4-bit floating point (future support)
        INT16,  ///< 16-bit signed integer (int16_t)
        INT32,  ///< 32-bit signed integer (int32_t)
        UINT16, ///< 16-bit unsigned integer (uint16_t)
        UINT32, ///< 32-bit unsigned integer (uint32_t)
    };

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
    export size_t sizeof_tensor_type( TensorType type ) {
        switch ( type ) {
            case TensorType::FP32:
                return sizeof( float );
            case TensorType::FP16:
                return sizeof( half );
            case TensorType::BF16:
                return sizeof( nv_bfloat16 );
            case TensorType::FP8:
                // NVIDIA CUDA offers different FP8 formats:
                // - __nv_fp8_e4m3: 4 exponent bits, 3 mantissa bits, plus sign bit
                // - __nv_fp8_e5m2: 5 exponent bits, 2 mantissa bits, plus sign bit
                // Currently using e4m3 format as it has better precision for typical neural network values
                return sizeof( __nv_fp8_e4m3 );
            case TensorType::FP4:
                // Placeholder for future FP4 support
                // Actual size may depend on NVIDIA's implementation
                return 1; // Assuming 1 byte storage for now
            case TensorType::INT16:
                return sizeof( int16_t );
            case TensorType::INT32:
                return sizeof( int );
            case TensorType::UINT16:
                return sizeof( uint16_t );
            case TensorType::UINT32:
                return sizeof( uint32_t );
            default:
                throw std::runtime_error( "Unknown tensor type" );
        }
    };

    /**
     * @brief Determines the tensor type for a float pointer.
     * @param val Pointer to a float value
     * @return TensorType Always returns TensorType::FP32
     */
    export TensorType tensor_type_of( float* val ) { return TensorType::FP32; };

    /**
     * @brief Determines the tensor type for a half pointer.
     * @param val Pointer to a half value
     * @return TensorType Always returns TensorType::FP16
     */
    export TensorType tensor_type_of( half* val ) { return TensorType::FP16; };

    /**
     * @brief Determines the tensor type for an nv_bfloat16 pointer.
     * @param val Pointer to an nv_bfloat16 value
     * @return TensorType Always returns TensorType::BF16
     */
    export TensorType tensor_type_of( nv_bfloat16* val ) { return TensorType::BF16; };

    /**
     * @brief Determines the tensor type for an __nv_fp8_e4m3 pointer.
     * @param val Pointer to an __nv_fp8_e4m3 value
     * @return TensorType Always returns TensorType::FP8
     */
    export TensorType tensor_type_of( __nv_fp8_e4m3* val ) { return TensorType::FP8; };

    /**
     * @brief Determines the tensor type for an int16_t pointer.
     * @param val Pointer to an int16_t value
     * @return TensorType Always returns TensorType::INT16
     */
    export TensorType tensor_type_of( int16_t* val ) { return TensorType::INT16; };

    /**
     * @brief Determines the tensor type for an int pointer.
     * @param val Pointer to an int value
     * @return TensorType Always returns TensorType::INT32
     */
    export TensorType tensor_type_of( int* val ) { return TensorType::INT32; };

    /**
     * @brief Determines the tensor type for a uint16_t pointer.
     * @param val Pointer to a uint16_t value
     * @return TensorType Always returns TensorType::UINT16
     */
    export TensorType tensor_type_of( uint16_t* val ) { return TensorType::UINT16; };

    /**
     * @brief Determines the tensor type for a uint32_t pointer.
     * @param val Pointer to a uint32_t value
     * @return TensorType Always returns TensorType::UINT32
     */
    export TensorType tensor_type_of( uint32_t* val ) { return TensorType::UINT32; };

    /**
     * @brief Converts a tensor type to its string representation.
     *
     * This function provides a human-readable string representation of each
     * tensor data type, which can be used for logging, debugging, or serialization.
     *
     * @param type The tensor data type
     * @return std::string The string representation of the tensor type
     */
    export std::string to_string( TensorType type ) {
        switch ( type ) {
            case TensorType::FP32:
                return "FP32";
            case TensorType::FP16:
                return "FP16";
            case TensorType::BF16:
                return "BF16";
            case TensorType::FP8:
                return "FP8";
            case TensorType::FP4:
                return "FP4";
            case TensorType::INT16:
                return "INT16";
            case TensorType::INT32:
                return "INT32";
            case TensorType::UINT16:
                return "UINT16";
            case TensorType::UINT32:
                return "UINT32";
            default:
                return "Unknown Tensor type";
        }
    };
}