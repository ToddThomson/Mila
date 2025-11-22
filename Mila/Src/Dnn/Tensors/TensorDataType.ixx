/**
 * @file TensorDataType.ixx
 * @brief Abstract tensor data type enumeration and traits system for device-agnostic tensor operations
 *
 * This module provides a comprehensive type abstraction layer that enables tensor operations
 * across different compute devices (CPU, CUDA, Metal, OpenCL) without exposing device-specific
 * concrete types to host compilation. The system supports standard floating-point and integer
 * types as well as advanced precision formats including FP8, FP4, and packed sub-byte types.
 */

module;
#include <string>
#include <stdexcept>

export module Dnn.TensorDataType;

namespace Mila::Dnn
{
    /**
     * @brief Enumeration of supported abstract tensor data types
     *
     * Defines device-agnostic tensor data types that can be mapped to concrete
     * implementations on different compute devices. This abstraction prevents
     * host compilation issues with device-specific types while enabling
     * compile-time dispatch and optimization.
     *
     * Supported categories:
     * - Standard floating-point: FP32
     * - Reduced precision floating-point: FP16, BF16, FP8_E4M3, FP8_E5M2
     * - Integer types: Various widths from 8-bit to 32-bit, signed and unsigned
     *
     * @note Device-only types (FP16, BF16, FP8) require device-accessible memory
     * @note Packed sub-byte types (FP4, INT4, UINT4) are planned for future implementation
     */
    export enum class TensorDataType {
        FP32,     ///< 32-bit IEEE 754 floating point, host-compatible
        FP16,     ///< 16-bit half precision floating point, device-only
        BF16,     ///< 16-bit brain floating point, device-only
        FP8_E4M3, ///< 8-bit floating point with 4-bit exponent and 3-bit mantissa, device-only
        FP8_E5M2, ///< 8-bit floating point with 5-bit exponent and 2-bit mantissa, device-only
        
        // Future packed types (not yet implemented)
        // FP4_E2M1, ///< 4-bit floating point with 2-bit exponent and 1-bit mantissa, packed - FUTURE
        // FP4_E3M0, ///< 4-bit floating point with 3-bit exponent and 0-bit mantissa, packed - FUTURE
        // INT4,     ///< 4-bit signed integer, packed - FUTURE
        // UINT4,    ///< 4-bit unsigned integer, packed - FUTURE
        
        INT8,     ///< 8-bit signed integer
        INT16,    ///< 16-bit signed integer, host-compatible
        INT32,    ///< 32-bit signed integer, host-compatible
        UINT8,    ///< 8-bit unsigned integer
        UINT16,   ///< 16-bit unsigned integer, host-compatible
        UINT32,   ///< 32-bit unsigned integer, host-compatible
    };

    /**
     * @brief Converts TensorDataType enumeration to human-readable string
     */
    export inline std::string tensorDataTypeToString( TensorDataType type )
    {
        switch (type)
        {
            case TensorDataType::FP32: return "FP32";
            case TensorDataType::FP16: return "FP16";
            case TensorDataType::BF16: return "BF16";
            case TensorDataType::FP8_E4M3: return "FP8_E4M3";
            case TensorDataType::FP8_E5M2: return "FP8_E5M2";
                // FUTURE: packed types commented out
                // case TensorDataType::FP4_E2M1: return "FP4_E2M1";
                // case TensorDataType::FP4_E3M0: return "FP4_E3M0";
                // case TensorDataType::INT4: return "INT4";
                // case TensorDataType::UINT4: return "UINT4";
            case TensorDataType::INT8: return "INT8";
            case TensorDataType::INT16: return "INT16";
            case TensorDataType::INT32: return "INT32";
            case TensorDataType::UINT8: return "UINT8";
            case TensorDataType::UINT16: return "UINT16";
            case TensorDataType::UINT32: return "UINT32";
            default: return "Unknown";
        }
    };

    export TensorDataType parseTensorDataType( const std::string& type_str )
    {
        if (type_str == "FP32")
            return TensorDataType::FP32;
        if (type_str == "FP16")
            return TensorDataType::FP16;
        if (type_str == "BF16")
            return TensorDataType::BF16;
        if (type_str == "FP8_E4M3")
            return TensorDataType::FP8_E4M3;
        if (type_str == "FP8_E5M2")
            return TensorDataType::FP8_E5M2;
        // FUTURE: packed types commented out
        // if (type_str == "FP4_E2M1")
        // return TensorDataType::FP4_E2M1;
        // if (type_str == "FP4_E3M0")
        // return TensorDataType::FP4_E3M0;
        // if (type_str == "INT4")
        // return TensorDataType::INT4;
        // if (type_str == "UINT4")
        // return TensorDataType::UINT4;
        if (type_str == "INT8")
            return TensorDataType::INT8;
        if (type_str == "INT16")
            return TensorDataType::INT16;
        if (type_str == "INT32")
            return TensorDataType::INT32;
        if (type_str == "UINT8")
            return TensorDataType::UINT8;
        if (type_str == "UINT16")
            return TensorDataType::UINT16;
        if (type_str == "UINT32")
            return TensorDataType::UINT32;
        
        throw std::invalid_argument( "Unknown TensorDataType string: " + type_str );
    }

    /**
     * @brief Alias for TensorDataType enumeration
     *
     * Provides a concise alias for the TensorDataType enumeration
     * to improve code readability in tensor-related contexts.
	 */
	export using dtype_t = TensorDataType;
}