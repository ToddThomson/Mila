module;
#include <string>
#include <sstream>
#include <cuda_fp16.h>
#include <stdexcept>
#include <numeric>
#include <variant>

export module Dnn.TensorType;

namespace Mila::Dnn
{
    // enumerator to indentify the datatype of a tensor.
    export enum class TensorType {
        kFP32, // float
        kFP64, // double
        kFP16, // half
        kINT16, // int16_t
        kINT32, // int32_t
        kINT64, // int64_t
        //kBF16, // nv_bfloat16
        kEmptyType,
    };

    // Given a datatype enum, returns the underlying number of bytes
    // for a scalar of that type
    export size_t sizeof_tensor_type( TensorType type ) {
        switch ( type ) {
        case TensorType::kFP64:
            return sizeof( double );
        case TensorType::kFP32:
            return sizeof( float );
        case TensorType::kFP16:
            return sizeof( half );
        case TensorType::kINT16:
            return sizeof( int16_t );
        case TensorType::kINT32:
            return sizeof( int );
        case TensorType::kINT64:
            return sizeof( int64_t );
//      case TensorType::BF16:
//            return sizeof( nv_bfloat16 );
        default:
            throw std::runtime_error("Unknown tensor type");
        }
    };

    export TensorType tensor_type_of( double* f ) { return TensorType::kFP64; };
    export TensorType tensor_type_of( float* f ) { return TensorType::kFP32; };
    export TensorType tensor_type_of( half* f ) { return TensorType::kFP16; };
    export TensorType tensor_type_of( int16_t* f ) { return TensorType::kINT16; };
    export TensorType tensor_type_of( int* f ) { return TensorType::kINT32; };
    export TensorType tensor_type_of( int64_t* f ) { return TensorType::kINT64; };

    // Define a type trait to map T to the corresponding ElementType variant
    export template <TensorType T>
    struct TensorTypeMapper;

    export template <>
    struct TensorTypeMapper<TensorType::kFP32> {
        using type = float;
    };

    export template <>
    struct TensorTypeMapper<TensorType::kFP64> {
        using type = double;
    };

    export template <>
    struct TensorTypeMapper<TensorType::kFP16> {
        using type = half;
    };

    export template <>
    struct TensorTypeMapper<TensorType::kINT16> {
        using type = int16_t;
    };

    export template <>
    struct TensorTypeMapper<TensorType::kINT32> {
        using type = int;
    };

    export template <>
    struct TensorTypeMapper<TensorType::kINT64> {
        using type = int64_t;
    };

    export std::string to_string( TensorType type ) {
        switch ( type ) {
        case TensorType::kFP64:
            return "kFP64";
        case TensorType::kFP32:
            return "kFP32";
        case TensorType::kFP16:
            return "kFP16";
        case TensorType::kINT16:
            return "kINT16";
        case TensorType::kINT32:
            return "kINT32";
        case TensorType::kINT64:
            return "kINT64";
        }
		return "Unknown Tensor type";
    }
}