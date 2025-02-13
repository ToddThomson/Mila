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
    export enum class TensorType {
        FP32, // float
        FP16, // half
        BF16, // nv_bfloat16
		FP8, // nv_float8
        INT16, // int16_t
        INT32, // int32_t
    };

    // Given a datatype enum, returns the underlying number of bytes
    // for a scalar of that type
    export size_t sizeof_tensor_type( TensorType type ) {
        switch ( type ) {
        case TensorType::FP32:
            return sizeof( float );
        case TensorType::FP16:
            return sizeof( half );
        case TensorType::BF16:
            return sizeof( nv_bfloat16 );
        case TensorType::FP8:
			return sizeof( __nv_fp8_e4m3 );
        case TensorType::INT16:
            return sizeof( int16_t );
        case TensorType::INT32:
            return sizeof( int );
        default:
            throw std::runtime_error("Unknown tensor type");
        }
    };

    // TODO: FP8 BFLOAT16
    export TensorType tensor_type_of( float* val ) { return TensorType::FP32; };
    export TensorType tensor_type_of( half* val ) { return TensorType::FP16; };
    export TensorType tensor_type_of( int16_t* val ) { return TensorType::INT16; };
    export TensorType tensor_type_of( int* val ) { return TensorType::INT32; };
    

    export std::string to_string( TensorType type ) {
        switch ( type ) {
            case TensorType::FP32:
                return "FP32";
            case TensorType::FP16:
                return "FP16";
            case TensorType::INT16:
                return "INT16";
            case TensorType::INT32:
                return "INT32";
                return "Unknown Tensor type";
        }
    };
}