/**
 * @file TensorTraits.ixx
 * @brief Provides type traits for tensor data types with compile-time type information.
 */

module;
#include <vector>
#include <type_traits>
#include <cstdint>
#include <string_view>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

export module Dnn.TensorTraits;

namespace Mila::Dnn
{
	/**
	 * @brief Primary template for tensor type traits.
	 *
	 * This trait structure provides compile-time information about tensor element types.
	 * The primary template is intentionally undefined; specializations must be provided
	 * for each supported type.
	 *
	 * @tparam T The tensor element type
	 */
	export template <typename T>
	struct TensorTrait;

	/**
	 * @brief Specialization of TensorTrait for float type.
	 */
	template <>
	struct TensorTrait<float> {
		static constexpr bool is_float_type = true;
		static constexpr bool is_integer_type = false;
		static constexpr bool is_supported = true;
		static constexpr std::string_view type_name = "FP32";
		static constexpr size_t size_in_bytes = sizeof( float );
	};

	/**
	 * @brief Specialization of TensorTrait for half-precision float type.
	 */
	template <>
	struct TensorTrait<half> {
		static constexpr bool is_float_type = true;
		static constexpr bool is_integer_type = false;
		static constexpr bool is_supported = true;
		static constexpr std::string_view type_name = "FP16";
		static constexpr size_t size_in_bytes = sizeof( half );
	};

	/**
	 * @brief Specialization of TensorTrait for NVIDIA bfloat16 type.
	 */
	template <>
	struct TensorTrait<nv_bfloat16> {
		static constexpr bool is_float_type = true;
		static constexpr bool is_integer_type = false;
		static constexpr bool is_supported = true;
		static constexpr std::string_view type_name = "BF16";
		static constexpr size_t size_in_bytes = sizeof( nv_bfloat16 );
	};

	/**
	 * @brief Specialization of TensorTrait for 8-bit floating point type (e4m3).
	 */
	template <>
	struct TensorTrait<__nv_fp8_e4m3> {
		static constexpr bool is_float_type = true;
		static constexpr bool is_integer_type = false;
		static constexpr bool is_supported = true;
		static constexpr std::string_view type_name = "FP8_E4M3";
		static constexpr size_t size_in_bytes = sizeof( __nv_fp8_e4m3 );
	};

	/**
	 * @brief Specialization of TensorTrait for alternative 8-bit floating point type (e5m2).
	 */
	template <>
	struct TensorTrait<__nv_fp8_e5m2> {
		static constexpr bool is_float_type = true;
		static constexpr bool is_integer_type = false;
		static constexpr bool is_supported = true;
		static constexpr std::string_view type_name = "FP8_E5M2";
		static constexpr size_t size_in_bytes = sizeof( __nv_fp8_e5m2 );
	};

	/**
	 * @brief Specialization of TensorTrait for 16-bit signed integer type.
	 */
	template <>
	struct TensorTrait<int16_t> {
		static constexpr bool is_float_type = false;
		static constexpr bool is_integer_type = true;
		static constexpr bool is_supported = true;
		static constexpr std::string_view type_name = "INT16";
		static constexpr size_t size_in_bytes = sizeof( int16_t );
	};

	/**
	 * @brief Specialization of TensorTrait for 32-bit signed integer type.
	 */
	template <>
	struct TensorTrait<int> {
		static constexpr bool is_float_type = false;
		static constexpr bool is_integer_type = true;
		static constexpr bool is_supported = true;
		static constexpr std::string_view type_name = "INT32";
		static constexpr size_t size_in_bytes = sizeof( int );
	};

	/**
	 * @brief Specialization of TensorTrait for 16-bit unsigned integer type.
	 */
	template <>
	struct TensorTrait<uint16_t> {
		static constexpr bool is_float_type = false;
		static constexpr bool is_integer_type = true;
		static constexpr bool is_supported = true;
		static constexpr std::string_view type_name = "UINT16";
		static constexpr size_t size_in_bytes = sizeof( uint16_t );
	};

	/**
	 * @brief Specialization of TensorTrait for 32-bit unsigned integer type.
	 */
	template <>
	struct TensorTrait<uint32_t> {
		static constexpr bool is_float_type = false;
		static constexpr bool is_integer_type = true;
		static constexpr bool is_supported = true;
		static constexpr std::string_view type_name = "UINT32";
		static constexpr size_t size_in_bytes = sizeof( uint32_t );
	};

	/**
	 * @brief Get the string representation of a tensor element type.
	 *
	 * @tparam T The tensor element type
	 * @return constexpr std::string_view The type name as a string
	 */
	export template <typename T>
		constexpr std::string_view tensor_type_name() {
		return TensorTrait<T>::type_name;
	}

	/**
	 * @brief Get the size in bytes of a tensor element type.
	 *
	 * @tparam T The tensor element type
	 * @return constexpr size_t Size in bytes
	 */
	export template <typename T>
		constexpr size_t tensor_type_size() {
		return TensorTrait<T>::size_in_bytes;
	}

	/**
	 * @brief Concept that constrains types to those with valid tensor trait specializations.
	 *
	 * @tparam T The type to check for valid tensor trait
	 */
	export template <typename T>
		concept ValidTensorType = requires {
			{ TensorTrait<T>::is_supported } -> std::convertible_to<bool>;
			requires TensorTrait<T>::is_supported == true;
	};

	/**
	 * @brief Concept that constrains types to valid floating-point tensor types.
	 *
	 * @tparam T The type to check for valid floating-point tensor type
	 */
	export template <typename T>
		concept ValidFloatTensorType = ValidTensorType<T> && TensorTrait<T>::is_float_type;

	/**
	 * @brief Concept that verifies both types are valid floating-point tensor types.
	 *
	 * @tparam TPrecision The precision tensor element type (must be a floating-point type)
	 * @tparam TInput The input tensor element type (must be a floating-point type)
	 */
	export template <typename TPrecision, typename TInput>
		concept ValidFloatTensorTypes = ValidFloatTensorType<TPrecision> && ValidFloatTensorType<TInput>;

	/**
	 * @brief Concept that verifies both input and compute types have valid tensor trait mappings.
	 *
	 * @tparam TInput The input tensor element type
	 * @tparam TCompute The computation tensor element type
	 */
	export template <typename TPrecision, typename TInput>
		concept ValidTensorTypes = ValidTensorType<TPrecision> && ValidTensorType<TInput>;
}