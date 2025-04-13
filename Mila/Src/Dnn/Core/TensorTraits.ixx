/**
 * @file TensorTraits.ixx
 * @brief Provides type traits for mapping C++ types to tensor data types.
 */

module;
#include <vector>
#include <cuda_fp16.h>
#include <type_traits>

export module Dnn.TensorTraits;

import Dnn.TensorType;

namespace Mila::Dnn
{
	/**
	 * @brief Primary template for mapping C++ types to tensor data types.
	 *
	 * This trait structure provides a mapping between native C++ types and
	 * the corresponding TensorType enumeration. The primary template is
	 * intentionally undefined; specializations must be provided for each
	 * supported type.
	 *
	 * @tparam TElementType The C++ type to map to a tensor type
	 */
	template <typename T>
	struct TensorTypeTrait;

	/**
	 * @brief Specialization of TensorTypeTrait for float type.
	 */
	template <>
	struct TensorTypeTrait<float> {
		static constexpr TensorType value = TensorType::FP32;
	};

	/**
	 * @brief Specialization of TensorTypeTrait for half-precision float type.
	 */
	template <>
	struct TensorTypeTrait<half> {
		static constexpr TensorType value = TensorType::FP16;
	};

	/**
	 * @brief Specialization of TensorTypeTrait for 16-bit signed integer type.
	 */
	template <>
	struct TensorTypeTrait<int16_t> {
		static constexpr TensorType value = TensorType::INT16;
	};

	/**
	 * @brief Specialization of TensorTypeTrait for 32-bit signed integer type.
	 */
	template <>
	struct TensorTypeTrait<int> {
		static constexpr TensorType value = TensorType::INT32;
	};

	/**
	 * @brief Specialization of TensorTypeTrait for 16-bit unsigned integer type.
	 */
	template <>
	struct TensorTypeTrait<uint16_t> {
		static constexpr TensorType value = TensorType::UINT16;
	};

	/**
	 * @brief Specialization of TensorTypeTrait for 32-bit unsigned integer type.
	 */
	template <>
	struct TensorTypeTrait<uint32_t> {
		static constexpr TensorType value = TensorType::UINT32;
	};

	/**
	 * @brief Concept that constrains types to those that have a valid tensor type mapping.
	 *
	 * This concept ensures that a type TElementType has a corresponding TensorTypeTrait
	 * specialization and that the trait provides a value that is convertible to
	 * the TensorType enumeration.
	 *
	 * @tparam TElementType The type to check for valid tensor type mapping
	 */
	export template <typename T>
		concept ValidTensorType = requires {
			{ TensorTypeTrait<T>::value } -> std::convertible_to<TensorType>;
	};

	/**
	 * @brief Concept that verifies both input and compute types have valid tensor type mappings.
	 *
	 * This concept is used primarily with template operations and modules that work with
	 * both input and computation tensor types, ensuring both types have valid mappings.
	 *
	 * @tparam TInput The input tensor element type
	 * @tparam TCompute The computation tensor element type
	 */
	export template <typename TInput, typename TCompute>
		concept ValidTensorTypes = ValidTensorType<TInput> && ValidTensorType<TCompute>;
}
