/**
 * @file TensorHelpers.ixx
 * @brief Provides utility functions for tensor initialization and manipulation.
 */

module;
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <random>
#include <cmath>
#include <type_traits>

export module Dnn.TensorHelpers;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Core.RandomGenerator;

namespace Mila::Dnn
{
    namespace Detail
    {
        // Helper for converting float values to any supported tensor type
        template<typename T>
        T convert_from_float( float value ) {
            if constexpr ( std::is_same_v<T, float> ) {
                return value;
            }
            else if constexpr ( TensorTrait<T>::is_float_type ) {
                // Safe conversion for all floating point types
                return static_cast<T>(value);
            }
            else if constexpr ( TensorTrait<T>::is_integer_type ) {
                // Round float to nearest integer value
                return static_cast<T>(std::round( value ));
            }
            else {
                // Fallback for any other types
                return static_cast<T>(value);
            }
        }

        // Helper for float-based random distribution
        template<typename TDataType, typename MR>
        void fill_with_float_distribution( Tensor<TDataType, MR>& tensor, float min_val, float max_val ) {

            auto gen = Core::RandomGenerator::getInstance().getGenerator();
            std::uniform_real_distribution<float> dis( min_val, max_val );

            if constexpr ( std::is_same_v<MR, Compute::CudaMemoryResource> ) {
                // For device memory, use host tensor as intermediate
                auto temp = tensor.template to<Compute::HostMemoryResource>();

                for ( size_t i = 0; i < temp.size(); ++i ) {
                    temp.raw_data()[ i ] = convert_from_float<TDataType>( dis( gen ) );
                }
                tensor = temp.template to<MR>();
            }
            else {
                // Direct filling for host memory
                for ( size_t i = 0; i < tensor.size(); ++i ) {
                    tensor.raw_data()[ i ] = convert_from_float<TDataType>( dis( gen ) );
                }
            }
        }

        // Helper for integer-based random distribution
        template<typename TDataType, typename MR>
        void fill_with_int_distribution( Tensor<TDataType, MR>& tensor, TDataType min_val, TDataType max_val ) {

            auto gen = Core::RandomGenerator::getInstance().getGenerator();
            std::uniform_int_distribution<TDataType> dis( min_val, max_val );

            if constexpr ( std::is_same_v<MR, Compute::CudaMemoryResource> ) {
                // For device memory, use host tensor as intermediate
                auto temp = tensor.template to<Compute::HostMemoryResource>();

                TDataType* temp_data = temp.raw_data();
                for ( size_t i = 0; i < temp.size(); ++i ) {
                    temp_data[ i ] = dis( gen );
                }
                tensor = temp.template to<MR>();
            }
            else {
                // Direct filling for host memory
                TDataType* tensor_data = tensor.raw_data();
                for ( size_t i = 0; i < tensor.size(); ++i ) {
                    tensor_data[ i ] = dis( gen );
                }
            }
        }
    }

    /**
     * @brief Initializes a tensor with random values within a specified range.
     *
     * This function populates a tensor with random values uniformly distributed
     * between the specified minimum and maximum values. It leverages TensorTraits
     * to properly handle all supported tensor types:
     * - Floating-point types (float, half, nv_bfloat16, __nv_fp8_e4m3, __nv_fp8_e5m2)
     * - Integer types (int, int16_t, uint16_t, uint32_t)
     *
     * For floating-point types, values are sampled from a uniform real distribution.
     * For integer types, values are sampled from a uniform int distribution.
     *
     * The function uses the global random number generator from Mila's initialization,
     * ensuring consistent and reproducible random values across all tensor operations.
     *
     * @tparam TDataType The element data type of the tensor
     * @tparam MR The memory resource type used by the tensor
     * @param tensor The tensor to initialize with random values
     * @param min The minimum value for the random distribution
     * @param max The maximum value for the random distribution
     */
    export template<typename TDataType, typename MR = Compute::CpuMemoryResource>
        requires ValidTensorType<TDataType>&& std::is_base_of_v<Compute::MemoryResource, MR>
    void random( Tensor<TDataType, MR>& tensor, TDataType min, TDataType max ) {
        if constexpr ( TensorTrait<TDataType>::is_float_type ) {
            // For all floating-point types defined in TensorTraits
            Detail::fill_with_float_distribution( tensor, static_cast<float>(min), static_cast<float>(max) );
        }
        else if constexpr ( TensorTrait<TDataType>::is_integer_type ) {
            // For all integer types defined in TensorTraits
            Detail::fill_with_int_distribution( tensor, min, max );
        }
    }

    /**
     * @brief Initializes a tensor with Xavier/Glorot uniform initialization.
     *
     * Xavier initialization helps maintain consistent gradients across neural network
     * layers by initializing weights from a uniform distribution with carefully
     * calculated bounds based on input and output dimensions.
     *
     * Distribution range: [-limit, limit] where limit = sqrt(6 / (input_size + output_size))
     *
     * This implementation leverages TensorTraits to support all tensor types:
     * - Floating-point types use the standard Xavier distribution
     * - Integer types use a scaled distribution appropriate for their range
     *
     * The function uses the global random number generator from Mila's initialization,
     * ensuring consistent and reproducible random values across all tensor operations.
     *
     * @tparam TDataType The element data type of the tensor
     * @tparam MR The memory resource type used by the tensor
     * @param tensor The tensor to initialize
     * @param input_size The size of the input dimension
     * @param output_size The size of the output dimension
     *
     * @see http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
     */
    export template<typename TDataType, typename MR>
        requires ValidTensorType<TDataType>&& std::is_base_of_v<Compute::MemoryResource, MR>
    void xavier( Tensor<TDataType, MR>& tensor, size_t input_size, size_t output_size ) {
        float limit = std::sqrt( 6.0f / (input_size + output_size) );

        if constexpr ( TensorTrait<TDataType>::is_float_type ) {
            Detail::fill_with_float_distribution( tensor, -limit, limit );
        }
        else if constexpr ( TensorTrait<TDataType>::is_integer_type ) {

            if constexpr ( std::is_signed_v<TDataType> ) {
                TDataType int_limit = static_cast<TDataType>(std::round( limit *
                    static_cast<float>(std::numeric_limits<TDataType>::max()) ));
                Detail::fill_with_int_distribution( tensor, -int_limit, int_limit );
            }
            else {
                TDataType int_max = static_cast<TDataType>(std::round( 2.0f * limit *
                    static_cast<float>(std::numeric_limits<TDataType>::max()) ));
                TDataType int_min = 0;
                Detail::fill_with_int_distribution( tensor, int_min, int_max );
            }
        }
    }
}