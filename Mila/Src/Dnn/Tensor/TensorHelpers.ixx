/**
 * @file TensorHelpers.ixx
 * @brief Provides utility functions for tensor initialization and manipulation.
 */

module;
#include <cuda_fp16.h>
#include <random>
#include <cmath>
#include <type_traits>

export module Dnn.TensorHelpers;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn
{
    /**
 * @brief Initializes a tensor with random values within a specified range.
 *
 * This function populates a tensor with random floating-point values
 * uniformly distributed between the specified minimum and maximum values.
 * It handles both host and device memory resources appropriately, copying
 * data to the host for initialization if needed.
 *
 * @tparam TElementType The element data type of the tensor (float, half, etc.)
 * @tparam MR The memory resource type used by the tensor
 * @param tensor The tensor to initialize with random values
 * @param min The minimum value for the random distribution
 * @param max The maximum value for the random distribution
 *
 * @note Uses a fixed seed (42) for reproducible results rather than truly random values
 */
    export template<typename TElementType, typename MR = Compute::CpuMemoryResource>
        requires ValidTensorType<TElementType> && std::is_base_of_v<Compute::MemoryResource, MR>
    void random( Tensor<TElementType, MR>& tensor, TElementType min, TElementType max ) {
        std::random_device rd;
        std::mt19937 gen( 42 ); // TODO: rd() );

        if constexpr ( std::is_same_v<TElementType, half> ) {
            // For half precision, we need to use float for random generation and then convert
            float min_float = static_cast<float>( min );
            float max_float = static_cast<float>( max );
            //float min_float = __half2float( min );
            //float max_float = __half2float( max );
            std::uniform_real_distribution<float> dis( min_float, max_float );

            if constexpr ( std::is_same_v<MR, Compute::CudaMemoryResource> ) {
                auto temp = tensor.template to<Compute::HostMemoryResource>();

                for ( size_t i = 0; i < temp.size(); ++i ) {
                    float random_val = dis( gen );
                    temp.raw_data()[ i ] = static_cast<half>( random_val );
                    //temp.raw_data()[ i ] = __float2half( random_val );
                }
                tensor = temp.template to<MR>();
            }
            else {
                for ( size_t i = 0; i < tensor.size(); ++i ) {
                    float random_val = dis( gen );
                    tensor.raw_data()[ i ] = static_cast<half>( random_val );
                    //tensor.raw_data()[ i ] = __float2half( random_val );
                }
            }
        }
        else {
            // Original implementation for float and other types
            std::uniform_real_distribution<TElementType> dis( min, max );

            if constexpr ( std::is_same_v<MR, Compute::CudaMemoryResource> ) {
                auto temp = tensor.template to<Compute::HostMemoryResource>();

                TElementType* temp_data = temp.raw_data();
                for ( size_t i = 0; i < temp.size(); ++i ) {
                    temp_data[ i ] = dis( gen );
                }
                tensor = temp.template to<MR>();
            }
            else {
                TElementType* tensor_data = tensor.raw_data();
                for ( size_t i = 0; i < tensor.size(); ++i ) {
                    tensor_data[ i ] = dis( gen );
                }
            }
        }
    }

    /**
     * @brief Initializes a tensor with Xavier/Glorot uniform initialization.
     *
     * Xavier initialization is a method designed to keep the scale of gradients
     * roughly the same in all layers of a neural network. It initializes weights
     * with values sampled from a uniform distribution with limits calculated as
     * a function of the input and output sizes of the layer.
     *
     * The distribution range is [-limit, limit] where:
     * limit = sqrt(6 / (input_size + output_size))
     *
     * @tparam TElementType The element data type of the tensor (float, half, etc.)
     * @tparam MR The memory resource type used by the tensor
     * @param tensor The tensor to initialize with Xavier initialization
     * @param input_size The size of the input dimension
     * @param output_size The size of the output dimension
     *
     * @note Uses a fixed seed (42) for reproducible results rather than truly random values
     * @see http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
     */
    export template<typename TElementType, typename MR>
        requires ValidTensorType<TElementType>&& std::is_base_of_v<Compute::MemoryResource, MR>
    void xavier( Tensor<TElementType, MR>& tensor, size_t input_size, size_t output_size ) {
        float limit = std::sqrt( 6.0f / (input_size + output_size) );

        if constexpr ( std::is_same_v<TElementType, half> ) {
            // For half precision, convert the limit and use float for distribution
            std::random_device rd;
            std::mt19937 gen( 42 ); // TJT: revert back to rd() );
            std::uniform_real_distribution<float> dis( -limit, limit );

            if constexpr ( std::is_same_v<MR, Compute::CudaMemoryResource> ) {
                auto temp = tensor.template to<Compute::HostMemoryResource>();

                half* temp_data = temp.raw_data();
                for ( size_t i = 0; i < temp.size(); ++i ) {
                    float random_val = dis( gen );
                    //temp_data[ i ] = __float2half( random_val );
                    temp_data[ i ] = static_cast<half>( random_val );
                }
                tensor = temp.template to<MR>();
            }
            else {
                half* tensor_data = tensor.raw_data();
                for ( size_t i = 0; i < tensor.size(); ++i ) {
                    float random_val = dis( gen );
                    //tensor_data[ i ] = __float2half( random_val );
                    tensor_data[ i ] = static_cast<half>( random_val );
                }
            }
        }
        else {
            // Original implementation for float and other types
            std::random_device rd;
            std::mt19937 gen( 42 ); // TJT: revert back to rd() );
            std::uniform_real_distribution<TElementType> dis( -limit, limit );

            if constexpr ( std::is_same_v<MR, Compute::CudaMemoryResource> ) {
                auto temp = tensor.template to<Compute::HostMemoryResource>();

                TElementType* temp_data = temp.raw_data();
                for ( size_t i = 0; i < temp.size(); ++i ) {
                    temp_data[ i ] = dis( gen );
                }
                tensor = temp.template to<MR>();
            }
            else {
                TElementType* tensor_data = tensor.raw_data();
                for ( size_t i = 0; i < tensor.size(); ++i ) {
                    tensor_data[ i ] = dis( gen );
                }
            }
        }
    }

}