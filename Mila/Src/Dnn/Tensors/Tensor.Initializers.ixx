/**
 * @file Tensor.Initializers.ixx
 * @brief Tensor initialization algorithms with host distribution generation and backend dispatch
 */

module;
#include <vector>
#include <random>
#include <cmath>
#include <type_traits>
#include <concepts>
#include <span>

export module Dnn.TensorInitializers;

import Core.RandomGenerator;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Compute.MemoryResourceTraits;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaManagedMemoryResource;
import Compute.CudaPinnedMemoryResource;

namespace Mila::Dnn
{
    /**
     * @brief Host value type mapping for tensor data types
     *
     * Maps all integer tensor types to int32_t host generation,
     * and all floating tensor types to float host generation.
     */
    template<TensorDataType TDataType>
    using host_value_t = std::conditional_t<TensorDataTypeTraits<TDataType>::is_integer_type, int32_t, float>;

    namespace Detail
    {
        /**
         * @brief Generates host-native random values using appropriate distribution
         */
        template<TensorDataType TDataType>
        auto generate_host_values( size_t count, host_value_t<TDataType> min_val, host_value_t<TDataType> max_val ) {
            auto gen = Core::RandomGenerator::getInstance().getGenerator();

            if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type)
            {
                std::uniform_int_distribution<int32_t> dis( min_val, max_val );

                std::vector<int32_t> values( count );
                for (size_t i = 0; i < count; ++i)
                {
                    values[i] = dis( gen );
                }
                
                return values;
            }
            else
            {
                std::uniform_real_distribution<float> dis( min_val, max_val );

                std::vector<float> values( count );
                for (size_t i = 0; i < count; ++i)
                {
                    values[i] = dis( gen );
                }
                return values;
            }
        }

        /**
         * @brief Core uniform distribution implementation with TensorOps dispatch
         *
         * Generates host values and forwards them to TensorOps::fill for device
         * dispatch and conversion. Uses span to avoid extra copies.
         */
        template<TensorDataType TDataType, typename TMemoryResource>
        void fill_uniform_distribution( Tensor<TDataType, TMemoryResource>& tensor,
            host_value_t<TDataType> min_val, host_value_t<TDataType> max_val ) {

            auto host_values = generate_host_values<TDataType>( tensor.size(), min_val, max_val );

            // Create span over generated host values and forward to device-dispatching fill.
            using HV = host_value_t<TDataType>;
            std::span<const HV> values_span{ reinterpret_cast<const HV*>(host_values.data()), host_values.size() };

            fill( tensor, values_span );
        }

        /**
         * @brief Constant value fill with TensorOps dispatch
         *
         * Broadcasts a host scalar into the tensor using device-dispatched fill.
         */
        template<TensorDataType TDataType, typename TMemoryResource>
        void fill_constant_from_host( Tensor<TDataType, TMemoryResource>& tensor, host_value_t<TDataType> host_value ) {
            fill( tensor, host_value );
        }

        /**
         * @brief Xavier initialization using host-native calculations with backend dispatch
         */
        template<TensorDataType TDataType, typename TMemoryResource>
        void initialize_xavier( Tensor<TDataType, TMemoryResource>& tensor,
            size_t input_size, size_t output_size ) {
            // Handle edge cases
            if (input_size == 0 && output_size == 0)
            {
                // Fallback initialization
                if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type)
                {
                    fill_uniform_distribution( tensor, -1, 1 );
                }
                else
                {
                    fill_uniform_distribution( tensor, -0.01f, 0.01f );
                }
                return;
            }

            size_t total_size = (input_size == 0) ? output_size :
                (output_size == 0) ? input_size :
                (input_size + output_size);

            float limit = std::sqrt( 6.0f / static_cast<float>(total_size) );

            // Apply Xavier range using appropriate host type
            if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type)
            {
                // For integer tensors, ensure reasonable range (minimum ±1)
                int32_t int_limit = std::max( static_cast<int32_t>(std::round( limit )), 1 );
                fill_uniform_distribution( tensor, -int_limit, int_limit );
            }
            else
            {
                // For float tensors, use computed limit directly
                fill_uniform_distribution( tensor, -limit, limit );
            }
        }
    }

    // ====================================================================
    // Public API Functions
    // ====================================================================

    /**
     * @brief Fills tensor with random values using host-native distributions
     *
     * Generates random values on host using standard distributions, then dispatches
     * to appropriate backend for device transfer and quantization to target type.
     *
     * @param tensor Target tensor to initialize
     * @param min_val Minimum value (int32_t for integer tensors, float for floating tensors)
     * @param max_val Maximum value (int32_t for integer tensors, float for floating tensors)
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void random( Tensor<TDataType, TMemoryResource>& tensor, host_value_t<TDataType> min_val, host_value_t<TDataType> max_val ) {
        if (min_val > max_val)
        {
            throw std::invalid_argument( "min_val must be <= max_val" );
        }

        Detail::fill_uniform_distribution( tensor, min_val, max_val );
    }

    /**
     * @brief Xavier/Glorot uniform initialization with host-native calculations
     *
     * Computes initialization range on host, generates distribution values,
     * then dispatches to backend for device transfer and quantization.
     *
     * @param tensor Target tensor to initialize
     * @param input_size Number of input connections
     * @param output_size Number of output connections
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void xavier( Tensor<TDataType, TMemoryResource>& tensor, size_t input_size, size_t output_size ) {
        Detail::initialize_xavier( tensor, input_size, output_size );
    }

    /**
     * @brief Fills tensor with zeros using backend-optimized implementations
     *
     * Dispatches to backend for efficient zero-fill operations, potentially
     * using device-native memset operations for optimal performance.
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void zeros( Tensor<TDataType, TMemoryResource>& tensor ) {
        if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type)
        {
            Detail::fill_constant_from_host( tensor, static_cast<host_value_t<TDataType>>(0) );
        }
        else
        {
            Detail::fill_constant_from_host( tensor, static_cast<host_value_t<TDataType>>(0.0f) );
        }
    }

    /**
     * @brief Fills tensor with ones using backend-optimized implementations
     *
     * Dispatches to backend for efficient constant-fill operations with
     * appropriate quantization to target tensor data type.
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void ones( Tensor<TDataType, TMemoryResource>& tensor ) {
        if constexpr (TensorDataTypeTraits<TDataType>::is_integer_type)
        {
            Detail::fill_constant_from_host( tensor, static_cast<host_value_t<TDataType>>(1) );
        }
        else
        {
            Detail::fill_constant_from_host( tensor, static_cast<host_value_t<TDataType>>(1.0f) );
        }
    }

    // ====================================================================
    // Convenience Overloads for Common Patterns
    // ====================================================================

    /**
     * @brief Random initialization with symmetric range for integer tensors
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>&&
    TensorDataTypeTraits<TDataType>::is_integer_type
        void random( Tensor<TDataType, TMemoryResource>& tensor, int32_t magnitude ) {
        random( tensor, -magnitude, magnitude );
    }

    /**
     * @brief Random initialization with symmetric range for floating tensors
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>&& TensorDataTypeTraits<TDataType>::is_float_type
    void random( Tensor<TDataType, TMemoryResource>& tensor, float magnitude ) {
        random( tensor, -magnitude, magnitude );
    }
}