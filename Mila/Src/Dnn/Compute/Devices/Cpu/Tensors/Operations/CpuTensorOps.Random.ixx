/**
 * @file CpuTensorOps.Random.ixx
 * @brief CPU random initialization partition for tensor buffers.
 *
 * Host-side normal and uniform random initialization using the standard library
 * distributions, seeded from Core::RandomGenerator. Mirrors the CUDA RandomOps
 * surface so TensorOps<Cpu> exposes the same fill_normal/fill_uniform entry points.
 */

module;
#include <random>
#include <type_traits>

export module Compute.CpuTensorOps:Random;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.ExecutionContext;
import Compute.DeviceType;
import Compute.CpuTensorDataTypeTraits;
import Core.RandomGenerator;

namespace Mila::Dnn::Compute::Cpu
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief CPU specialization of random tensor initialization.
     *
     * Generates host-side values with std::mt19937 (seeded from the global
     * Core::RandomGenerator) and the standard normal/uniform distributions, converting
     * to the tensor's native element type with the same static_cast idiom as FillOps.
     * The ExecutionContext is accepted for API uniformity with CUDA but ignored: CPU
     * generation is synchronous.
     */
    export struct RandomOps
    {
        /**
         * @brief Fill a float tensor with values drawn from N(mean, stddev^2).
         *
         * @note A generator copy is taken per call from Core::RandomGenerator, matching
         *       the established convention; reproducibility tracks the global seed.
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource> && TensorDataTypeTraits<TDataType>::is_float_type
        static void fill_normal(
            Tensor<TDataType, TMemoryResource>& tensor,
            float mean,
            float stddev,
            [[maybe_unused]] IExecutionContext* exec_context = nullptr )
        {
            const size_t n = tensor.size();
            if ( n == 0 )
                return;

            using NativeType = typename CpuTensorDataTypeTraits::template native_type<TDataType>;

            NativeType* dst = static_cast<NativeType*>( tensor.data() );
            std::mt19937 generator = Core::RandomGenerator::getInstance().getGenerator();
            std::normal_distribution<float> distribution( mean, stddev );

            for ( size_t i = 0; i < n; ++i )
            {
                dst[ i ] = static_cast<NativeType>( distribution( generator ) );
            }
        }

        /**
         * @brief Fill a float tensor with uniform values in [min_val, max_val).
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource> && TensorDataTypeTraits<TDataType>::is_float_type
        static void fill_uniform(
            Tensor<TDataType, TMemoryResource>& tensor,
            float min_val,
            float max_val,
            [[maybe_unused]] IExecutionContext* exec_context = nullptr )
        {
            const size_t n = tensor.size();
            if ( n == 0 )
                return;

            using NativeType = typename CpuTensorDataTypeTraits::template native_type<TDataType>;

            NativeType* dst = static_cast<NativeType*>( tensor.data() );
            std::mt19937 generator = Core::RandomGenerator::getInstance().getGenerator();
            std::uniform_real_distribution<float> distribution( min_val, max_val );

            for ( size_t i = 0; i < n; ++i )
            {
                dst[ i ] = static_cast<NativeType>( distribution( generator ) );
            }
        }
    };
}
