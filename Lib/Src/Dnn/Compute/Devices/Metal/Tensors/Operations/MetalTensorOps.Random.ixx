/**
 * @file MetalTensorOps.Random.ixx
 * @brief Random values for a Metal tensor, drawn on the host, since the memory is shared.
 */

export module Compute.MetalTensorOps:Random;

import std;
import :Fill;
import Compute.ExecutionContext;
import Core.RandomGenerator;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal
{
    /// Runs on the host now, once the context's earlier work has finished. Reproducibility follows the global seed, as on the CPU.
    export struct RandomOps
    {
        /// Fill a floating point tensor with values drawn from N(mean, stddev^2).
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource> && TensorDataTypeTraits<TDataType>::is_float_type
        static void fill_normal(
            Tensor<TDataType, TMemoryResource>& tensor,
            float mean,
            float stddev,
            IExecutionContext* exec_context = nullptr )
        {
            draw( tensor, std::normal_distribution<float>( mean, stddev ), exec_context );
        }

        /// Fill a floating point tensor with uniform values in [min_val, max_val).
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource> && TensorDataTypeTraits<TDataType>::is_float_type
        static void fill_uniform(
            Tensor<TDataType, TMemoryResource>& tensor,
            float min_val,
            float max_val,
            IExecutionContext* exec_context = nullptr )
        {
            draw( tensor, std::uniform_real_distribution<float>( min_val, max_val ), exec_context );
        }

    private:

        template<TensorDataType TDataType, typename TMemoryResource, typename TDistribution>
        static void draw( Tensor<TDataType, TMemoryResource>& tensor, TDistribution distribution, IExecutionContext* exec_context )
        {
            std::mt19937 generator = Core::RandomGenerator::getInstance().getGenerator();
            std::vector<float> values( static_cast<std::size_t>( tensor.size() ) );

            for ( float& value : values )
            {
                value = distribution( generator );
            }

            FillOps::fill( tensor, std::span<const float>( values ), exec_context );
        }
    };
}
