/**
 * @file CpuTensorOps.Math.ixx
 * @brief CPU tensor mathematical operations partition
 */

module;
#include <memory>
#include <stdexcept>
#include <source_location>
#include <cmath>

export module Dnn.TensorOps:Math.Cpu;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorTraits;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;

namespace Mila::Dnn
{
    template<typename TComputeDeviceTag> struct TensorOps;

    export template<>
    struct TensorOps<Compute::CpuComputeDeviceTag>
    {
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static Tensor<TDataType, TMemoryResource> add( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
            if (a.shape() != b.shape()) {
                throw std::invalid_argument( "Tensor shapes must match for add operation" );
            }
            
            auto context = a.getDeviceContext();

            //if (!context || context != b.deviceContext()) {
            //    throw std::invalid_argument( "Tensors must have the same valid device context" );
            //}

            Tensor<TDataType, TMemoryResource> result( context, a.shape() );

            //Math::add<TDataType>( a.rawData(), b.rawData(), result.rawData(), a.size(), std::dynamic_pointer_cast<CudaDeviceContext>(context) );

            return result;
        }
    };
}