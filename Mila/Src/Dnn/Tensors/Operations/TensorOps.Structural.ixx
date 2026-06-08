/**
 * @file TensorOps.Structural.ixx
 * @brief Device-dispatched structural operations for tensors.
  */

module;
#include <type_traits>

export module Dnn.TensorOps:Structural;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorTypes;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps.Base;
import Compute.DeviceType;
import Compute.ExecutionContext;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void split( 
        const Tensor<TDataType, TMemoryResource>& input,
        Tensor<TDataType, TMemoryResource>& output_a,
        Tensor<TDataType, TMemoryResource>& output_b,
        IExecutionContext* exec_context = nullptr )
    {
        constexpr DeviceType device = TMemoryResource::device_type;
        TensorOps<device>::split( input, output_a, output_b, exec_context );
    }

    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void split(
        const Tensor<TDataType, TMemoryResource>& input,
        Tensor<TDataType, TMemoryResource>& output_a,
        Tensor<TDataType, TMemoryResource>& output_b,
        Tensor<TDataType, TMemoryResource>& output_c,
        IExecutionContext* exec_context = nullptr )
    {
        constexpr DeviceType device = TMemoryResource::device_type;
        TensorOps<device>::split( input, output_a, output_b, output_c, exec_context );
    }
}