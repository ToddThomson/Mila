module;
#include <string>
#include <iostream>
#include <sstream>
#include <format>

export module Dnn.TensorHelpers;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Compute.Device;
import Compute.CpuMemoryResource;
import Compute.MemoryResource;
import Compute.IExecutionContext;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Debug dump a concrete tensor to the log.
     *
     * Template parameters:
     *  - `TPrecision` : tensor element precision (dtype_t::FP32, FP16, ...)
     *  - `MemoryResource` : memory resource type used by the concrete Tensor (CpuMemoryResource, CudaDeviceMemoryResource, ...)
     *
     * The function attempts a `dynamic_cast` to the concrete `Tensor<TPrecision, MemoryResource>`.
     * If the concrete tensor is host-accessible the contents are printed directly via `toString(true)`.
     * Otherwise a host copy (`Tensor<TPrecision, CpuMemoryResource>`) is created and printed.
     *
     * Notes:
     *  - Intended for short-lived debug logging in tests and components.
     *  - Callers should pick the template parameters that match the component context (e.g., `TensorType` alias).
     */
    export template<TensorDataType TPrecision, typename MemoryResource>
        void debugDumpTensor( const ITensor& t, const std::string& label )
    {
        using ConcreteTensor = Tensor<TPrecision, MemoryResource>;

        const ConcreteTensor* ct = dynamic_cast<const ConcreteTensor*>(&t);
        if ( !ct )
        {
            std::clog << label << ": (not a Tensor<" << int( TPrecision ) << "," << typeid(MemoryResource).name() << ">)\n";
            return;
        }

        if constexpr ( MemoryResource::is_host_accessible )
        {
            std::clog << label << ":\n" << ct->toString( true ) << std::endl;
        }
        else
        {
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_copy( Device::Cpu(), ct->shape() );
            host_copy.setName( std::format( "{}.host_copy", label ) );

            copy( *ct, host_copy );

            std::clog << label << " (host copy):\n" << host_copy.toString( true ) << std::endl;
        }
    }
}