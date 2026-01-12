module;
#include <string>
#include <iostream>
#include <sstream>
#include <format>
#include <iomanip>

export module Dnn.TensorHelpers;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorHostTypeMap;
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
     * If the concrete tensor is host-accessible the contents are printed directly (first `maxElements` values).
     * Otherwise a host copy (`Tensor<TPrecision, CpuMemoryResource>`) is created and the first `maxElements`
     * values are printed. This avoids flooding logs while giving a quick numeric snapshot.
     *
     * Notes:
     *  - Intended for short-lived debug logging in tests and components.
     *  - Callers should pick the template parameters that match the component context (e.g., `TensorType` alias).
     */
    export template<TensorDataType TPrecision, typename MemoryResource>
        void debugDumpTensor( const ITensor& t, const std::string& label, size_t maxElements = 8 )
    {
        using ConcreteTensor = Tensor<TPrecision, MemoryResource>;

        const ConcreteTensor* ct = dynamic_cast<const ConcreteTensor*>(&t);
        if ( !ct )
        {
            std::clog << label << ": (not a Tensor<" << int( TPrecision ) << "," << typeid(MemoryResource).name() << ">)\n";
            return;
        }

        // Determine how many elements to show (cap at tensor size)
        const size_t total = static_cast<size_t>( ct->size() );
        const size_t show = ( total < maxElements ) ? total : maxElements;

        if constexpr ( MemoryResource::is_host_accessible )
        {
            std::ostringstream oss;
            oss << label << " (first " << show << " of " << total << "): ";

            using HostType = typename TensorHostTypeMap<TPrecision>::host_type;
            const HostType* data = ct->data();

            for ( size_t i = 0; i < show; ++i )
            {
                if ( i ) oss << ", ";
                oss << std::fixed << std::setprecision( 6 ) << static_cast<double>( data[ i ] );
            }

            std::clog << oss.str() << std::endl;
        }
        else
        {
            // Non-host tensor: create host copy and print first elements
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_copy( Device::Cpu(), ct->shape() );
            host_copy.setName( std::format( "{}.host_copy", label ) );

            copy( *ct, host_copy );

            std::ostringstream oss;
            oss << label << " (host copy, first " << show << " of " << total << "): ";

            using HostType = typename TensorHostTypeMap<TPrecision>::host_type;
            const HostType* data = host_copy.data();

            for ( size_t i = 0; i < show; ++i )
            {
                if ( i ) oss << ", ";
                oss << std::fixed << std::setprecision( 6 ) << static_cast<double>( data[ i ] );
            }

            std::clog << oss.str() << std::endl;
        }

        return;
    }
}