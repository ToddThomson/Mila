/**
 * @file ComputeResource.ixx
 * @brief Base class for compute resources used in the Mila neural network framework.
 */

module;
#include <cstddef>

export module Compute.ComputeResource;

import Dnn.TensorDataType;

namespace Mila::Dnn::Compute
{
    export class ComputeResource {
    public:
        ComputeResource() = default;
        virtual ~ComputeResource() = default;

        // Type-erased fill function - derived classes implement this
        virtual void fillImpl( void* data, size_t element_count, const void* value_ptr, size_t element_size ) = 0;

        // Public template interface similar to std::fill
        template<typename T>
        void fill( T* data, size_t count, const T& value ) {
            // Pass the pointer to the value for type-erased implementation
            fillImpl( data, count, &value, sizeof( T ) );
        }

        // Device-specific memory copy operation
        virtual void memcpy( void* dst, const void* src, size_t size_bytes ) = 0;
    };
}