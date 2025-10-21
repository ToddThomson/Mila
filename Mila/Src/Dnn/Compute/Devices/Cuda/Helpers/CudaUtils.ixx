/**
 * @file CudaUtils.ixx
 * @brief CUDA utility functions for memory operations across different CUDA memory resource types
 */

module;
#include <cuda_runtime.h>
#include <source_location>
#include <cstddef>

export module Cuda.Utils;

import Cuda.Error;

namespace Mila::Dnn::Compute::Cuda
{
    /**
     * @brief Common implementation of memory fill operation for CUDA memory types
     *
     * This function provides an optimized implementation for filling CUDA memory
     * with repeated patterns. It uses different strategies based on pattern size
     * and fill count:
     * - For zero patterns: Uses highly optimized cudaMemset
     * - For small counts: Direct copy of pattern to each location
     * - For large counts: Exponential growth strategy to minimize operations
     *
     * @param data Destination pointer (in CUDA memory)
     * @param count Number of elements to fill
     * @param value_ptr Pointer to the pattern value (in host memory)
     * @param value_size Size of each element in bytes
     * @param src_location Source location for error reporting (default: current location)
     *
     * @throws CudaError If any CUDA operation fails
     */
    //export void fillMemory(
    //    void* data,
    //    std::size_t count,
    //    const void* value_ptr,
    //    std::size_t value_size,
    //    const std::source_location& src_location = std::source_location::current() ) {
    //    if ( count == 0 || value_size == 0 ) {
    //        return;
    //    }

    //    // Check if the value is all zeros for optimization
    //    bool is_zero = true;
    //    const unsigned char* bytes = static_cast<const unsigned char*>(value_ptr);
    //    for ( size_t i = 0; i < value_size; i++ ) {
    //        if ( bytes[ i ] != 0 ) {
    //            is_zero = false;
    //            break;
    //        }
    //    }

    //    if ( is_zero ) {
    //        // Zero-fill optimization using cudaMemset
    //        cudaError_t status = cudaMemset( data, 0, count * value_size );
    //        cudaCheckStatus( status, src_location );
    //        return;
    //    }

    //    // For non-zero patterns, use a temporary buffer approach
    //    void* pattern_buffer = nullptr;
    //    cudaError_t status = cudaMalloc( &pattern_buffer, value_size );
    //    cudaCheckStatus( status, src_location );

    //    try {
    //        // Copy the pattern to device memory
    //        status = cudaMemcpy( pattern_buffer, value_ptr, value_size, cudaMemcpyHostToDevice );
    //        cudaCheckStatus( status, src_location );

    //        // For small counts, copy directly
    //        if ( count <= 16 ) {
    //            for ( size_t i = 0; i < count; i++ ) {
    //                void* dst = static_cast<char*>( data ) + (i * value_size);
    //                status = cudaMemcpy( dst, pattern_buffer, value_size, cudaMemcpyDeviceToDevice );
    //                cudaCheckStatus( status, src_location );
    //            }
    //        }
    //        else {
    //            // For larger counts, use an exponential filling strategy
    //            // First, copy the initial pattern
    //            status = cudaMemcpy( data, pattern_buffer, value_size, cudaMemcpyDeviceToDevice );
    //            cudaCheckStatus( status, src_location );

    //            // Double the filled region until we've covered the entire buffer
    //            size_t filled = 1;
    //            while ( filled < count ) {
    //                size_t copy_size = std::min( filled, count - filled ) * value_size;
    //                status = cudaMemcpy(
    //                    static_cast<char*>( data ) + (filled * value_size),
    //                    data,
    //                    copy_size,
    //                    cudaMemcpyDeviceToDevice
    //                );
    //                cudaCheckStatus( status, src_location );
    //                filled = std::min( filled * 2, count );
    //            }
    //        }
    //    }
    //    catch ( ... ) {
    //        // Clean up pattern buffer on error
    //        cudaFree( pattern_buffer );
    //        throw;
    //    }

    //    // Free the temporary pattern buffer
    //    status = cudaFree( pattern_buffer );
    //    cudaCheckStatus( status, src_location );
    //}
}