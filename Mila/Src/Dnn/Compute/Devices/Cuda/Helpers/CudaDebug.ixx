module;
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>

export module Cuda.Debug;

import Dnn.TensorTypes;
import CublasLt.Error;
import Utils.Logger;

export namespace Mila::Dnn::Compute::Cuda
{
    /**
     * @brief Helper to dump a single 2D row-major matrix (host memory)
     *
     * Indexing: element (row=r, col=c) -> host_data[r * cols + c]
     */
    export template<typename T>
    void dump_2d_rowmajor_host(
        std::ostringstream& oss,
        const T* host_data,
        int rows,
        int cols,
        int max_display,
        int indent = 0 )
    {
        std::string indent_str( indent, ' ' );

        int rows_display = std::min( rows, max_display );
        int cols_display = std::min( cols, max_display );

        for ( int r = 0; r < rows_display; ++r )
        {
            oss << indent_str << "[ ";
            for ( int c = 0; c < cols_display; ++c )
            {
                T value = host_data[ r * cols + c ];
                oss << std::setw( 10 ) << static_cast<float>( value );
                if ( c < cols_display - 1 ) oss << " ";
            }
            if ( cols_display < cols )
            {
                oss << " ... (" << (cols - cols_display) << " more)";
            }
            oss << " ]\n";
        }

        if ( rows_display < rows )
        {
            oss << indent_str << "... (" << (rows - rows_display) << " more rows)\n";
        }
    }

    /**
     * @brief Debug utility to dump row-major tensor from device memory
     *
     * This utility copies data from device to host, then properly interprets
     * the row-major layout for display.
     *
     * @tparam T Data type (float, __half, etc.)
     * @param device_data Device pointer to row-major data
     * @param shape Shape vector (shape_t)
     * @param name Tensor name for display
     * @param max_display_size Maximum elements to display per dimension
     * @param stream CUDA stream for async copy (nullptr for default stream)
     * @return String representation of the tensor
     */
    export template<typename T = float>
    std::string dump_tensor(
        const T* device_data,
        const shape_t& shape,
        const std::string& name = "tensor",
        int max_display_size = 16,
        cudaStream_t stream = nullptr )
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision( 6 );

        // Validate input
        if ( shape.empty() || device_data == nullptr )
        {
            return name + ": <invalid tensor>\n";
        }

        // Calculate total size
        size_t total_size = 1;
        for ( auto dim : shape )
        {
            if ( dim <= 0 )
            {
                return name + ": <invalid shape>\n";
            }
            total_size *= static_cast<size_t>(dim);
        }

        // Allocate host memory
        std::vector<T> host_data( total_size );

        // Copy from device to host
        cudaError_t err;
        if ( stream != nullptr )
        {
            err = cudaMemcpyAsync( host_data.data(), device_data,
                total_size * sizeof( T ),
                cudaMemcpyDeviceToHost,
                stream );
            if ( err == cudaSuccess )
            {
                err = cudaStreamSynchronize( stream );
            }
        }
        else
        {
            err = cudaMemcpy( host_data.data(), device_data,
                total_size * sizeof( T ),
                cudaMemcpyDeviceToHost );
        }

        if ( err != cudaSuccess )
        {
            return name + ": <cudaMemcpy failed: " +
                std::string( cudaGetErrorString( err ) ) + ">\n";
        }

        // Header
        oss << "\n=== Row-Major Tensor: " << name << " ===\n";
        oss << "Shape: [";
        for ( size_t i = 0; i < shape.size(); ++i )
        {
            oss << shape[ i ];
            if ( i < shape.size() - 1 ) oss << ", ";
        }
        oss << "] (row-major for last 2 dims)\n";
        oss << "Total elements: " << total_size << "\n\n";

        const T* read_ptr = host_data.data();

        // Determine what to display based on dimensionality
        if ( shape.size() == 2 )
        {
            // Simple 2D matrix [rows, cols] in row-major
            int rows = static_cast<int>(shape[ 0 ]);
            int cols = static_cast<int>(shape[ 1 ]);
            dump_2d_rowmajor_host( oss, read_ptr, rows, cols, max_display_size, 0 );
        }
        else if ( shape.size() == 4 )
        {
            // Common case: [B, NH, rows, cols]
            int B = static_cast<int>(shape[ 0 ]);
            int NH = static_cast<int>(shape[ 1 ]);
            int rows = static_cast<int>(shape[ 2 ]);
            int cols = static_cast<int>(shape[ 3 ]);

            int b_display = std::min( B, max_display_size / 2 );
            int nh_display = std::min( NH, max_display_size / 2 );

            for ( int b = 0; b < b_display; ++b )
            {
                oss << "Batch " << b << ":\n";
                for ( int nh = 0; nh < nh_display; ++nh )
                {
                    oss << "  Head " << nh << ":\n";

                    // Pointer to this (b, nh) matrix (row-major block)
                    size_t matrix_offset = (static_cast<size_t>( b ) * NH + static_cast<size_t>( nh )) * rows * cols;
                    dump_2d_rowmajor_host( oss, read_ptr + matrix_offset, rows, cols, max_display_size, 10 );
                    oss << "\n";
                }
                if ( nh_display < NH )
                {
                    oss << "  ... (" << (NH - nh_display) << " more heads)\n";
                }
            }
            if ( b_display < B )
            {
                oss << "... (" << (B - b_display) << " more batches)\n";
            }
        }
        else if ( shape.size() == 3 )
        {
            // [B, rows, cols] case (row-major)
            int B = static_cast<int>( shape[ 0 ] );
            int rows = static_cast<int>( shape[ 1 ] );
            int cols = static_cast<int>( shape[ 2 ] );

            int b_display = std::min( B, max_display_size / 2 );

            for ( int b = 0; b < b_display; ++b )
            {
                oss << "Batch " << b << ":\n";
                size_t matrix_offset = static_cast<size_t>( b ) * rows * cols;
                dump_2d_rowmajor_host( oss, read_ptr + matrix_offset, rows, cols, max_display_size, 2 );
                oss << "\n";
            }
            if ( b_display < B )
            {
                oss << "... (" << (B - b_display) << " more batches)\n";
            }
        }
        else
        {
            oss << "Unsupported shape dimensionality (" << shape.size()
                << ") for row-major display\n";
            oss << "Supported: 2D [rows, cols], 3D [B, rows, cols], 4D [B, NH, rows, cols]\n";
        }

        return oss.str();
    }
}