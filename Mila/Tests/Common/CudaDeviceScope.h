/**
 * @file CudaDeviceScope.h
 * @brief Makes a CUDA device current for a scope and restores the previous one on exit.
 */

#pragma once

#include <cuda_runtime.h>

namespace Mila::Tests::Common
{
    /**
     * @brief Makes a CUDA device current for a scope and restores the previous one on exit, on every
     * path out including a skip or a failed assertion.
     *
     * A test that changes the current device and leaves it changed moves every later test that reads
     * device memory without naming a device.
     */
    class ScopedCurrentCudaDevice
    {
    public:
        explicit ScopedCurrentCudaDevice( int ordinal )
        {
            if ( cudaGetDevice( &previous_ ) != cudaSuccess )
                previous_ = -1;

            selected_ = cudaSetDevice( ordinal ) == cudaSuccess;
        }

        ~ScopedCurrentCudaDevice()
        {
            if ( previous_ >= 0 )
                cudaSetDevice( previous_ );
        }

        ScopedCurrentCudaDevice( const ScopedCurrentCudaDevice& ) = delete;
        ScopedCurrentCudaDevice& operator=( const ScopedCurrentCudaDevice& ) = delete;

        [[nodiscard]] bool selected() const noexcept
        {
            return selected_;
        }

    private:
        int previous_{ -1 };
        bool selected_{ false };
    };
}
