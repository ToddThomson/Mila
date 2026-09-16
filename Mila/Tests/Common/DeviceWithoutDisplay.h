/**
 * @file DeviceWithoutDisplay.h
 * @brief Finds a visible CUDA device that drives no display, for measurements of device memory that
 * a desktop would otherwise move.
 */

#pragma once

#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <optional>

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

    /**
     * @brief The lowest CUDA ordinal whose adapter drives no display, or nullopt when every visible
     * device drives one or NVML cannot answer.
     *
     * On an adapter that drives a display, Windows lowers a process's video memory budget once the
     * process has written about 8 GiB, and CUDA's free memory is that budget less the process's usage
     * (Specifications/MemoryFootprint.md 11.5). A consumption measured there moves with whatever the
     * desktop holds on the card, so it cannot hold a tight bound.
     *
     * CUDA ordinals and NVML indices are ordered differently, so each device is matched by its PCI
     * address. NVML's display mode is deprecated on current drivers; display active is what is asked.
     */
    inline std::optional<int> findCudaDeviceWithoutDisplay()
    {
        if ( nvmlInit_v2() != NVML_SUCCESS )
            return std::nullopt;

        int count = 0;
        std::optional<int> found;

        if ( cudaGetDeviceCount( &count ) != cudaSuccess )
            count = 0;

        for ( int ordinal = 0; ordinal < count && !found; ++ordinal )
        {
            cudaDeviceProp properties{};

            if ( cudaGetDeviceProperties( &properties, ordinal ) != cudaSuccess )
                continue;

            char bus_id[ NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE ]{};
            std::snprintf( bus_id, sizeof( bus_id ), NVML_DEVICE_PCI_BUS_ID_FMT,
                static_cast<unsigned>( properties.pciDomainID ),
                static_cast<unsigned>( properties.pciBusID ),
                static_cast<unsigned>( properties.pciDeviceID ) );

            nvmlDevice_t device{};
            nvmlEnableState_t display_active = NVML_FEATURE_ENABLED;

            if ( nvmlDeviceGetHandleByPciBusId_v2( bus_id, &device ) == NVML_SUCCESS
                && nvmlDeviceGetDisplayActive( device, &display_active ) == NVML_SUCCESS
                && display_active == NVML_FEATURE_DISABLED )
            {
                found = ordinal;
            }
        }

        nvmlShutdown();

        return found;
    }
}
