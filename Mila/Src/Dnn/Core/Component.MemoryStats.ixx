/**
 * @file ComponentMemoryStats.ixx
 * @brief Memory allocation statistics for component inspection.
 *
 * Provides a breakdown of GPU and host memory allocated by a component
 * across the three lifecycle-defined allocation categories: parameters,
 * state, and gradients.
 */

module;
#include <cstddef>
#include <string>
#include <format>

export module Dnn.Component:MemoryStats;

namespace Mila::Dnn
{
    /**
     * @brief Memory allocation breakdown for a single component.
     *
     * Reflects the current allocation state at the moment of the call.
     * Categories map directly onto the component build lifecycle:
     *
     *   After construction        parameters only
     *   After build()             parameters + state
     *   After setTraining(true)   parameters + state + gradients
     *
     * All sizes are in bytes. Device and host allocations are tracked
     * separately as they represent distinct, independently constrained
     * resources.
     */
    export struct MemoryStats
    {
        // ----------------------------------------------------------------
        // Device memory (GPU)
        // ----------------------------------------------------------------

        /// Learnable parameter buffers (weights, biases).
        /// Allocated at construction. Static for the component lifetime.
        std::size_t device_parameter_bytes{ 0 };

        /// Forward and decode output buffers, KV cache.
        /// Allocated at build(). Static after build().
        std::size_t device_state_bytes{ 0 };

        /// Input and parameter gradient buffers.
        /// Allocated lazily on first setTraining(true). Retained thereafter.
        std::size_t device_gradient_bytes{ 0 };

        // ----------------------------------------------------------------
        // Host memory (CPU)
        // ----------------------------------------------------------------

        /// Learnable parameter buffers pinned on host (if any).
        std::size_t host_parameter_bytes{ 0 };

        /// Forward and decode output buffers on host (if any).
        std::size_t host_state_bytes{ 0 };

        /// Gradient buffers on host (if any).
        std::size_t host_gradient_bytes{ 0 };

        // ----------------------------------------------------------------
        // Aggregates
        // ----------------------------------------------------------------

        /**
         * @brief Total device memory allocated by this component.
         */
        [[nodiscard]] std::size_t totalDeviceBytes() const noexcept
        {
            return device_parameter_bytes + device_state_bytes + device_gradient_bytes;
        }

        /**
         * @brief Total host memory allocated by this component.
         */
        [[nodiscard]] std::size_t totalHostBytes() const noexcept
        {
            return host_parameter_bytes + host_state_bytes + host_gradient_bytes;
        }

        /**
         * @brief Total memory allocated across all categories and locations.
         */
        [[nodiscard]] std::size_t totalBytes() const noexcept
        {
            return totalDeviceBytes() + totalHostBytes();
        }

        /**
         * @brief Accumulate another MemoryStats into this one.
         *
         * Used by CompositeComponent and Network to aggregate child stats.
         */
        MemoryStats& operator+=( const MemoryStats& rhs ) noexcept
        {
            device_parameter_bytes += rhs.device_parameter_bytes;
            device_state_bytes += rhs.device_state_bytes;
            device_gradient_bytes += rhs.device_gradient_bytes;
            host_parameter_bytes += rhs.host_parameter_bytes;
            host_state_bytes += rhs.host_state_bytes;
            host_gradient_bytes += rhs.host_gradient_bytes;
            
            return *this;
        }

        /**
         * @brief Produce a human-readable summary of this stats instance.
         *
         * @return Formatted multi-line string.
         */
        [[nodiscard]] std::string toString() const
        {
            auto fmt_bytes = []( std::size_t bytes ) -> std::string
                {
                    constexpr std::size_t KB = 1024;
                    constexpr std::size_t MB = 1024 * KB;
                    constexpr std::size_t GB = 1024 * MB;

                    if ( bytes >= GB )
                        return std::format( "{:.2f} GB", static_cast<double>(bytes) / GB );
                    if ( bytes >= MB )
                        return std::format( "{:.2f} MB", static_cast<double>(bytes) / MB );
                    if ( bytes >= KB )
                        return std::format( "{:.2f} KB", static_cast<double>(bytes) / KB );
                    return std::format( "{} B", bytes );
                };

            const std::string sep = "  +----------------------+---------------+---------------+\n";

            auto row = [&]( const std::string& label, std::size_t dev, std::size_t host ) -> std::string
                {
                    return std::format( "  | {:<20} | {:>13} | {:>13} |\n",
                        label, fmt_bytes( dev ), fmt_bytes( host ) );
                };

            return "Memory Statistics\n"
                + sep
                + std::format( "  | {:<20} | {:>13} | {:>13} |\n", "Category", "Device", "Host" )
                + sep
                + row( "Parameters", device_parameter_bytes, host_parameter_bytes )
                + row( "State", device_state_bytes, host_state_bytes )
                + row( "Gradients", device_gradient_bytes, host_gradient_bytes )
                + sep
                + row( "Total", totalDeviceBytes(), totalHostBytes() )
                + sep
                + std::format( "  Grand total: {}", fmt_bytes( totalBytes() ) );
        }
    };

    /**
     * @brief Aggregate two MemoryStats instances.
     */
    export [[nodiscard]] MemoryStats operator+( MemoryStats lhs, const MemoryStats& rhs ) noexcept
    {
        lhs += rhs;
        return lhs;
    }
}
