/**
 * @file Component.MemoryStats.ixx
 * @brief Memory allocation statistics for component inspection, and the vocabulary for
 *        describing what a deployment would cost before it is built.
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

import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;

namespace Mila::Dnn
{
    /**
     * @brief Storage bytes occupied by an element count of a given tensor data type.
     *
     * The counterpart to Tensor::getStorageSize() for buffers that do not exist yet --
     * getRequiredMemory() sizes allocations before build() makes them. Sub-byte types
     * are the reason this cannot be a multiply: FP4 packs two elements per byte, so
     * element_count * size_in_bytes overstates a packed weight by exactly 2x.
     *
     * @param element_count Logical elements, not bytes.
     */
    export template<TensorDataType TDataType>
    constexpr std::size_t storageBytes( dim_t element_count ) noexcept
    {
        constexpr std::size_t bits = []() constexpr {
            if constexpr ( requires { TensorDataTypeTraits<TDataType>::bits_per_element; } )
                return TensorDataTypeTraits<TDataType>::bits_per_element;
            else
                return TensorDataTypeTraits<TDataType>::size_in_bytes * std::size_t( 8 );
            }();

        return ( static_cast<std::size_t>( element_count ) * bits + 7 ) / 8;
    }

    /**
     * @brief Memory allocation breakdown for a single component.
     *
     * Reflects the current allocation state at the moment of the call.
     * Categories map directly onto the component build lifecycle:
     *
     *   After construction        nothing -- construction allocates no device memory
     *   After build()             parameters + state
     *   After setTrainingMode()   parameters + state + gradients
     *
     * Parameters are allocated in onBuilding(), not in the constructor. That is what
     * lets Component::getRequiredMemory() report a footprint from a constructed but
     * unbuilt graph. See Specifications/MemoryFootprint.md section 3.1.
     *
     * All sizes are in bytes. Device and host allocations are tracked
     * separately as they represent distinct, independently constrained
     * resources.
     */
    export struct MemoryStats
    {
        // REVIEW: Field names 'device_*' and 'host_*' are misleading. 'device_*' means
        // compute backend memory (GPU VRAM for CUDA, system RAM for CPU backend), while
        // 'host_*' is reserved for explicitly host-pinned allocations that accompany
        // device tensors. Consider renaming to 'compute_*' and 'pinned_*' or similar
        // to reflect actual intent.

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
            // REVIEW: A bit heavy. Keep for now.
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

    /**
     * @brief How a transformer would chunk a prefill pass at a given context length.
     *
     * Deliberately not a field of MemoryStats: that type is summed across a component tree,
     * and a chunk size is a single decision the transformer makes rather than a quantity
     * children contribute to.
     *
     * Both families resolve the chunk by walking a rung table downward and taking the largest
     * rung whose row cost fits an activation budget. The budget shrinks as context grows,
     * because the KV cache it shares VRAM with grows, so a longer context can silently buy a
     * smaller chunk. That is the fact this type exists to make askable.
     */
    export struct PrefillChunking
    {
        /// Rows per prefill chunk this context length would use.
        dim_t chunk_rows{ 0 };

        /// The largest rung this context length permits before the activation budget is
        /// applied. Equal to chunk_rows when the budget did not reduce the chunk.
        dim_t unconstrained_chunk_rows{ 0 };

        /// False when even the floor rung exceeds the budget, so chunk_rows is the floor used
        /// in spite of the budget rather than one that fits under it.
        bool fits_activation_budget{ true };

        /// True when the activation budget forced a smaller chunk than the context permits.
        [[nodiscard]] bool isBudgetConstrained() const noexcept
        {
            return chunk_rows < unconstrained_chunk_rows;
        }
    };

    /**
     * @brief What a deployment would cost, and how it would prefill.
     *
     * Fitting in memory and running well are different questions, and a caller choosing a
     * context length needs both: the largest context that fits can be one where the prefill
     * chunk has walked down to its floor. The two answers come from one graph construction
     * because they are two readings of the same arithmetic.
     */
    export struct DeploymentFootprint
    {
        MemoryStats memory;

        PrefillChunking prefill;
    };
}
