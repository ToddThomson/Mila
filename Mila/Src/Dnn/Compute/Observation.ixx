/**
 * @file Observation.ixx
 * @brief Vocabulary for activation observation: the compute pass, the stage, the observer.
 *
 * Design of record is Specifications/Observability.md. These types are shared by
 * Component (which publishes) and IExecutionContext (which transports).
 */

module;
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <string_view>

export module Compute.Observation;

import Dnn.ITensor;

namespace Mila::Dnn::Compute
{
    /**
     * @brief The compute entry point that produced an observed value.
     *
     * Declared complete while only the inference passes are implemented, so that adding
     * Backward later is additive rather than a change to the observer signature.
     */
    export enum class ComputePass : uint8_t
    {
        Forward = 0,
        Prefill = 1,
        Decode = 2,
        Backward = 3
    };

    export constexpr std::string_view computePassToString( ComputePass pass ) noexcept
    {
        switch ( pass )
        {
            case ComputePass::Forward: return "forward";
            case ComputePass::Prefill: return "prefill";
            case ComputePass::Decode: return "decode";
            case ComputePass::Backward: return "backward";
        }

        return "unknown";
    }

    /**
     * @brief A set of compute passes, small enough to test on the publication path.
     *
     * Resolved once when an observer attaches, so a component tests a mask rather than
     * matching a path or a pass name on every call.
     */
    export class ComputePassMask
    {
    public:
        constexpr ComputePassMask() noexcept = default;

        constexpr ComputePassMask( std::initializer_list<ComputePass> passes ) noexcept
        {
            for ( ComputePass pass : passes )
            {
                add( pass );
            }
        }

        static constexpr ComputePassMask inference() noexcept
        {
            return { ComputePass::Forward, ComputePass::Prefill, ComputePass::Decode };
        }

        constexpr void add( ComputePass pass ) noexcept
        {
            bits_ |= toBit( pass );
        }

        constexpr bool contains( ComputePass pass ) const noexcept
        {
            return (bits_ & toBit( pass )) != 0;
        }

        constexpr bool empty() const noexcept
        {
            return bits_ == 0;
        }

        constexpr bool operator==( const ComputePassMask& ) const noexcept = default;

    private:
        static constexpr uint8_t toBit( ComputePass pass ) noexcept
        {
            return static_cast<uint8_t>( 1u << static_cast<uint8_t>( pass ) );
        }

        uint8_t bits_{ 0 };
    };

    /**
     * @brief A stage a component publishes, and the passes it is published on.
     *
     * The stage set differs by pass: an attention block's prefill produces score
     * intermediates its decode never materializes.
     */
    export struct ObservableStage
    {
        std::string name;
        ComputePassMask passes;
    };

    /**
     * @brief Receives one published activation.
     *
     * The tensor is BORROWED and is valid only for the duration of this call. It is also
     * ordered on the publishing component's stream rather than valid on the host:
     * publication never synchronizes, because synchronizing is the clearest way for a probe
     * to change what it observes. An observer needing host-readable values calls
     * IExecutionContext::synchronize() itself, or issues its own copy on that stream.
     */
    export using ActivationObserver = std::function<void(
        std::string_view path,
        ComputePass pass,
        std::string_view stage,
        const ITensor& value )>;
}
