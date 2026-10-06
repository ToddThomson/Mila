/**
 * @file DecodeHarness.h
 * @brief Decode steps driven the way a model drives them: one persistent token tensor, and a count of called passes.
 *
 * A decode recording holds its input tensor's address, so a caller that makes a new token tensor every step is
 * recorded again every step and never replays; the models keep one, and so does a test. The count of called steps
 * is how a test proves a run replayed at all. Include after `import Mila;`.
 */

#pragma once

#include <cstdint>

namespace Mila::Tests::Common
{
    /**
     * @brief `TNetwork` with a count of the decode passes that ran called rather than replayed: decode steps,
     *        multi-token decodes and drafts.
     *
     * A replayed pass never enters its family's method, so a run of any length with replay on calls it exactly three
     * times for each recording -- the priming pass, the recorded one, and the self-check -- and more only when the
     * recording was discarded or turned off.
     */
    template<typename TNetwork>
    class CountedDecodeNetwork : public TNetwork
    {
    public:
        using TNetwork::TNetwork;

        [[nodiscard]] int calledDecodeSteps() const noexcept
        {
            return called_decode_steps_;
        }

        [[nodiscard]] int calledDecodeTokens() const noexcept
        {
            return called_decode_tokens_;
        }

        [[nodiscard]] int calledDrafts() const noexcept
        {
            return called_drafts_;
        }

    protected:
        typename TNetwork::TensorType& onDecode(
            const typename TNetwork::TokenIndexType& input, Mila::Dnn::dim_t position ) override
        {
            ++called_decode_steps_;

            return TNetwork::onDecode( input, position );
        }

        typename TNetwork::TensorType& onDecodeTokens(
            const typename TNetwork::TokenIndexType& input, Mila::Dnn::dim_t position ) override
        {
            ++called_decode_tokens_;

            return TNetwork::onDecodeTokens( input, position );
        }

        void onDraftTokens( typename TNetwork::TokenIndexType& tokens, Mila::Dnn::dim_t position ) override
        {
            ++called_drafts_;

            TNetwork::onDraftTokens( tokens, position );
        }

    private:
        int called_decode_steps_{ 0 };
        int called_decode_tokens_{ 0 };
        int called_drafts_{ 0 };
    };

    /// The [1, 1] device token tensor a network decodes from, written in place each step.
    template<typename TNetwork>
    class DecodeInput
    {
    public:
        explicit DecodeInput( const TNetwork& network )
            : device_( network.getDeviceId(), Mila::Dnn::shape_t{ 1, 1 } )
            , host_( Mila::Dnn::Compute::Device::Cpu(), Mila::Dnn::shape_t{ 1, 1 } )
        {
        }

        const typename TNetwork::TokenIndexType& set( std::int32_t token )
        {
            host_.data()[ 0 ] = token;
            Mila::Dnn::copy( host_, device_ );

            return device_;
        }

    private:
        typename TNetwork::TokenIndexType device_;
        Mila::Dnn::Tensor<Mila::Dnn::TensorDataType::INT32, Mila::Dnn::Compute::CpuMemoryResource> host_;
    };
}
