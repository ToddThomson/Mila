/**
 * @file TokenStreamer.ixx
 * @brief Token streaming abstractions for autoregressive generation.
 *
 * Provides TokenStreamer (per-token callback), TokenSink (batched span callback),
 * and BufferedTokenStreamer (batches tokens before forwarding to a TokenSink).
 */
module;
#include <span>
#include <cstdint>
#include <concepts>
#include <array>

export module Dnn.TokenStreamer;

namespace Mila::Dnn
{
    /**
     * @brief Satisfied by any callable accepting a single decoded token.
     *
     * Used as the callback type in LanguageModel::generate().
     */
    export template<class T>
        concept TokenStreamer = requires(T && t, int32_t tok)
    {
        t( tok );
    };

    /**
     * @brief Satisfied by any callable accepting a span of decoded tokens.
     *
     * Used as the sink type for BufferedTokenStreamer.
     */
    export template<class T>
        concept TokenSink = requires(T && t, std::span<const int32_t> s)
    {
        t( s );
    };

    /**
     * @brief Buffers BufSize tokens before forwarding a contiguous span to Sink.
     *
     * Reduces per-token overhead when Sink involves cross-thread posting or I/O.
     * Satisfies TokenStreamer via operator()(int32_t). Call finish() after
     * generation completes to flush any partial buffer and invoke Sink::on_finish()
     * if present.
     *
     * Sink is held by reference; the caller must ensure Sink outlives this object.
     *
     * @tparam BufSize Tokens to buffer before each flush. Defaults to 8.
     * @tparam Sink    A TokenSink callable.
     */
    export template<class Sink, size_t BufSize = 8>
        requires TokenSink<Sink>
    class BufferedTokenStreamer
    {
    public:
        explicit BufferedTokenStreamer( Sink& sink )
            : sink_( sink )
        {
        }

        void operator()( int32_t tok )
        {
            buf_[ count_++ ] = tok;

            if ( count_ == BufSize )
                flush();
        }

        void flush()
        {
            if ( count_ == 0 ) return;
            sink_( std::span<const int32_t>( buf_.data(), count_ ) );
            count_ = 0;
        }

        void finish()
        {
            flush();

            if constexpr ( requires(Sink & s)
            {
                s.on_finish();
            } )
                sink_.on_finish();
        }

    private:
        Sink& sink_;
        std::array<int32_t, BufSize> buf_{};
        size_t count_ = 0;
    };
}