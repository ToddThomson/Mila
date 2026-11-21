/**
 * @file Assistant.RoundTimer.ixx
 * @brief A round as the page shows it: how long until its first token, and how fast the rest came.
 */

export module Assistant.RoundTimer;

import std;

namespace Mila::Assistant
{
    /**
     * @brief Times one round from the agent's events: the agent says that a token arrived, never how long it took.
     *
     * Timed here, in the application, as every measurement the person is shown is.
     */
    export class RoundTimer
    {
    public:

        void begin()
        {
            start_ = Clock::now();
            first_ = start_;
            last_ = start_;
            tokens_ = 0;
        }

        void token()
        {
            const auto now = Clock::now();

            if ( tokens_ == 0 )
            {
                first_ = now;
            }

            last_ = now;
            ++tokens_;
        }

        [[nodiscard]] int tokens() const noexcept
        {
            return tokens_;
        }

        /// From the round's start to its first token: the prompt's prefill, mostly.
        [[nodiscard]] double prefillMilliseconds() const noexcept
        {
            return std::chrono::duration<double, std::milli>( first_ - start_ ).count();
        }

        /// From the first token to the last; zero for a round of one token or none.
        [[nodiscard]] double decodeSeconds() const noexcept
        {
            return tokens_ > 1 ? std::chrono::duration<double>( last_ - first_ ).count() : 0.0;
        }

        /// Tokens after the first, per second of decoding; zero when there is nothing to divide.
        [[nodiscard]] double decodeRate() const noexcept
        {
            const double seconds = decodeSeconds();

            return seconds > 0.0 ? ( tokens_ - 1 ) / seconds : 0.0;
        }

    private:

        using Clock = std::chrono::steady_clock;

        Clock::time_point start_{};
        Clock::time_point first_{};
        Clock::time_point last_{};
        int tokens_{ 0 };
    };
}
