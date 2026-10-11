/**
 * @file Chat.ScopedLogSuppression.ixx
 * @brief Silences library logging for the duration of a prediction scan.
 */

export module Chat.ScopedLogSuppression;

import Mila;

namespace Mila::ChatApp
{
    /**
     * @brief Silences library logging for the duration of a prediction scan.
     *
     * Constructing a graph logs, and a scan constructs one per candidate: Gemma warns that it
     * cannot prefill efficiently at long context, which arrived 35 times at startup before this.
     * A prediction is not a deployment. What a probe learns is returned, never printed -- the same
     * contract predictFootprint already holds, extended to the library it calls into.
     *
     * Warnings from the LOAD are untouched, which is the point: the one at the context actually
     * chosen is a fact about this session, where the other thirty-four were about contexts nobody
     * asked for.
     *
     * Exported because it belongs to every caller that probes in bulk. The /model list ladder was
     * first written without it and leaked exactly the warning this was built to
     * suppress -- a Gemma prefill complaint about a 128K context nobody had asked to run at,
     * printed above the table it was probing for.
     */
    export class ScopedLogSuppression
    {
    public:
        ScopedLogSuppression()
            : restore_( Logging::Logger::defaultLogger().getLevel() )
        {
            Logging::Logger::defaultLogger().setLevel( Logging::LogLevel::Error );
        }

        ~ScopedLogSuppression()
        {
            Logging::Logger::defaultLogger().setLevel( restore_ );
        }

        ScopedLogSuppression( const ScopedLogSuppression& ) = delete;
        ScopedLogSuppression& operator=( const ScopedLogSuppression& ) = delete;

    private:
        Logging::LogLevel restore_;
    };
}
