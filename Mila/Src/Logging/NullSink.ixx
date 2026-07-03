/**
 * @file NullSink.ixx
 * @brief No-op logging sink for the Mila logging infrastructure.
 *
 * Discards all log records unconditionally. Intended for test harnesses
 * and any context where log output is unwanted. Zero allocation and zero
 * stream I/O on every call path.
 */

module;
#include <iostream>
#include <string>
#include <source_location>

export module Logging.NullSink;

import Logging.Logger;

namespace Mila::Logging
{
    /**
     * @brief A logging sink that silently discards all records.
     *
     * All emit methods are no-ops. isEnabled() always returns false,
     * allowing callers that guard expensive message construction behind
     * an isEnabled() check to short-circuit at zero cost.
     *
     * @code
     * // Typical wiring in a test harness:
     * auto sink = std::make_shared<Mila::Logging::NullSink>();
     * Mila::Logging::Logger::setDefaultLogger( sink.get() );
     * @endcode
     */
    export class NullSink : public Logger
    {
    public:
        NullSink() = default;

        // -------------------------------------------------------------------------
        // Logger interface -- control
        // -------------------------------------------------------------------------

        /// @brief No-op. NullSink has no meaningful level.
        void setLevel( LogLevel ) override
        {
        }

        /// @brief Returns Critical, the highest level, ensuring isEnabled() is
        ///        always false for every real level passed by callers.
        LogLevel getLevel() const override
        {
            return LogLevel::Critical;
        }

        /// @brief Always returns false. No record passes the filter.
        bool isEnabled( LogLevel ) const override
        {
            return false;
        }

        // -------------------------------------------------------------------------
        // Logger interface -- emit methods (all no-ops)
        // -------------------------------------------------------------------------

        /// @brief Discards the record.
        void log_trace( std::string_view,
            const std::source_location & = std::source_location::current() ) override
        {
        }

        /// @brief Discards the record.
        void log_debug( std::string_view,
            const std::source_location & = std::source_location::current() ) override
        {
        }

        /// @brief Discards the record.
        void log_info( std::string_view,
            const std::source_location & = std::source_location::current() ) override
        {
        }

        /// @brief Discards the record.
        void log_warning( std::string_view,
            const std::source_location & = std::source_location::current() ) override
        {
        }

        /// @brief Discards the record.
        void log_error( std::string_view,
            const std::source_location & = std::source_location::current() ) override
        {
        }

        /// @brief Discards the record.
        void log_critical( std::string_view,
            const std::source_location & = std::source_location::current() ) override
        {
        }

        /// @brief Discards the record.
        void log( std::string_view, LogLevel,
            const std::source_location & = std::source_location::current() ) override
        {
        }
    };
}