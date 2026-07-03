module;
#include <iostream>
#include <string>
#include <source_location>
#include <format>
#include <chrono>
#include <mutex>
#include <ctime>
#include <iomanip>

/**
 * @file ConsoleSink.ixx
 * @brief Console-based logging sink for the Mila logging infrastructure.
 *
 * Provides a thread-safe logger that writes formatted log records to
 * std::cout (Trace through Warning) and std::cerr (Error and Critical).
 * Intended for interactive CLI use and development. For server deployments
 * use FileSink instead, as this sink writes to the process standard streams.
 */
export module Logging.ConsoleSink;

import Logging.Logger;

namespace Mila::Logging
{
    /**
     * @brief Thread-safe logging sink that writes formatted records to the console.
     *
     * Implements the Logger interface, routing records at Error level and above
     * to std::cerr and all others to std::cout. Each record is prefixed with
     * an optional timestamp and source location to aid diagnostics.
     *
     * Sink selection is an application-level concern. Wire this sink via
     * Mila::initializeLogger() for CLI or development scenarios.
     *
     * @code
     * // Typical wiring in main or Mila::initialize():
     * auto sink = std::make_shared<Mila::Logging::ConsoleSink>( Mila::Logging::LogLevel::Info );
     * Mila::Logging::Logger::setDefaultLogger( sink.get() );
     * @endcode
     */
    export class ConsoleSink : public Logger
    {
    public:
        /**
         * @brief Constructs a ConsoleSink with the specified minimum log level.
         * @param initialLevel Records below this level are silently discarded.
         */
        explicit ConsoleSink( LogLevel initialLevel = LogLevel::Info )
            : currentLevel_( initialLevel )
        {
        }

        // -------------------------------------------------------------------------
        // Logger interface -- control
        // -------------------------------------------------------------------------

        /// @brief Sets the minimum level at which records are emitted.
        void setLevel( LogLevel level ) override
        {
            currentLevel_ = level;
        }

        /// @brief Returns the current minimum log level.
        LogLevel getLevel() const override
        {
            return currentLevel_;
        }

        /**
         * @brief Returns true if records at @p level would be emitted.
         * @param level The level to test.
         */
        bool isEnabled( LogLevel level ) const override
        {
            return level >= currentLevel_;
        }

        // -------------------------------------------------------------------------
        // Logger interface -- configuration
        // -------------------------------------------------------------------------

        /**
         * @brief Controls whether a timestamp prefix is prepended to each record.
         * @param include Pass false to suppress timestamps (e.g. in unit tests).
         */
        void setIncludeTimestamp( bool include )
        {
            includeTimestamp_ = include;
        }

        /**
         * @brief Controls whether source location (file, line, function) is
         *        prepended to each record.
         * @param include Pass false to suppress location info.
         */
        void setIncludeSourceLocation( bool include )
        {
            includeSourceLocation_ = include;
        }

        // -------------------------------------------------------------------------
        // Logger interface -- emit methods
        // -------------------------------------------------------------------------

        /// @brief Emits a record at Trace level.
        void log_trace( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override
        {
            logImpl( message, LogLevel::Trace, location );
        }

        /// @brief Emits a record at Debug level.
        void log_debug( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override
        {
            logImpl( message, LogLevel::Debug, location );
        }

        /// @brief Emits a record at Info level.
        void log_info( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override
        {
            logImpl( message, LogLevel::Info, location );
        }

        /// @brief Emits a record at Warning level.
        void log_warning( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override
        {
            logImpl( message, LogLevel::Warning, location );
        }

        /// @brief Emits a record at Error level. Output goes to std::cerr.
        void log_error( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override
        {
            logImpl( message, LogLevel::Error, location );
        }

        /// @brief Emits a record at Critical level. Output goes to std::cerr.
        void log_critical( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override
        {
            logImpl( message, LogLevel::Critical, location );
        }

        /**
         * @brief Emits a record at an explicitly specified level.
         * @param message  The log message.
         * @param level    The severity level for this record.
         * @param location Automatically captured call-site location.
         */
        void log( std::string_view message, LogLevel level,
            const std::source_location& location = std::source_location::current() ) override
        {
            logImpl( message, level, location );
        }

    private:
        LogLevel currentLevel_ = LogLevel::Info;
        mutable std::mutex logMutex_;
        bool includeTimestamp_ = true;
        bool includeSourceLocation_ = true;

        /**
         * @brief Maps a LogLevel enumerator to its fixed-width string label.
         * @param level The level to convert.
         * @return A null-terminated string of consistent width for aligned output.
         */
        static constexpr const char* logLevelToString( LogLevel level )
        {
            switch ( level )
            {
                case LogLevel::Trace:    return "TRACE";
                case LogLevel::Debug:    return "DEBUG";
                case LogLevel::Info:     return "INFO ";
                case LogLevel::Warning:  return "WARN ";
                case LogLevel::Error:    return "ERROR";
                case LogLevel::Critical: return "CRIT ";
                default:                 return "UNKN ";
            }
        }

        /**
         * @brief Builds a timestamp string in HH:MM:SS.mmm format.
         * @return The formatted timestamp followed by a space, or an empty
         *         string if timestamps are disabled.
         */
        std::string getCurrentTimestamp() const
        {
            if ( !includeTimestamp_ ) return "";

            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t( now );
            auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;

            std::tm tm_buf{};

        #ifdef _MSC_VER
            localtime_s( &tm_buf, &time_t_now );
        #else
            localtime_r( &time_t_now, &tm_buf );
        #endif

            std::ostringstream oss;
            oss << std::put_time( &tm_buf, "%H:%M:%S" );
            oss << '.' << std::setfill( '0' ) << std::setw( 3 ) << now_ms.count() << ' ';
            return oss.str();
        }

        /**
         * @brief Builds a source location string in filename:line:function format.
         *
         * The file path is reduced to the filename component only. The function
         * name is trimmed to the bare name without namespace qualifiers or
         * parameter list, keeping the output concise even for templated functions.
         *
         * @param location The source location to format.
         * @return The formatted location string followed by ": ", or an empty
         *         string if source location is disabled.
         */
        std::string getLocationInfo( const std::source_location& location ) const
        {
            if ( !includeSourceLocation_ ) return "";

            std::string_view full_path( location.file_name() );
            auto last_slash = full_path.find_last_of( "/\\" );
            std::string_view filename = (last_slash == std::string_view::npos)
                ? full_path
                : full_path.substr( last_slash + 1 );

            // Trim to bare function name: strip namespace qualifiers and parameter list.
            std::string_view func_name( location.function_name() );
            auto last_colon = func_name.find_last_of( ':' );
            std::string_view qualified = (last_colon == std::string_view::npos)
                ? func_name
                : func_name.substr( last_colon + 1 );
            auto paren = qualified.find( '(' );
            std::string_view short_func = (paren == std::string_view::npos)
                ? qualified
                : qualified.substr( 0, paren );

            return std::format( "{}:{}:{}: ", filename, location.line(), short_func );
        }

        /**
         * @brief Core emit implementation called by all public log methods.
         *
         * Checks the level filter, formats the record, acquires the mutex, then
         * writes to the appropriate stream. Uses '\\n' rather than std::endl to
         * avoid a flush on every record; Critical records are explicitly flushed
         * to ensure visibility before a potential process termination.
         *
         * @param message  The log message.
         * @param level    The severity level of this record.
         * @param location The call-site source location.
         */
        void logImpl( std::string_view message, LogLevel level,
            const std::source_location& location )
        {
            if ( !isEnabled( level ) ) return;

            std::string timestamp = getCurrentTimestamp();
            std::string locationInfo = getLocationInfo( location );
            const char* levelStr = logLevelToString( level );

            std::ostream& out = (level >= LogLevel::Error) ? std::cerr : std::cout;

            std::lock_guard<std::mutex> lock( logMutex_ );
            out << timestamp << '[' << levelStr << "] " << locationInfo << message << '\n';

            if ( level >= LogLevel::Critical )
            {
                out.flush();
            }
        }
    };
}