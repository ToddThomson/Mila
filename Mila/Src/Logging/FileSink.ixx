/**
 * @file FileSink.ixx
 * @brief File-based logging sink for the Mila logging infrastructure.
 *
 * Writes formatted log records to a file, truncating on open so each
 * process run starts with a clean log. Intended for server deployments
 * (e.g. the FastAPI inference server) where the process standard streams
 * are owned by the host and are not available for library output.
 *
 * @note Log file rollover (size-based rotation) is not yet implemented
 *       but the design intentionally isolates the open/write/flush
 *       responsibilities to make a future RolloverFileSink straightforward
 *       to derive or compose.
 */

module;
#include <fstream>
#include <string>
#include <string_view>
#include <source_location>
#include <format>
#include <chrono>
#include <mutex>
#include <ctime>
#include <iomanip>
#include <filesystem>
#include <stdexcept>

export module Logging.FileSink;

import Logging.Logger;

namespace Mila::Logging
{
    /**
     * @brief Thread-safe logging sink that writes formatted records to a file.
     *
     * The output file is opened with std::ios::trunc on construction so each
     * process run produces a clean log. If the file cannot be opened the
     * constructor throws std::runtime_error; the logger is never left in a
     * half-open state.
     *
     * Records below Error level are terminated with '\\n'. Critical records
     * receive an explicit flush to maximise visibility before a potential
     * process termination.
     *
     * @code
     * // Typical wiring for the FastAPI inference server:
     * auto sink = std::make_shared<Mila::Logging::FileSink>(
     *     "mila.log", Mila::Logging::LogLevel::Info );
     * Mila::Logging::Logger::setDefaultLogger( sink.get() );
     * @endcode
     */
    export class FileSink : public Logger
    {
    public:
        /**
         * @brief Opens @p path for writing and sets the minimum log level.
         *
         * The file is truncated on open. Parent directories must already exist.
         *
         * @param path         Path to the log file.
         * @param initialLevel Records below this level are silently discarded.
         * @throws std::runtime_error if the file cannot be opened.
         */
        explicit FileSink( const std::filesystem::path& path,
            LogLevel initialLevel = LogLevel::Info )
            : currentLevel_( initialLevel ), path_( path )
        {
            fileStream_.open( path_, std::ios::out | std::ios::trunc );

            if ( !fileStream_.is_open() )
            {
                throw std::runtime_error(
                    std::format( "FileSink: failed to open log file '{}'", path_.string() ) );
            }
        }

        /// @brief Flushes and closes the log file.
        ~FileSink() override
        {
            if ( fileStream_.is_open() )
            {
                fileStream_.flush();
                fileStream_.close();
            }
        }

        // FileSink owns a file handle — not copyable, not movable.
        FileSink( const FileSink& ) = delete;
        FileSink& operator=( const FileSink& ) = delete;
        FileSink( FileSink&& ) = delete;
        FileSink& operator=( FileSink&& ) = delete;

        // -------------------------------------------------------------------------
        // Logger interface — control
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
        // Logger interface — configuration
        // -------------------------------------------------------------------------

        /**
         * @brief Controls whether a timestamp prefix is prepended to each record.
         * @param include Pass false to suppress timestamps.
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

        /**
         * @brief Returns the path of the log file this sink is writing to.
         * @return The filesystem path passed at construction.
         */
        const std::filesystem::path& path() const
        {
            return path_;
        }

        // -------------------------------------------------------------------------
        // Logger interface — emit methods
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

        /// @brief Emits a record at Error level.
        void log_error( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override
        {
            logImpl( message, LogLevel::Error, location );
        }

        /// @brief Emits a record at Critical level. Explicitly flushes after write.
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
        std::filesystem::path path_;
        std::ofstream fileStream_;
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
         * parameter list.
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
         * writes to the file stream. Uses '\\n' for all levels; Critical records
         * receive an explicit flush to maximise visibility before a potential
         * process termination.
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

            std::lock_guard<std::mutex> lock( logMutex_ );
            fileStream_ << timestamp << '[' << levelStr << "] " << locationInfo << message << '\n';

            if ( level >= LogLevel::Critical )
            {
                fileStream_.flush();
            }
        }
    };
}