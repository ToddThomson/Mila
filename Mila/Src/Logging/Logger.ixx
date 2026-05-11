/**
 * @file Logger.ixx
 * @brief Abstract logging interface and static facade for the Mila logging infrastructure.
 *
 * Defines the Logger base class, the LogLevel enumeration, and the static
 * convenience methods that form the primary call-site API. Concrete sink
 * implementations (ConsoleSink, FileSink, NullSink) derive from Logger and
 * are registered via setDefaultLogger().
 *
 * Message formatting is the caller's responsibility. Use std::format at the
 * call site before passing the result to a log method:
 *
 * @code
 * auto message = std::format( "Loaded {} tensors from '{}'", count, path.string() );
 * Mila::Logging::Logger::info( message );
 * @endcode
 *
 * A default logger must be registered via setDefaultLogger() before any log
 * call is made. Failure to do so will throw std::runtime_error. The correct
 * place to register is Mila::initialize(). For test harnesses that want silent
 * logging, register a NullSink.
 */

module;
#include <source_location>
#include <stdexcept>
#include <string_view>

export module Logging.Logger;

namespace Mila::Logging
{
    /**
     * @brief Severity levels for log records, ordered from most verbose to most severe.
     *
     * A sink configured at a given level will emit all records at that level
     * and above. For example, a sink at Warning will emit Warning, Error, and
     * Critical records and discard Trace, Debug, and Info.
     */
    export enum class LogLevel
    {
        Trace,    ///< Very detailed diagnostic information, typically per-operation.
        Debug,    ///< Flow-level information useful during development.
        Info,     ///< Normal operational milestones (model loaded, server ready).
        Warning,  ///< Unexpected conditions that do not prevent continued operation.
        Error,    ///< Failures that may allow the application to continue.
        Critical  ///< Severe failures likely to cause process termination.
    };

    /**
     * @brief Abstract logging interface and static facade.
     *
     * Concrete sinks derive from this class and implement the virtual log_*
     * methods. The static convenience methods (info, warning, etc.) delegate
     * to whichever sink has been registered via setDefaultLogger().
     *
     * Calling any static log method before a sink has been registered via
     * setDefaultLogger() is a programming error and will throw std::runtime_error.
     * Register a NullSink in test harnesses that do not require log output.
     *
     * @note This class is not copyable or movable — sinks own resources
     *       (file handles, mutexes) that must not be duplicated.
     */
    export class Logger
    {
    public:
        virtual ~Logger() = default;

        Logger( const Logger& ) = delete;
        Logger& operator=( const Logger& ) = delete;
        Logger( Logger&& ) = delete;
        Logger& operator=( Logger&& ) = delete;

        // -------------------------------------------------------------------------
        // Default logger registration
        // -------------------------------------------------------------------------

        /**
         * @brief Registers a sink as the process-wide default logger.
         *
         * The caller retains ownership. The pointer must remain valid for the
         * lifetime of any subsequent log calls. Pass nullptr to deregister.
         *
         * @param logger Pointer to the sink to register, or nullptr to clear.
         */
        static void setDefaultLogger( Logger* logger )
        {
            defaultLogger_ = logger;
        }

        /**
         * @brief Returns a reference to the registered default logger.
         *
         * @throws std::runtime_error if no default logger has been registered.
         *         Call Mila::initialize() before making any log calls.
         *
         * @return Reference to the active default logger.
         */
        static Logger& defaultLogger()
        {
            if ( !defaultLogger_ )
            {
                throw std::runtime_error(
                    "Mila: log call before initializeLogger() — "
                    "call Mila::initialize() first." );
            }

            return *defaultLogger_;
        }

        // -------------------------------------------------------------------------
        // Static facade — primary call-site API
        // -------------------------------------------------------------------------

        /**
         * @brief Emits a record at Trace level via the default logger.
         * @param message  The pre-formatted log message.
         * @param location Automatically captured call-site location.
         */
        static void trace( std::string_view message,
            const std::source_location& location = std::source_location::current() )
        {
            defaultLogger().log_trace( message, location );
        }

        /**
         * @brief Emits a record at Debug level via the default logger.
         * @param message  The pre-formatted log message.
         * @param location Automatically captured call-site location.
         */
        static void debug( std::string_view message,
            const std::source_location& location = std::source_location::current() )
        {
            defaultLogger().log_debug( message, location );
        }

        /**
         * @brief Emits a record at Info level via the default logger.
         * @param message  The pre-formatted log message.
         * @param location Automatically captured call-site location.
         */
        static void info( std::string_view message,
            const std::source_location& location = std::source_location::current() )
        {
            defaultLogger().log_info( message, location );
        }

        /**
         * @brief Emits a record at Warning level via the default logger.
         * @param message  The pre-formatted log message.
         * @param location Automatically captured call-site location.
         */
        static void warning( std::string_view message,
            const std::source_location& location = std::source_location::current() )
        {
            defaultLogger().log_warning( message, location );
        }

        /**
         * @brief Emits a record at Error level via the default logger.
         * @param message  The pre-formatted log message.
         * @param location Automatically captured call-site location.
         */
        static void error( std::string_view message,
            const std::source_location& location = std::source_location::current() )
        {
            defaultLogger().log_error( message, location );
        }

        /**
         * @brief Emits a record at Critical level via the default logger.
         * @param message  The pre-formatted log message.
         * @param location Automatically captured call-site location.
         */
        static void critical( std::string_view message,
            const std::source_location& location = std::source_location::current() )
        {
            defaultLogger().log_critical( message, location );
        }

        // -------------------------------------------------------------------------
        // Virtual interface — implemented by concrete sinks
        // -------------------------------------------------------------------------

        /// @brief Emits a record at Trace level.
        virtual void log_trace( std::string_view message,
            const std::source_location& location = std::source_location::current() ) = 0;

        /// @brief Emits a record at Debug level.
        virtual void log_debug( std::string_view message,
            const std::source_location& location = std::source_location::current() ) = 0;

        /// @brief Emits a record at Info level.
        virtual void log_info( std::string_view message,
            const std::source_location& location = std::source_location::current() ) = 0;

        /// @brief Emits a record at Warning level.
        virtual void log_warning( std::string_view message,
            const std::source_location& location = std::source_location::current() ) = 0;

        /// @brief Emits a record at Error level.
        virtual void log_error( std::string_view message,
            const std::source_location& location = std::source_location::current() ) = 0;

        /// @brief Emits a record at Critical level.
        virtual void log_critical( std::string_view message,
            const std::source_location& location = std::source_location::current() ) = 0;

        /**
         * @brief Emits a record at an explicitly specified level.
         * @param message  The pre-formatted log message.
         * @param level    The severity level for this record.
         * @param location Automatically captured call-site location.
         */
        virtual void log( std::string_view message, LogLevel level,
            const std::source_location& location = std::source_location::current() ) = 0;

        /// @brief Sets the minimum level at which records are emitted.
        virtual void setLevel( LogLevel level ) = 0;

        /// @brief Returns the current minimum log level.
        virtual LogLevel getLevel() const = 0;

        /**
         * @brief Returns true if records at @p level would be emitted.
         * @param level The level to test.
         */
        virtual bool isEnabled( LogLevel level ) const = 0;

    protected:
        Logger() = default;

    private:
        inline static Logger* defaultLogger_{ nullptr };
    };
}
