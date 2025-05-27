module;
#include <source_location>
#include <stdexcept>
#include <format>
#include <string_view>

export module Utils.Logger;

import Dnn.Tensor;

namespace Mila::Utils
{
    export enum class LogLevel {
        Trace,    // Very detailed information, useful for debugging
        Debug,    // Detailed information on the flow through the system
        Info,     // Informational messages highlighting normal progress
        Warning,  // Potential issues that aren't errors
        Error,    // Error events that might allow the application to continue
        Critical  // Critical errors that may cause the application to terminate
    };

    export class Logger {
    private:
        inline static Logger* defaultLogger_{ nullptr };

    public:
        virtual ~Logger() = default;

        static void setDefaultLogger( Logger* logger ) {
            defaultLogger_ = logger;
        }

        static Logger& defaultLogger() {
            if ( !defaultLogger_ ) {
                throw std::runtime_error( "No default logger has been set" );
            }
            return *defaultLogger_;
        }

        // Static convenience methods for direct logging
        static void trace( std::string_view message,
            const std::source_location& location = std::source_location::current() ) {
            defaultLogger().log_trace( message, location );
        }

        static void debug( std::string_view message,
            const std::source_location& location = std::source_location::current() ) {
            defaultLogger().log_debug( message, location );
        }

        static void info( std::string_view message,
            const std::source_location& location = std::source_location::current() ) {
            defaultLogger().log_info( message, location );
        }

        static void warning( std::string_view message,
            const std::source_location& location = std::source_location::current() ) {
            defaultLogger().log_warning( message, location );
        }

        static void error( std::string_view message,
            const std::source_location& location = std::source_location::current() ) {
            defaultLogger().log_error( message, location );
        }

        static void critical( std::string_view message,
            const std::source_location& location = std::source_location::current() ) {
            defaultLogger().log_critical( message, location );
        }

        // Static format methods
        template<typename... Args>
        static void trace_fmt( std::format_string<Args...> fmt, Args&&... args ) {
            const std::source_location location = std::source_location::current();
            defaultLogger().trace_fmt( fmt, std::forward<Args>( args )... );
        }

        template<typename... Args>
        static void debug_fmt( std::format_string<Args...> fmt, Args&&... args ) {
            const std::source_location location = std::source_location::current();
            defaultLogger().debug_fmt( fmt, std::forward<Args>( args )... );
        }

        template<typename... Args>
        static void info_fmt( std::format_string<Args...> fmt, Args&&... args ) {
            const std::source_location location = std::source_location::current();
            defaultLogger().info_fmt( fmt, std::forward<Args>( args )... );
        }

        template<typename... Args>
        static void warning_fmt( std::format_string<Args...> fmt, Args&&... args ) {
            const std::source_location location = std::source_location::current();
            defaultLogger().warning_fmt( fmt, std::forward<Args>( args )... );
        }

        template<typename... Args>
        static void error_fmt( std::format_string<Args...> fmt, Args&&... args ) {
            const std::source_location location = std::source_location::current();
            defaultLogger().error_fmt( fmt, std::forward<Args>( args )... );
        }

        template<typename... Args>
        static void critical_fmt( std::format_string<Args...> fmt, Args&&... args ) {
            const std::source_location location = std::source_location::current();
            defaultLogger().critical_fmt( fmt, std::forward<Args>( args )... );
        }

        // Virtual logging methods with log_ prefix to avoid name conflicts with static methods
        virtual void log_trace( std::string_view message,
            const std::source_location& location = std::source_location::current() ) = 0;
        virtual void log_debug( std::string_view message,
            const std::source_location& location = std::source_location::current() ) = 0;
        virtual void log_info( std::string_view message,
            const std::source_location& location = std::source_location::current() ) = 0;
        virtual void log_warning( std::string_view message,
            const std::source_location& location = std::source_location::current() ) = 0;
        virtual void log_error( std::string_view message,
            const std::source_location& location = std::source_location::current() ) = 0;
        virtual void log_critical( std::string_view message,
            const std::source_location& location = std::source_location::current() ) = 0;

        // Generic log method
        virtual void log( std::string_view message, LogLevel level,
            const std::source_location& location = std::source_location::current() ) = 0;

        // Control methods
        virtual void setLevel( LogLevel level ) = 0;
        virtual LogLevel getLevel() const = 0;
        virtual bool isEnabled( LogLevel level ) const = 0;
    };
}