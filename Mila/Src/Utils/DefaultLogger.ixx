module;
#include <iostream>
#include <string>
#include <source_location>
#include <format>
#include <chrono>
#include <mutex>
#include <ctime>
#include <iomanip>

export module Utils.DefaultLogger;

import Utils.Logger;

namespace Mila::Utils
{
    export class DefaultLogger : public Logger {
    private:
        LogLevel currentLevel_ = LogLevel::Info;
        mutable std::mutex logMutex_;
        bool includeTimestamp_ = true;
        bool includeSourceLocation_ = true;

        // Converts LogLevel enum to a string representation
        static constexpr const char* logLevelToString( LogLevel level ) {
            switch ( level ) {
                case LogLevel::Trace:    return "TRACE";
                case LogLevel::Debug:    return "DEBUG";
                case LogLevel::Info:     return "INFO ";
                case LogLevel::Warning:  return "WARN ";
                case LogLevel::Error:    return "ERROR";
                case LogLevel::Critical: return "CRIT ";
                default:                 return "UNKN ";
            }
        }

        // Gets the current timestamp as a formatted string
        std::string getCurrentTimestamp() const {
            if ( !includeTimestamp_ ) return "";

            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t( now );
            auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;

            std::ostringstream oss;
            std::tm tm_buf{};

        #ifdef _MSC_VER
            localtime_s( &tm_buf, &time_t_now );
        #else
            localtime_r( &time_t_now, &tm_buf );
        #endif

            // More concise timestamp format: HH:MM:SS.mmm
            oss << std::put_time( &tm_buf, "%H:%M:%S" );
            oss << '.' << std::setfill( '0' ) << std::setw( 3 ) << now_ms.count() << " ";
            return oss.str();
        }

        // Gets source location information as a string
        std::string getLocationInfo( const std::source_location& location ) const {
            if ( !includeSourceLocation_ ) return "";

            // Extract just the filename without path
            std::string_view full_path( location.file_name() );
            size_t last_slash = full_path.find_last_of( "/\\" );
            std::string_view filename = (last_slash == std::string_view::npos) ?
                full_path : full_path.substr( last_slash + 1 );

            // Extract just the function name without namespace/class prefixes if desired
            std::string_view func_name( location.function_name() );
            size_t last_colon = func_name.find_last_of( ":" );
            std::string_view short_func = (last_colon == std::string_view::npos) ?
                func_name : func_name.substr( last_colon + 1 );

            // Concise format: filename:line:function
            return std::format( "{}:{}:{}: ", filename, location.line(), short_func );
        }


        // Internal log implementation
        void logImpl( std::string_view message, LogLevel level, const std::source_location& location ) {
            if ( !isEnabled( level ) ) return;

            std::string timestamp = getCurrentTimestamp();
            std::string locationInfo = getLocationInfo( location );
            std::string levelStr = logLevelToString( level );

            // Lock to prevent interleaved output from multiple threads
            std::lock_guard<std::mutex> lock( logMutex_ );

            // Choose output stream based on level
            std::ostream& outStream = (level >= LogLevel::Error) ? std::cerr : std::cout;

            // Write the formatted message
            outStream << timestamp << "[" << levelStr << "] " << locationInfo << message << std::endl;
        }

    public:
        // Constructor allows setting the initial log level
        DefaultLogger( LogLevel initialLevel = LogLevel::Info )
            : currentLevel_( initialLevel ) {}

        // Control methods
        void setLevel( LogLevel level ) override {
            currentLevel_ = level;
        }

        LogLevel getLevel() const override {
            return currentLevel_;
        }

        bool isEnabled( LogLevel level ) const override {
            return level >= currentLevel_;
        }

        // Configure timestamp inclusion
        void setIncludeTimestamp( bool include ) {
            includeTimestamp_ = include;
        }

        // Configure source location inclusion
        void setIncludeSourceLocation( bool include ) {
            includeSourceLocation_ = include;
        }

        // Main logging methods - renamed to match the Logger.ixx interface
        void log_trace( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override {
            logImpl( message, LogLevel::Trace, location );
        }

        void log_debug( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override {
            logImpl( message, LogLevel::Debug, location );
        }

        void log_info( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override {
            logImpl( message, LogLevel::Info, location );
        }

        void log_warning( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override {
            logImpl( message, LogLevel::Warning, location );
        }

        void log_error( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override {
            logImpl( message, LogLevel::Error, location );
        }

        void log_critical( std::string_view message,
            const std::source_location& location = std::source_location::current() ) override {
            logImpl( message, LogLevel::Critical, location );
        }

        // Generic log method
        void log( std::string_view message, LogLevel level,
            const std::source_location& location = std::source_location::current() ) override {
            logImpl( message, level, location );
        }
    };
}