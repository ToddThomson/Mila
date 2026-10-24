/**
 * @file Worker.Windows.cpp
 * @brief Starts a worker on Windows: its own process, in a kill-on-close job, with only the handles and
 *        environment it is given.
 */

module;
#define NOMINMAX
#include <windows.h>

module ToolBroker.Worker;

import std;
import ToolBroker.LineChannel;
import ToolBroker.WorkerLimits;
import ToolBroker.WorkerProgram;

namespace Mila::ToolBroker
{
    namespace
    {
        [[noreturn]] void throwLastError( const char* what )
        {
            throw std::system_error( static_cast<int>( ::GetLastError() ), std::system_category(), what );
        }

        /** Closes a handle on every path out of `launchWorker` that does not hand it on. */
        class OwnedHandle
        {
        public:
            OwnedHandle() = default;

            explicit OwnedHandle( HANDLE handle ) noexcept
                : handle_( handle )
            {
            }

            ~OwnedHandle()
            {
                if ( handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE )
                {
                    ::CloseHandle( handle_ );
                }
            }

            OwnedHandle( const OwnedHandle& ) = delete;
            OwnedHandle& operator=( const OwnedHandle& ) = delete;

            HANDLE get() const noexcept
            {
                return handle_;
            }

            HANDLE* out() noexcept
            {
                return &handle_;
            }

            HANDLE release() noexcept
            {
                return std::exchange( handle_, nullptr );
            }

        private:
            HANDLE handle_ = nullptr;
        };

        /** Owns an initialized attribute list, which must be deleted before its storage is freed. */
        class AttributeList
        {
        public:
            explicit AttributeList( DWORD count )
            {
                SIZE_T size = 0;
                ::InitializeProcThreadAttributeList( nullptr, count, 0, &size );
                storage_.resize( size );

                if ( !::InitializeProcThreadAttributeList( get(), count, 0, &size ) )
                {
                    throwLastError( "Preparing a worker's attributes failed" );
                }
            }

            ~AttributeList()
            {
                ::DeleteProcThreadAttributeList( get() );
            }

            AttributeList( const AttributeList& ) = delete;
            AttributeList& operator=( const AttributeList& ) = delete;

            LPPROC_THREAD_ATTRIBUTE_LIST get() noexcept
            {
                return reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>( storage_.data() );
            }

        private:
            std::vector<std::byte> storage_;
        };

        /**
         * Quotes one argument so the C runtime's parsing gives it back unchanged: backslashes are literal except
         * where they run into a quote, so those runs are doubled.
         */
        std::wstring quoteArgument( const std::wstring& argument )
        {
            if ( !argument.empty() && argument.find_first_of( L" \t\n\v\"" ) == std::wstring::npos )
            {
                return argument;
            }

            std::wstring quoted = L"\"";
            auto position = argument.begin();

            while ( true )
            {
                std::size_t backslashes = 0;

                while ( position != argument.end() && *position == L'\\' )
                {
                    ++position;
                    ++backslashes;
                }

                if ( position == argument.end() )
                {
                    quoted.append( backslashes * 2, L'\\' );
                    break;
                }

                if ( *position == L'"' )
                {
                    quoted.append( backslashes * 2 + 1, L'\\' );
                } else {
                    quoted.append( backslashes, L'\\' );
                }

                quoted.push_back( *position );
                ++position;
            }

            quoted.push_back( L'"' );

            return quoted;
        }

        std::wstring commandLine( const WorkerProgram& program )
        {
            std::wstring line = quoteArgument( program.executable.wstring() );

            for ( const auto& argument : program.arguments )
            {
                line.push_back( L' ' );
                line.append( quoteArgument( argument ) );
            }

            return line;
        }

        /** The block CreateProcess takes: `name=value` entries, each ended by a null, then one more null. */
        std::wstring environmentBlock( const WorkerLimits& limits )
        {
            auto variables = limits.environment;

            for ( const auto& [name, value] : variables )
            {
                if ( name.empty() || name.find_first_of( std::wstring_view( L"=\0", 2 ) ) != std::wstring::npos
                    || value.find( L'\0' ) != std::wstring::npos )
                {
                    throw std::invalid_argument( "A worker's environment variable has an invalid name or value" );
                }
            }

            // Windows keeps its own block sorted by name, ignoring case.
            std::ranges::sort( variables, []( const auto& left, const auto& right )
                {
                    return ::CompareStringOrdinal( left.first.c_str(), -1, right.first.c_str(), -1, TRUE ) == CSTR_LESS_THAN;
                } );

            std::wstring block;

            for ( const auto& [name, value] : variables )
            {
                block.append( name );
                block.push_back( L'=' );
                block.append( value );
                block.push_back( L'\0' );
            }

            block.push_back( L'\0' );

            if ( variables.empty() )
            {
                block.push_back( L'\0' );
            }

            return block;
        }

        HANDLE createJob()
        {
            HANDLE job = ::CreateJobObjectW( nullptr, nullptr );

            if ( job == nullptr )
            {
                throwLastError( "Creating a worker's job failed" );
            }

            // No breakaway limit is set, so nothing the worker starts with CreateProcess can leave the job.
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION information{};
            information.BasicLimitInformation.LimitFlags =
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;

            if ( !::SetInformationJobObject( job, JobObjectExtendedLimitInformation, &information, sizeof( information ) ) )
            {
                ::CloseHandle( job );
                throwLastError( "Setting a worker's job limits failed" );
            }

            return job;
        }
    }

    Worker::Worker( void* process, void* job, std::uint32_t process_id, LineChannel channel )
        : process_( process ), job_( job ), process_id_( process_id ), channel_( std::move( channel ) )
    {
    }

    Worker::~Worker()
    {
        stop();
    }

    Worker::Worker( Worker&& other ) noexcept
        : process_( std::exchange( other.process_, nullptr ) ),
          job_( std::exchange( other.job_, nullptr ) ),
          process_id_( other.process_id_ ),
          channel_( std::move( other.channel_ ) )
    {
    }

    Worker& Worker::operator=( Worker&& other ) noexcept
    {
        if ( this != &other )
        {
            stop();
            process_ = std::exchange( other.process_, nullptr );
            job_ = std::exchange( other.job_, nullptr );
            process_id_ = other.process_id_;
            channel_ = std::move( other.channel_ );
        }

        return *this;
    }

    LineChannel& Worker::channel() noexcept
    {
        return channel_;
    }

    std::uint32_t Worker::processId() const noexcept
    {
        return process_id_;
    }

    bool Worker::isRunning() const
    {
        return process_ != nullptr && ::WaitForSingleObject( process_, 0 ) == WAIT_TIMEOUT;
    }

    void Worker::stop() noexcept
    {
        channel_.closeSending();

        if ( process_ != nullptr )
        {
            ::WaitForSingleObject( process_, static_cast<DWORD>( kExitGrace.count() ) );
        }

        // Closed even when the worker exited on its own, since what it started may still be running.
        if ( job_ != nullptr )
        {
            ::CloseHandle( job_ );
            job_ = nullptr;
        }

        if ( process_ != nullptr )
        {
            ::CloseHandle( process_ );
            process_ = nullptr;
        }
    }

    Worker launchWorker( const WorkerProgram& program, const WorkerLimits& limits )
    {
        auto environment = environmentBlock( limits );
        auto command = commandLine( program );
        const auto directory = limits.project_area.wstring();

        OwnedHandle job( createJob() );

        SECURITY_ATTRIBUTES inheritable{ sizeof( SECURITY_ATTRIBUTES ), nullptr, TRUE };
        OwnedHandle child_input;
        OwnedHandle input;
        OwnedHandle output;
        OwnedHandle child_output;

        if ( !::CreatePipe( child_input.out(), input.out(), &inheritable, 0 )
            || !::CreatePipe( output.out(), child_output.out(), &inheritable, 0 ) )
        {
            throwLastError( "Creating a worker's pipes failed" );
        }

        // The broker's ends stay out of the worker; only the handle list below is inherited in any case.
        ::SetHandleInformation( input.get(), HANDLE_FLAG_INHERIT, 0 );
        ::SetHandleInformation( output.get(), HANDLE_FLAG_INHERIT, 0 );

        OwnedHandle error_sink( ::CreateFileW( L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
            &inheritable, OPEN_EXISTING, 0, nullptr ) );

        if ( error_sink.get() == INVALID_HANDLE_VALUE )
        {
            throwLastError( "Opening a worker's error sink failed" );
        }

        std::array<HANDLE, 3> inherited{ child_input.get(), child_output.get(), error_sink.get() };
        HANDLE job_handle = job.get();

        AttributeList attributes( 2 );

        if ( !::UpdateProcThreadAttribute( attributes.get(), 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                inherited.data(), sizeof( HANDLE ) * inherited.size(), nullptr, nullptr )
            || !::UpdateProcThreadAttribute( attributes.get(), 0, PROC_THREAD_ATTRIBUTE_JOB_LIST,
                &job_handle, sizeof( HANDLE ), nullptr, nullptr ) )
        {
            throwLastError( "Setting a worker's attributes failed" );
        }

        STARTUPINFOEXW startup{};
        startup.StartupInfo.cb = sizeof( startup );
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        startup.StartupInfo.hStdInput = child_input.get();
        startup.StartupInfo.hStdOutput = child_output.get();
        startup.StartupInfo.hStdError = error_sink.get();
        startup.lpAttributeList = attributes.get();

        PROCESS_INFORMATION process{};

        if ( !::CreateProcessW( nullptr, command.data(), nullptr, nullptr, TRUE,
                EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
                environment.data(), directory.empty() ? nullptr : directory.c_str(),
                &startup.StartupInfo, &process ) )
        {
            throwLastError( "Starting a worker failed" );
        }

        ::CloseHandle( process.hThread );

        return Worker( process.hProcess, job.release(), process.dwProcessId,
            LineChannel( output.release(), input.release() ) );
    }
}
