/**
 * @file ToolDesktop.ixx
 * @brief A desktop of Mila's own in the person's window station, which a contained program runs on so that it never
 *        sees the person's screen, windows or input. Windows only.
 */

module;

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <windows.h>
#include <sddl.h>

export module AI.ToolDesktop;

import std;

namespace Mila::AI
{
    // Of module linkage rather than internal, since the exported class's members use it.
    namespace ToolDesktopDetail
    {
        /// The calling process's user, as an SDDL SID string.
        std::wstring userSid()
        {
            HANDLE token = nullptr;

            if ( !OpenProcessToken( GetCurrentProcess(), TOKEN_QUERY, &token ) )
            {
                throw std::runtime_error( std::format( "the process's token could not be opened (error {})", GetLastError() ) );
            }

            DWORD length = 0;
            GetTokenInformation( token, TokenUser, nullptr, 0, &length );
            std::vector<std::byte> buffer( length );
            const BOOL read = GetTokenInformation( token, TokenUser, buffer.data(), length, &length );
            CloseHandle( token );

            if ( !read )
            {
                throw std::runtime_error( std::format( "the process's user could not be read (error {})", GetLastError() ) );
            }

            wchar_t* text = nullptr;

            if ( !ConvertSidToStringSidW( reinterpret_cast<TOKEN_USER*>( buffer.data() )->User.Sid, &text ) )
            {
                throw std::runtime_error( std::format( "the process's user could not be named (error {})", GetLastError() ) );
            }

            std::wstring sid( text );
            LocalFree( text );

            return sid;
        }
    }

    /**
     * @brief The desktop of ContainedPrograms.md 4.4, closed when this goes.
     *
     * Its DACL grants SYSTEM, the person, ALL APPLICATION PACKAGES and ALL RESTRICTED APPLICATION PACKAGES, and it is
     * labelled low integrity: what user32.dll needs to start in an AppContainer (microsoft/mxc#1465). A program
     * started from a thread that uses it lands on it.
     */
    export class ToolDesktop
    {
    public:

        /// @throws std::runtime_error when the desktop cannot be made.
        ToolDesktop()
        {
            static std::atomic<unsigned> made{ 0 };

            const std::wstring name = std::format( L"Mila-{}-{}", GetCurrentProcessId(), made++ );
            const std::wstring sddl = std::format( L"D:(A;;GA;;;SY)(A;;GA;;;{})(A;;GA;;;AC)(A;;GA;;;S-1-15-2-2)S:(ML;;NW;;;LW)", ToolDesktopDetail::userSid() );

            PSECURITY_DESCRIPTOR descriptor = nullptr;

            if ( !ConvertStringSecurityDescriptorToSecurityDescriptorW( sddl.c_str(), SDDL_REVISION_1, &descriptor, nullptr ) )
            {
                throw std::runtime_error( std::format( "the desktop's security could not be built (error {})", GetLastError() ) );
            }

            SECURITY_ATTRIBUTES attributes{ sizeof( attributes ), descriptor, FALSE };
            const HDESK desktop = CreateDesktopW( name.c_str(), nullptr, nullptr, 0, GENERIC_ALL, &attributes );
            const DWORD error = GetLastError();
            LocalFree( descriptor );

            if ( desktop == nullptr )
            {
                throw std::runtime_error( std::format( "a desktop for contained programs could not be made (error {})", error ) );
            }

            desktop_ = reinterpret_cast<std::uintptr_t>( desktop );
        }

        ToolDesktop( const ToolDesktop& ) = delete;
        ToolDesktop& operator=( const ToolDesktop& ) = delete;

        ~ToolDesktop()
        {
            CloseDesktop( reinterpret_cast<HDESK>( desktop_ ) );
        }

        /**
         * @brief Makes this the calling thread's desktop.
         *
         * Only a thread that owns no window and no hook can change its desktop, so a thread made for the run does.
         *
         * @throws std::runtime_error when the thread's desktop cannot be changed.
         */
        void useOnThisThread() const
        {
            if ( !SetThreadDesktop( reinterpret_cast<HDESK>( desktop_ ) ) )
            {
                throw std::runtime_error( std::format( "the run's thread could not use its desktop (error {})", GetLastError() ) );
            }
        }

    private:

        std::uintptr_t desktop_{ 0 };
    };
}
