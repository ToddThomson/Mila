/**
 * @file Assistant.SearchKeyStore.Windows.ixx
 * @brief Assistant.SearchKeyStore on Windows: the person's Windows Credential Manager.
 *
 * One of three candidate files for Assistant.SearchKeyStore, one per platform (AssistantInterface.md 8).
 */

module;

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <wincred.h>

export module Assistant.SearchKeyStore;

import std;

namespace Mila::Assistant
{
    /**
     * @brief The web search key, kept by the operating system's credential store for the person's account
     *        (MilaAssistant.md 3.7): a generic credential named by the store's name.
     *
     * The key is never read into a conversation, a log or the page, which learns only whether a key is set.
     */
    export class SearchKeyStore
    {
    public:

        static constexpr std::string_view kDefaultName = "Mila Assistant web search key";

        /// @param name What the credential is called; a test gives its own so the person's key is never touched.
        explicit SearchKeyStore( std::string name = std::string( kDefaultName ) )
            : name_( widened( name ) )
        {
        }

        /// The key, or nothing when none is kept.
        [[nodiscard]] std::optional<std::string> read() const
        {
            PCREDENTIALW credential = nullptr;

            if ( !CredReadW( name_.c_str(), CRED_TYPE_GENERIC, 0, &credential ) )
            {
                return std::nullopt;
            }

            std::string key( reinterpret_cast<const char*>( credential->CredentialBlob ), credential->CredentialBlobSize );
            CredFree( credential );

            if ( key.empty() )
            {
                return std::nullopt;
            }

            return key;
        }

        /// @throws std::runtime_error when the credential store refuses it.
        void write( const std::string& key ) const
        {
            CREDENTIALW credential{};
            credential.Type = CRED_TYPE_GENERIC;
            credential.TargetName = const_cast<LPWSTR>( name_.c_str() );
            credential.CredentialBlobSize = static_cast<DWORD>( key.size() );
            credential.CredentialBlob = reinterpret_cast<LPBYTE>( const_cast<char*>( key.data() ) );

            // Kept for the person on this computer, never roamed with their account.
            credential.Persist = CRED_PERSIST_LOCAL_MACHINE;

            if ( !CredWriteW( &credential, 0 ) )
            {
                throw std::runtime_error( std::format(
                    "the search key could not be saved in Windows Credential Manager (error {})", GetLastError() ) );
            }
        }

        void remove() const noexcept
        {
            (void)CredDeleteW( name_.c_str(), CRED_TYPE_GENERIC, 0 );
        }

    private:

        static std::wstring widened( const std::string& text )
        {
            if ( text.empty() )
            {
                return {};
            }

            const int length = MultiByteToWideChar( CP_UTF8, 0, text.data(), static_cast<int>( text.size() ), nullptr, 0 );
            std::wstring wide( static_cast<std::size_t>( length ), L'\0' );
            MultiByteToWideChar( CP_UTF8, 0, text.data(), static_cast<int>( text.size() ), wide.data(), length );

            return wide;
        }

        std::wstring name_;
    };
}
