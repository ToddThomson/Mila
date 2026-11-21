/**
 * @file Assistant.SearchKeyStore.Apple.ixx
 * @brief Assistant.SearchKeyStore on macOS: the person's login keychain.
 *
 * One of three candidate files for Assistant.SearchKeyStore, one per platform (AssistantInterface.md 8).
 */

module;

#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>

export module Assistant.SearchKeyStore;

import std;

namespace Mila::Assistant
{
    /**
     * @brief The web search key, kept by the operating system's credential store for the person's account
     *        (MilaAssistant.md 3.7): a generic password whose service is the store's name.
     *
     * The key is never read into a conversation, a log or the page, which learns only whether a key is set.
     */
    export class SearchKeyStore
    {
    public:

        static constexpr std::string_view kDefaultName = "Mila Assistant web search key";

        /// @param name What the keychain item is called; a test gives its own so the person's key is never touched.
        explicit SearchKeyStore( std::string name = std::string( kDefaultName ) )
            : name_( std::move( name ) )
        {
        }

        /// The key, or nothing when none is kept.
        [[nodiscard]] std::optional<std::string> read() const
        {
            const Owned<CFMutableDictionaryRef> query = itemQuery();
            CFDictionarySetValue( query.get(), kSecReturnData, kCFBooleanTrue );
            CFDictionarySetValue( query.get(), kSecMatchLimit, kSecMatchLimitOne );

            CFTypeRef found = nullptr;

            if ( SecItemCopyMatching( query.get(), &found ) != errSecSuccess || found == nullptr )
            {
                return std::nullopt;
            }

            const Owned<CFTypeRef> data( found );
            const auto bytes = static_cast<CFDataRef>( data.get() );
            std::string key( reinterpret_cast<const char*>( CFDataGetBytePtr( bytes ) ), static_cast<std::size_t>( CFDataGetLength( bytes ) ) );

            if ( key.empty() )
            {
                return std::nullopt;
            }

            return key;
        }

        /// @throws std::runtime_error when the keychain refuses it.
        void write( const std::string& key ) const
        {
            remove();

            const Owned<CFMutableDictionaryRef> item = itemQuery();
            const Owned<CFDataRef> data( CFDataCreate( nullptr, reinterpret_cast<const UInt8*>( key.data() ), static_cast<CFIndex>( key.size() ) ) );
            CFDictionarySetValue( item.get(), kSecValueData, data.get() );

            const OSStatus status = SecItemAdd( item.get(), nullptr );

            if ( status != errSecSuccess )
            {
                throw std::runtime_error( std::format( "the search key could not be saved in the keychain (error {})", status ) );
            }
        }

        void remove() const noexcept
        {
            const Owned<CFMutableDictionaryRef> query = itemQuery();
            (void)SecItemDelete( query.get() );
        }

    private:

        /// A Core Foundation object this code created, released when it goes.
        template <typename TReference>
        class Owned
        {
        public:

            explicit Owned( TReference reference ) noexcept : reference_( reference ) {}
            Owned( Owned&& other ) noexcept : reference_( std::exchange( other.reference_, nullptr ) ) {}
            Owned( const Owned& ) = delete;
            Owned& operator=( const Owned& ) = delete;
            Owned& operator=( Owned&& ) = delete;

            ~Owned()
            {
                if ( reference_ != nullptr )
                {
                    CFRelease( reference_ );
                }
            }

            [[nodiscard]] TReference get() const noexcept { return reference_; }

        private:

            TReference reference_;
        };

        /// The item's class, service and account: what names it, without its value.
        Owned<CFMutableDictionaryRef> itemQuery() const
        {
            Owned<CFMutableDictionaryRef> query( CFDictionaryCreateMutable( nullptr, 0,
                &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks ) );

            const Owned<CFStringRef> service( CFStringCreateWithCString( nullptr, name_.c_str(), kCFStringEncodingUTF8 ) );

            CFDictionarySetValue( query.get(), kSecClass, kSecClassGenericPassword );
            CFDictionarySetValue( query.get(), kSecAttrService, service.get() );
            CFDictionarySetValue( query.get(), kSecAttrAccount, CFSTR( "web search" ) );

            return query;
        }

        std::string name_;
    };
}
