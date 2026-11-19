/**
 * @file Desktop.SearchKeyStore.Linux.ixx
 * @brief Desktop.SearchKeyStore on Linux: the desktop's Secret Service keyring, through libsecret.
 *
 * One of three candidate files for Desktop.SearchKeyStore, one per platform (AssistantInterface.md 8).
 */

module;

#include <libsecret/secret.h>

export module Desktop.SearchKeyStore;

import std;

namespace Mila::Desktop
{
    /**
     * @brief The web search key, kept by the operating system's credential store for the person's account
     *        (MilaDesktop.md 3.7): a password in the default keyring, found by the store's name.
     *
     * The key is never read into a conversation, a log or the page, which learns only whether a key is set. With no
     * keyring running, as on a computer with no desktop session, a key cannot be kept, and saving one says so.
     */
    export class SearchKeyStore
    {
    public:

        static constexpr std::string_view kDefaultName = "Mila Assistant web search key";

        /// @param name What the keyring item is called; a test gives its own so the person's key is never touched.
        explicit SearchKeyStore( std::string name = std::string( kDefaultName ) )
            : name_( std::move( name ) )
        {
        }

        /// The key, or nothing when none is kept or no keyring can be reached.
        [[nodiscard]] std::optional<std::string> read() const
        {
            GError* error = nullptr;
            gchar* found = secret_password_lookup_sync( schema(), nullptr, &error, "name", name_.c_str(), nullptr );

            if ( error != nullptr )
            {
                g_error_free( error );
            }

            if ( found == nullptr )
            {
                return std::nullopt;
            }

            std::string key( found );
            secret_password_free( found );

            if ( key.empty() )
            {
                return std::nullopt;
            }

            return key;
        }

        /// @throws std::runtime_error when the keyring refuses it, or none is running.
        void write( const std::string& key ) const
        {
            GError* error = nullptr;
            const gboolean stored = secret_password_store_sync( schema(), SECRET_COLLECTION_DEFAULT, name_.c_str(),
                key.c_str(), nullptr, &error, "name", name_.c_str(), nullptr );

            if ( !stored || error != nullptr )
            {
                const std::string why = error != nullptr ? error->message : "it refused";

                if ( error != nullptr )
                {
                    g_error_free( error );
                }

                throw std::runtime_error( std::format( "the search key could not be saved in the keyring: {}", why ) );
            }
        }

        void remove() const noexcept
        {
            GError* error = nullptr;
            (void)secret_password_clear_sync( schema(), nullptr, &error, "name", name_.c_str(), nullptr );

            if ( error != nullptr )
            {
                g_error_free( error );
            }
        }

    private:

        static const SecretSchema* schema() noexcept
        {
            static const SecretSchema kSchema = {
                "org.mila.Assistant.SearchKey", SECRET_SCHEMA_NONE,
                { { "name", SECRET_SCHEMA_ATTRIBUTE_STRING }, { nullptr, SecretSchemaAttributeType( 0 ) } } };

            return &kSchema;
        }

        std::string name_;
    };
}
