/**
 * @file Creations.cpp
 * @brief Reading what a chat made from its stored calls, and comparing a written file with the disk.
 */

module Desktop.Creations;

import std;
import nlohmann.json;

import Mila;

namespace Mila::Desktop
{
    namespace
    {
        constexpr std::size_t kMostReadsBefore = 20;

        // The session's own tools answer in Mila's words and read nothing; the file tools that write are the creations.
        constexpr std::array<std::string_view, 5> kNotReads = {
            "write_file", "edit_file", "propose_plan", "update_step", "request_folder" };

        std::filesystem::path pathFromUtf8( const std::string& text )
        {
            return std::filesystem::path( std::u8string( text.begin(), text.end() ) );
        }

        std::string utf8Of( const std::filesystem::path& path )
        {
            const std::u8string text = path.u8string();

            return std::string( text.begin(), text.end() );
        }

        std::optional<std::string> digestOf( const std::filesystem::path& file )
        {
            std::ifstream in( file, std::ios::binary );

            if ( !in )
            {
                return std::nullopt;
            }

            Mila::Distribution::Sha256 hash;
            std::vector<char> buffer( 64 * 1024 );

            while ( in.read( buffer.data(), static_cast<std::streamsize>( buffer.size() ) ) || in.gcount() > 0 )
            {
                hash.update( buffer.data(), static_cast<std::size_t>( in.gcount() ) );
            }

            if ( in.bad() )
            {
                return std::nullopt;
            }

            return hash.finish();
        }

        // libc++ counts file time in a 128-bit integer; nanoseconds since 1970 fit 64 bits until 2262.
        std::int64_t modifiedCount( const std::filesystem::file_time_type& time )
        {
            return static_cast<std::int64_t>( time.time_since_epoch().count() );
        }

        /// Well-formed UTF-8: each lead byte followed by its continuation bytes, no overlong form and no surrogate.
        bool isUtf8( std::string_view text )
        {
            std::size_t index = 0;

            while ( index < text.size() )
            {
                const unsigned char lead = static_cast<unsigned char>( text[ index ] );
                std::size_t length = 0;
                std::uint32_t point = 0;

                if ( lead < 0x80u )
                {
                    ++index;

                    continue;
                }

                if ( lead >= 0xC2u && lead <= 0xDFu )
                {
                    length = 2;
                    point = lead & 0x1Fu;
                }
                else if ( lead >= 0xE0u && lead <= 0xEFu )
                {
                    length = 3;
                    point = lead & 0x0Fu;
                }
                else if ( lead >= 0xF0u && lead <= 0xF4u )
                {
                    length = 4;
                    point = lead & 0x07u;
                }
                else
                {
                    return false;
                }

                if ( index + length > text.size() )
                {
                    return false;
                }

                for ( std::size_t next = 1; next < length; ++next )
                {
                    const unsigned char continuation = static_cast<unsigned char>( text[ index + next ] );

                    if ( ( continuation & 0xC0u ) != 0x80u )
                    {
                        return false;
                    }

                    point = ( point << 6 ) | ( continuation & 0x3Fu );
                }

                const std::uint32_t least = length == 3 ? 0x800u : length == 4 ? 0x10000u : 0x80u;

                if ( point < least || point > 0x10FFFFu || ( point >= 0xD800u && point <= 0xDFFFu ) )
                {
                    return false;
                }

                index += length;
            }

            return true;
        }

        const nlohmann::json& messagesOf( const nlohmann::json& conversation )
        {
            static const nlohmann::json none = nlohmann::json::array();

            const auto found = conversation.find( "messages" );

            return found != conversation.end() && found->is_array() ? *found : none;
        }

        nlohmann::json orNull( const nlohmann::json& object, const char* key )
        {
            return object.is_object() ? object.value( key, nlohmann::json( nullptr ) ) : nlohmann::json( nullptr );
        }

        bool isRead( const nlohmann::json& call )
        {
            if ( !call.is_object() || !call.value( "result", nlohmann::json( nullptr ) ).is_string()
                || call.value( "is_error", false ) || call.value( "decision", nlohmann::json( nullptr ) ) == "deny" )
            {
                return false;
            }

            const std::string name = call.value( "name", std::string{} );

            return std::ranges::find( kNotReads, name ) == kNotReads.end();
        }

        nlohmann::json readEntry( const nlohmann::json& call )
        {
            const nlohmann::json summary = call.value( "summary", nlohmann::json( nullptr ) );

            return { { "tool", call.value( "name", std::string{} ) },
                { "what", summary.is_string() ? summary : call.value( "name", nlohmann::json( nullptr ) ) },
                { "time", orNull( call, "time" ) }, { "exposes", orNull( call, "exposes" ) } };
        }

        /// The fields every creation carries of the chat that made it.
        nlohmann::json chatFields( const nlohmann::json& conversation, std::string_view kind )
        {
            return { { "kind", kind }, { "conversation_id", orNull( conversation, "id" ) },
                { "title", orNull( conversation, "title" ) }, { "project", orNull( conversation, "project" ) } };
        }

        void addReads( nlohmann::json& creation, const std::vector<nlohmann::json>& reads )
        {
            const std::size_t first = reads.size() > kMostReadsBefore ? reads.size() - kMostReadsBefore : 0;
            nlohmann::json kept = nlohmann::json::array();
            bool exposed = false;

            for ( std::size_t index = 0; index < reads.size(); ++index )
            {
                exposed = exposed || reads[ index ].value( "exposes", nlohmann::json( false ) ) == true;

                if ( index >= first )
                {
                    kept.push_back( reads[ index ] );
                }
            }

            creation[ "read_before" ] = std::move( kept );
            creation[ "read_before_total" ] = reads.size();
            creation[ "exposed" ] = exposed;
        }

        template <typename Visit>
        void forEachCall( const nlohmann::json& message, Visit&& visit )
        {
            if ( !message.is_object() || !message.value( "tool_calls", nlohmann::json( nullptr ) ).is_array() )
            {
                return;
            }

            for ( const nlohmann::json& call : message[ "tool_calls" ] )
            {
                visit( call );
            }
        }
    }

    std::optional<std::string> writtenPath( const nlohmann::json& call )
    {
        if ( !call.is_object() )
        {
            return std::nullopt;
        }

        const std::string name = call.value( "name", std::string{} );
        const bool failed = call.value( "is_error", false );
        const bool ran = call.value( "result", nlohmann::json( nullptr ) ).is_string();

        if ( ( name != "write_file" && name != "edit_file" ) || failed || !ran )
        {
            return std::nullopt;
        }

        const nlohmann::json arguments_text = call.value( "arguments", nlohmann::json( nullptr ) );

        if ( !arguments_text.is_string() )
        {
            return std::nullopt;
        }

        const nlohmann::json arguments = nlohmann::json::parse( arguments_text.get<std::string>(), nullptr, false );

        if ( !arguments.is_object() || !arguments.value( "path", nlohmann::json( nullptr ) ).is_string() )
        {
            return std::nullopt;
        }

        return arguments[ "path" ].get<std::string>();
    }

    nlohmann::json fileRecord( const std::filesystem::path& file )
    {
        std::error_code error;
        const std::uintmax_t size = std::filesystem::file_size( file, error );

        if ( error )
        {
            return nullptr;
        }

        const std::filesystem::file_time_type modified = std::filesystem::last_write_time( file, error );

        if ( error )
        {
            return nullptr;
        }

        const std::optional<std::string> digest = digestOf( file );

        if ( !digest.has_value() )
        {
            return nullptr;
        }

        return { { "location", utf8Of( file ) }, { "sha256", *digest }, { "size", size },
            { "modified", modifiedCount( modified ) } };
    }

    nlohmann::json diskState( const nlohmann::json& written )
    {
        if ( !written.is_object() || !written.value( "location", nlohmann::json( nullptr ) ).is_string() )
        {
            return nullptr;
        }

        const std::filesystem::path file = pathFromUtf8( written[ "location" ].get<std::string>() );
        std::error_code error;

        if ( !std::filesystem::is_regular_file( file, error ) )
        {
            return "missing";
        }

        const std::uintmax_t size = std::filesystem::file_size( file, error );

        if ( error )
        {
            return nullptr;
        }

        if ( size != written.value( "size", std::uintmax_t{ 0 } ) )
        {
            return "changed";
        }

        const std::filesystem::file_time_type modified = std::filesystem::last_write_time( file, error );

        if ( error )
        {
            return nullptr;
        }

        if ( modifiedCount( modified ) == written.value( "modified", std::int64_t{ 0 } ) )
        {
            return "same";
        }

        // Touched, or rewritten to the same length: only its bytes say which.
        const std::optional<std::string> digest = digestOf( file );

        if ( !digest.has_value() )
        {
            return nullptr;
        }

        return *digest == written.value( "sha256", std::string{} ) ? "same" : "changed";
    }

    std::optional<nlohmann::json> fileCreation( const nlohmann::json& conversation, std::string_view path )
    {
        std::vector<nlohmann::json> reads;
        nlohmann::json writes = nlohmann::json::array();
        nlohmann::json location = nullptr;

        for ( const nlohmann::json& message : messagesOf( conversation ) )
        {
            forEachCall( message, [ & ]( const nlohmann::json& call )
                {
                    const std::optional<std::string> written_path = writtenPath( call );

                    if ( written_path.has_value() && *written_path == path )
                    {
                        const nlohmann::json written = orNull( call, "written" );

                        writes.push_back( { { "call_id", orNull( call, "call_id" ) }, { "tool", orNull( call, "name" ) },
                            { "time", orNull( call, "time" ) }, { "model", orNull( call, "model" ) },
                            { "step", orNull( call, "step" ) }, { "sha256", orNull( written, "sha256" ) } } );

                        if ( written.is_object() && written.value( "location", nlohmann::json( nullptr ) ).is_string() )
                        {
                            location = written[ "location" ];
                        }

                        return;
                    }

                    if ( writes.empty() && isRead( call ) )
                    {
                        reads.push_back( readEntry( call ) );
                    }
                } );
        }

        if ( writes.empty() )
        {
            return std::nullopt;
        }

        // A write stored before writes kept their file is found where the chat's folder is now.
        const nlohmann::json folder = orNull( conversation, "folder" );

        if ( location.is_null() && folder.is_string() )
        {
            location = utf8Of( pathFromUtf8( folder.get<std::string>() ) / pathFromUtf8( std::string( path ) ) );
        }

        nlohmann::json creation = chatFields( conversation, "file" );
        creation[ "path" ] = path;
        creation[ "location" ] = std::move( location );
        creation[ "made" ] = writes.front();
        creation[ "writes" ] = std::move( writes );
        addReads( creation, reads );

        return creation;
    }

    std::optional<nlohmann::json> replyCreation( const nlohmann::json& conversation, std::size_t index )
    {
        const nlohmann::json& messages = messagesOf( conversation );

        if ( index >= messages.size() )
        {
            return std::nullopt;
        }

        const nlohmann::json& reply = messages[ index ];

        if ( !reply.is_object() || reply.value( "role", std::string{} ) != "assistant"
            || !reply.value( "kept", nlohmann::json( nullptr ) ).is_string() )
        {
            return std::nullopt;
        }

        std::vector<nlohmann::json> reads;

        for ( std::size_t position = 0; position <= index; ++position )
        {
            forEachCall( messages[ position ], [ & ]( const nlohmann::json& call )
                {
                    if ( isRead( call ) )
                    {
                        reads.push_back( readEntry( call ) );
                    }
                } );
        }

        nlohmann::json creation = chatFields( conversation, "reply" );
        creation[ "message" ] = index;
        creation[ "kept" ] = reply[ "kept" ];
        creation[ "model" ] = orNull( reply, "model" );
        creation[ "text" ] = reply.value( "content", std::string{} );
        addReads( creation, reads );

        return creation;
    }

    nlohmann::json showableContent( const std::filesystem::path& file )
    {
        const nlohmann::json none = { { "content", nullptr }, { "content_kind", nullptr } };
        std::error_code error;

        if ( !std::filesystem::is_regular_file( file, error ) )
        {
            return none;
        }

        const std::uintmax_t size = std::filesystem::file_size( file, error );

        if ( error || size > kMostBytesShown )
        {
            return none;
        }

        std::ifstream in( file, std::ios::binary );
        std::string text( static_cast<std::size_t>( size ), '\0' );

        if ( !in.read( text.data(), static_cast<std::streamsize>( text.size() ) ) )
        {
            return none;
        }

        // A NUL marks a file that is not text; anything not UTF-8 would be replaced on the wire, and shown wrong.
        if ( text.find( '\0' ) != std::string::npos || !isUtf8( text ) )
        {
            return none;
        }

        std::string extension = utf8Of( file.extension() );
        std::ranges::transform( extension, extension.begin(), []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );

        return { { "content", text }, { "content_kind", extension == ".md" || extension == ".markdown" ? "markdown" : "text" } };
    }
}
