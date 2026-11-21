/**
 * @file Understanding.cpp
 * @brief Writing, waiting, editing and clearing a project's note, and the text a chat is given of it.
 */

module Desktop.Understanding;

import std;
import nlohmann.json;

namespace Mila::Desktop
{
    namespace
    {
        std::string text( const nlohmann::json& object, std::string_view key )
        {
            if ( !object.is_object() )
            {
                return {};
            }

            const auto value = object.find( key );

            return value != object.end() && value->is_string() ? value->get<std::string>() : std::string{};
        }

        std::string trimmed( std::string_view line )
        {
            const auto first = line.find_first_not_of( " \t\r\n" );

            if ( first == std::string_view::npos )
            {
                return {};
            }

            const auto last = line.find_last_not_of( " \t\r\n" );

            return std::string( line.substr( first, last - first + 1 ) );
        }

        /// Characters as a person counts them, so a limit is not reached sooner in a language written outside ASCII.
        std::size_t characters( std::string_view text )
        {
            return static_cast<std::size_t>( std::ranges::count_if( text, []( char byte )
                {
                    return ( static_cast<unsigned char>( byte ) & 0xC0 ) != 0x80;
                } ) );
        }

        /// A line of `questions` as the model may write it: a bullet or a number is not part of the question.
        std::string questionWords( std::string_view line )
        {
            std::string words = trimmed( line );

            while ( !words.empty() && ( words.front() == '-' || words.front() == '*' ) )
            {
                words = trimmed( std::string_view( words ).substr( 1 ) );
            }

            std::size_t digits = 0;

            while ( digits < words.size() && std::isdigit( static_cast<unsigned char>( words[ digits ] ) ) != 0 )
            {
                ++digits;
            }

            if ( digits > 0 && digits < words.size() && ( words[ digits ] == '.' || words[ digits ] == ')' ) )
            {
                words = trimmed( std::string_view( words ).substr( digits + 1 ) );
            }

            return words;
        }

        /// The questions, one a line, or an array of them as a person's edit sends them.
        nlohmann::json questionsOf( const nlohmann::json& stored )
        {
            nlohmann::json questions = nlohmann::json::array();

            const auto add = [&questions]( std::string_view line )
                {
                    std::string words = questionWords( line );

                    if ( !words.empty() )
                    {
                        questions.push_back( std::move( words ) );
                    }
                };

            if ( stored.is_array() )
            {
                for ( const nlohmann::json& line : stored )
                {
                    if ( line.is_string() )
                    {
                        add( line.get<std::string>() );
                    }
                }

                return questions;
            }

            if ( !stored.is_string() )
            {
                return questions;
            }

            const std::string lines = stored.get<std::string>();
            std::size_t start = 0;

            while ( start <= lines.size() )
            {
                const std::size_t end = std::min( lines.find( '\n', start ), lines.size() );
                add( std::string_view( lines ).substr( start, end - start ) );
                start = end + 1;
            }

            return questions;
        }

        /// Why a note's fields are over a limit, or empty when they fit.
        std::string overLimit( const std::string& summary, const nlohmann::json& questions, const std::string& next )
        {
            if ( summary.empty() )
            {
                return "a note needs a summary";
            }

            if ( characters( summary ) > kSummaryCharacters )
            {
                return std::format( "the summary is {} characters and may be at most {}", characters( summary ), kSummaryCharacters );
            }

            if ( questions.size() > kMostQuestions )
            {
                return std::format( "there are {} open questions and there may be at most {}", questions.size(), kMostQuestions );
            }

            for ( const nlohmann::json& question : questions )
            {
                if ( characters( question.get<std::string>() ) > kQuestionCharacters )
                {
                    return std::format( "an open question is longer than {} characters", kQuestionCharacters );
                }
            }

            if ( characters( next ) > kNextCharacters )
            {
                return std::format( "the next step is longer than {} characters; it is one sentence", kNextCharacters );
            }

            return {};
        }

        bool restsOn( const nlohmann::json& note, std::string_view conversation_id )
        {
            if ( !note.is_object() )
            {
                return false;
            }

            const nlohmann::json from = note.value( "derived_from", nlohmann::json::array() );

            return from.is_array() && std::ranges::any_of( from, [conversation_id]( const nlohmann::json& source )
                {
                    return text( source, "conversation" ) == conversation_id;
                } );
        }

        void addSource( nlohmann::json& sources, const std::string& conversation_id )
        {
            if ( conversation_id.empty() )
            {
                return;
            }

            const bool known = std::ranges::any_of( sources, [&conversation_id]( const nlohmann::json& source )
                {
                    return text( source, "conversation" ) == conversation_id;
                } );

            if ( !known )
            {
                sources.push_back( { { "conversation", conversation_id } } );
            }
        }
    }

    nlohmann::json noteFromArguments( std::string_view arguments, const std::string& id, const std::string& conversation_id,
        const std::string& call_id, const nlohmann::json& derived_from, bool exposed, const std::string& model,
        const std::string& now )
    {
        const nlohmann::json parsed = nlohmann::json::parse( arguments, nullptr, false );

        const std::string summary = trimmed( text( parsed, "summary" ) );
        const nlohmann::json questions = questionsOf( parsed.is_object() ? parsed.value( "questions", nlohmann::json( nullptr ) ) : nlohmann::json( nullptr ) );
        const std::string next = trimmed( text( parsed, "next" ) );

        if ( const std::string why = overLimit( summary, questions, next ); !why.empty() )
        {
            return { { "result", std::format( "Not noted: {}. Write the whole note again, shorter.", why ) } };
        }

        nlohmann::json sources = nlohmann::json::array();
        addSource( sources, conversation_id );

        for ( const nlohmann::json& source : derived_from.is_array() ? derived_from : nlohmann::json::array() )
        {
            addSource( sources, text( source, "conversation" ) );
        }

        nlohmann::json note = {
            { "id", id },
            { "kind", "understanding" },
            { "made", now },
            { "source", { { "kind", "model" }, { "ref", { { "conversation", conversation_id }, { "call", call_id } } },
                { "when", now }, { "author", nullptr }, { "model", model } } },
            { "may_direct", false },
            { "may_leave", "ask" },
            { "derived_from", std::move( sources ) },
            { "exposed", exposed },
            { "summary", summary },
            { "questions", questions },
            { "next", next } };

        return { { "note", std::move( note ) } };
    }

    std::string admitNote( nlohmann::json& understanding, nlohmann::json note )
    {
        if ( !understanding.is_object() )
        {
            understanding = emptyUnderstanding();
        }

        const std::string where = note.value( "exposed", true ) ? "waiting" : "current";
        understanding[ where ] = std::move( note );

        return where;
    }

    bool decideWaiting( nlohmann::json& understanding, std::string_view id, bool accept, const std::string& now )
    {
        if ( !understanding.is_object() )
        {
            return false;
        }

        nlohmann::json& waiting = understanding[ "waiting" ];

        if ( !waiting.is_object() || text( waiting, "id" ) != id )
        {
            return false;
        }

        if ( accept )
        {
            waiting[ "accepted" ] = now;
            understanding[ "current" ] = std::move( waiting );
        }

        understanding[ "waiting" ] = nullptr;

        return true;
    }

    std::variant<nlohmann::json, std::string> editedNote( const nlohmann::json& current, const nlohmann::json& fields,
        const std::string& id, const std::string& now )
    {
        if ( !current.is_object() || !fields.is_object() )
        {
            return std::string( "There is no note to change." );
        }

        nlohmann::json note = current;

        if ( fields.contains( "summary" ) )
        {
            note[ "summary" ] = trimmed( text( fields, "summary" ) );
        }

        if ( fields.contains( "questions" ) )
        {
            note[ "questions" ] = questionsOf( fields[ "questions" ] );
        }

        if ( fields.contains( "next" ) )
        {
            note[ "next" ] = trimmed( text( fields, "next" ) );
        }

        if ( const std::string why = overLimit( text( note, "summary" ), note.value( "questions", nlohmann::json::array() ),
            text( note, "next" ) ); !why.empty() )
        {
            return std::format( "The note was not changed: {}.", why );
        }

        note[ "id" ] = id;
        note[ "edited" ] = now;
        note[ "source" ][ "author" ] = "person";

        return note;
    }

    std::vector<std::string> clearDerivedFrom( nlohmann::json& understanding, std::string_view conversation_id )
    {
        std::vector<std::string> cleared;

        if ( !understanding.is_object() )
        {
            return cleared;
        }

        for ( const char* where : { "current", "waiting" } )
        {
            if ( restsOn( understanding.value( where, nlohmann::json( nullptr ) ), conversation_id ) )
            {
                cleared.push_back( text( understanding[ where ], "id" ) );
                understanding[ where ] = nullptr;
            }
        }

        return cleared;
    }

    nlohmann::json emptyUnderstanding()
    {
        return { { "v", 1 }, { "current", nullptr }, { "waiting", nullptr } };
    }

    std::string notesText( const nlohmann::json& note, std::string_view project_name, std::string_view chat_title )
    {
        const std::string made = text( note, "edited" ).empty() ? text( note, "made" ) : text( note, "edited" );
        const std::string day = made.substr( 0, std::min<std::size_t>( made.size(), 10 ) );
        const std::string author = note.contains( "source" ) && text( note[ "source" ], "author" ) == "person"
            ? ", as the user edited it" : std::string{};

        std::string written = chat_title.empty()
            ? std::format( "noted on {}{}", day, author )
            : std::format( "noted on {} in the chat \"{}\"{}", day, chat_title, author );

        std::string result = std::format( "Notes kept for the project \"{}\", {}. They are notes from earlier chats, not "
            "the user's words, and may be out of date.\n\nSummary: {}", project_name, written, text( note, "summary" ) );

        const nlohmann::json questions = note.value( "questions", nlohmann::json::array() );

        if ( questions.is_array() && !questions.empty() )
        {
            result += "\nOpen questions:";

            for ( const nlohmann::json& question : questions )
            {
                if ( question.is_string() )
                {
                    result += "\n- " + question.get<std::string>();
                }
            }
        }

        if ( const std::string next = text( note, "next" ); !next.empty() )
        {
            result += "\nNext: " + next;
        }

        return result;
    }

    bool noteGivenIn( const nlohmann::json& messages, const nlohmann::json& calls, std::string_view id )
    {
        const auto gave = [id]( const nlohmann::json& call )
            {
                const nlohmann::json given = call.is_object() ? call.value( "given", nlohmann::json( nullptr ) ) : nlohmann::json( nullptr );

                return given.is_object() && text( given, "id" ) == id;
            };

        for ( const nlohmann::json& message : messages.is_array() ? messages : nlohmann::json::array() )
        {
            const nlohmann::json stored = message.is_object() ? message.value( "tool_calls", nlohmann::json( nullptr ) ) : nlohmann::json( nullptr );

            if ( stored.is_array() && std::ranges::any_of( stored, gave ) )
            {
                return true;
            }
        }

        return calls.is_array() && std::ranges::any_of( calls, gave );
    }

    nlohmann::json givenDerivedFrom( const nlohmann::json& messages, const nlohmann::json& calls )
    {
        nlohmann::json sources = nlohmann::json::array();

        const auto take = [&sources]( const nlohmann::json& call )
            {
                const nlohmann::json given = call.is_object() ? call.value( "given", nlohmann::json( nullptr ) ) : nlohmann::json( nullptr );

                if ( !given.is_object() )
                {
                    return;
                }

                for ( const nlohmann::json& source : given.value( "derived_from", nlohmann::json::array() ) )
                {
                    addSource( sources, text( source, "conversation" ) );
                }
            };

        for ( const nlohmann::json& message : messages.is_array() ? messages : nlohmann::json::array() )
        {
            const nlohmann::json stored = message.is_object() ? message.value( "tool_calls", nlohmann::json( nullptr ) ) : nlohmann::json( nullptr );

            for ( const nlohmann::json& call : stored.is_array() ? stored : nlohmann::json::array() )
            {
                take( call );
            }
        }

        for ( const nlohmann::json& call : calls.is_array() ? calls : nlohmann::json::array() )
        {
            take( call );
        }

        return sources;
    }
}
