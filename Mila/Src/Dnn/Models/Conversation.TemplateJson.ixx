/**
 * @file Conversation.TemplateJson.ixx
 * @brief JSON written the way a chat template's `tojson` filter writes it under transformers.
 */

export module Dnn.Models.Conversation.TemplateJson;

import std;

import nlohmann.json;

namespace Mila::Dnn::Conversation
{
    namespace detail
    {
        inline void appendJsonString( std::string& out, const std::string& text )
        {
            out += '"';

            for ( const char character : text )
            {
                switch ( character )
                {
                    case '"': out += "\\\""; break;
                    case '\\': out += "\\\\"; break;
                    case '\n': out += "\\n"; break;
                    case '\r': out += "\\r"; break;
                    case '\t': out += "\\t"; break;
                    case '\b': out += "\\b"; break;
                    case '\f': out += "\\f"; break;

                    default:
                        if ( static_cast<unsigned char>( character ) < 0x20 )
                        {
                            out += std::format( "\\u{:04x}", static_cast<unsigned char>( character ) );
                        } else {
                            out += character;
                        }
                }
            }

            out += '"';
        }

        inline void appendTemplateJson( std::string& out, const nlohmann::ordered_json& value, int indent, int depth )
        {
            const bool container = value.is_object() || value.is_array();

            if ( !container )
            {
                if ( value.is_string() )
                {
                    appendJsonString( out, value.get<std::string>() );
                } else {
                    out += value.dump();
                }

                return;
            }

            const bool object = value.is_object();

            if ( value.empty() )
            {
                out += object ? "{}" : "[]";

                return;
            }

            // json.dumps: ", " between items on one line, "," and a newline when indenting.
            const auto separate = [&]( bool first )
            {
                if ( indent < 0 )
                {
                    out += first ? "" : ", ";

                    return;
                }

                out += first ? "\n" : ",\n";
                out += std::string( static_cast<std::size_t>( indent * ( depth + 1 ) ), ' ' );
            };

            out += object ? '{' : '[';

            bool first = true;

            for ( auto entry = value.begin(); entry != value.end(); ++entry )
            {
                separate( first );
                first = false;

                if ( object )
                {
                    appendJsonString( out, entry.key() );
                    out += ": ";
                }

                appendTemplateJson( out, entry.value(), indent, depth + 1 );
            }

            if ( indent >= 0 )
            {
                out += "\n";
                out += std::string( static_cast<std::size_t>( indent * depth ), ' ' );
            }

            out += object ? '}' : ']';
        }
    }

    /**
     * @brief json.dumps as transformers' `tojson` filter calls it.
     *
     * Keys in the order given, non-ASCII written as itself, and json.dumps' separators: ", " and
     * ": " on one line, or "," and a newline with each level indented when `indent` is given. Not
     * nlohmann's dump, which is compact and -- for nlohmann::json -- sorted: a template writes tool
     * signatures and arguments through this filter, so the difference would be in every prompt
     * that offers a tool.
     *
     * @param value The value to write, its object keys in the order they were parsed.
     * @param indent Spaces per level, as `tojson(indent=4)` passes it; negative writes one line.
     */
    export inline std::string templateJson( const nlohmann::ordered_json& value, int indent = -1 )
    {
        std::string out;
        detail::appendTemplateJson( out, value, indent, 0 );

        return out;
    }
}
