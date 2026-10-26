/**
 * @file ArgumentSchema.ixx
 * @brief A tool's parameters as a JSON Schema, and the check a call's arguments pass before the tool runs.
 */

export module AI.ArgumentSchema;

import std;
import nlohmann.json;

namespace Mila::AI
{
    /**
     * @brief The JSON Schema a tool's arguments must match (ToolCalling.md 3.2).
     *
     * Checks a stated part of JSON Schema, at any depth: type, properties, required, additionalProperties, enum,
     * const, items, minLength, maxLength, minimum, maximum, exclusiveMinimum, exclusiveMaximum, minItems and
     * maxItems. title, description, default, examples, format, $comment and $schema are annotations and check
     * nothing. A schema with any other keyword is refused, so no check is claimed that is not made.
     */
    export class ArgumentSchema
    {
    public:

        /// @throws std::invalid_argument for text that is not a JSON object, or a keyword this check does not make.
        explicit ArgumentSchema( std::string_view schema )
            : schema_( nlohmann::ordered_json::parse( schema, nullptr, false ) )
        {
            if ( schema_.is_discarded() || !schema_.is_object() )
            {
                throw std::invalid_argument( "parameters must be a JSON Schema object" );
            }

            refuseUnchecked( schema_, "" );
        }

        /**
         * @brief The first way the arguments fail the schema, written for the model to correct; nothing when they
         *        match.
         *
         * @param arguments The call's arguments as the model wrote them, which must be a JSON object.
         */
        [[nodiscard]] std::optional<std::string> check( std::string_view arguments ) const
        {
            const auto value = nlohmann::ordered_json::parse( arguments, nullptr, false );

            if ( value.is_discarded() )
            {
                return std::string( "the arguments are not JSON" );
            }

            if ( !value.is_object() )
            {
                return std::string( "the arguments are not a JSON object" );
            }

            return mismatch( schema_, value, "" );
        }

    private:

        static constexpr std::array<std::string_view, 7> kAnnotations{
            "title", "description", "default", "examples", "format", "$comment", "$schema" };

        static constexpr std::array<std::string_view, 7> kTypes{
            "object", "array", "string", "number", "integer", "boolean", "null" };

        static void refuseUnchecked( const nlohmann::ordered_json& schema, const std::string& location )
        {
            if ( !schema.is_object() )
            {
                throw std::invalid_argument( std::format( "the schema at '{}' is not an object", shown( location ) ) );
            }

            for ( const auto& item : schema.items() )
            {
                const std::string& keyword = item.key();
                const nlohmann::ordered_json& value = item.value();
                const std::string here = location + "/" + keyword;

                if ( std::ranges::contains( kAnnotations, keyword ) || keyword == "const" )
                {
                    continue;
                }

                if ( keyword == "type" )
                {
                    refuseUnknownTypes( value, here );
                }
                else if ( keyword == "properties" )
                {
                    if ( !value.is_object() )
                    {
                        throw std::invalid_argument( std::format( "'{}' must be an object", here ) );
                    }

                    for ( const auto& property : value.items() )
                    {
                        refuseUnchecked( property.value(), here + "/" + property.key() );
                    }
                }
                else if ( keyword == "required" )
                {
                    if ( !value.is_array()
                        || !std::all_of( value.begin(), value.end(), []( const auto& name ) { return name.is_string(); } ) )
                    {
                        throw std::invalid_argument( std::format( "'{}' must be an array of names", here ) );
                    }
                }
                else if ( keyword == "additionalProperties" )
                {
                    if ( !value.is_boolean() )
                    {
                        refuseUnchecked( value, here );
                    }
                }
                else if ( keyword == "items" )
                {
                    refuseUnchecked( value, here );
                }
                else if ( keyword == "enum" )
                {
                    if ( !value.is_array() || value.empty() )
                    {
                        throw std::invalid_argument( std::format( "'{}' must be an array of values", here ) );
                    }
                }
                else if ( keyword == "minLength" || keyword == "maxLength" || keyword == "minItems" || keyword == "maxItems" )
                {
                    if ( !value.is_number_unsigned() )
                    {
                        throw std::invalid_argument( std::format( "'{}' must be a count", here ) );
                    }
                }
                else if ( keyword == "minimum" || keyword == "maximum"
                    || keyword == "exclusiveMinimum" || keyword == "exclusiveMaximum" )
                {
                    if ( !value.is_number() )
                    {
                        throw std::invalid_argument( std::format( "'{}' must be a number", here ) );
                    }
                }
                else
                {
                    throw std::invalid_argument( std::format(
                        "the schema uses '{}' at '{}', which Mila does not check", keyword, shown( location ) ) );
                }
            }
        }

        static void refuseUnknownTypes( const nlohmann::ordered_json& value, const std::string& location )
        {
            const auto known = []( const nlohmann::ordered_json& type )
                {
                    return type.is_string() && std::ranges::contains( kTypes, type.get<std::string>() );
                };

            const bool valid = value.is_array()
                ? !value.empty() && std::all_of( value.begin(), value.end(), known )
                : known( value );

            if ( !valid )
            {
                throw std::invalid_argument( std::format( "'{}' is not a JSON Schema type: {}", location, value.dump() ) );
            }
        }

        static std::string shown( const std::string& location )
        {
            return location.empty() ? "/" : location;
        }

        /// The argument a path names, as the model reads it: "the arguments" at the top, else its name in quotes.
        static std::string where( const std::string& path )
        {
            return path.empty() ? "the arguments" : std::format( "'{}'", path );
        }

        static std::string child( const std::string& path, std::string_view name )
        {
            return path.empty() ? std::string( name ) : std::format( "{}.{}", path, name );
        }

        static bool hasType( const nlohmann::ordered_json& value, std::string_view type )
        {
            if ( type == "integer" )
            {
                return value.is_number_integer()
                    || ( value.is_number_float() && std::trunc( value.get<double>() ) == value.get<double>() );
            }

            if ( type == "number" )
            {
                return value.is_number();
            }

            return value.type_name() == type;
        }

        static std::string article( std::string_view type )
        {
            if ( type == "null" )
            {
                return "null";
            }

            const bool vowel = type.front() == 'a' || type.front() == 'i' || type.front() == 'o';

            return std::format( "{} {}", vowel ? "an" : "a", type );
        }

        static std::string valueKind( const nlohmann::ordered_json& value )
        {
            return article( value.is_number() ? std::string_view( "number" ) : std::string_view( value.type_name() ) );
        }

        /// Characters, as JSON Schema counts a string's length: code points, not bytes.
        static std::size_t characters( const std::string& text ) noexcept
        {
            return static_cast<std::size_t>( std::ranges::count_if(
                text, []( char byte ) { return ( static_cast<unsigned char>( byte ) & 0xC0 ) != 0x80; } ) );
        }

        static std::string counted( const nlohmann::ordered_json& count, std::string_view noun )
        {
            return std::format( "{} {}{}", count.dump(), noun, count.get<std::size_t>() == 1 ? "" : "s" );
        }

        static std::string joined( const nlohmann::ordered_json& values )
        {
            std::string text;

            for ( const auto& value : values )
            {
                text += std::format( "{}{}", text.empty() ? "" : ", ", value.dump() );
            }

            return text;
        }

        static std::optional<std::string> mismatch(
            const nlohmann::ordered_json& schema, const nlohmann::ordered_json& value, const std::string& path )
        {
            if ( const auto type = schema.find( "type" ); type != schema.end() )
            {
                std::vector<std::string> types;

                if ( type->is_array() )
                {
                    for ( const auto& name : *type )
                    {
                        types.push_back( name.get<std::string>() );
                    }
                }
                else
                {
                    types.push_back( type->get<std::string>() );
                }

                if ( !std::ranges::any_of( types, [&value]( const std::string& name ) { return hasType( value, name ); } ) )
                {
                    std::string expected;

                    for ( const auto& name : types )
                    {
                        expected += std::format( "{}{}", expected.empty() ? "" : " or ", article( name ) );
                    }

                    return std::format( "{} must be {}, not {}", where( path ), expected, valueKind( value ) );
                }
            }

            if ( const auto values = schema.find( "enum" ); values != schema.end()
                && std::find( values->begin(), values->end(), value ) == values->end() )
            {
                return std::format( "{} must be one of {}", where( path ), joined( *values ) );
            }

            if ( const auto constant = schema.find( "const" ); constant != schema.end() && *constant != value )
            {
                return std::format( "{} must be {}", where( path ), constant->dump() );
            }

            if ( value.is_string() )
            {
                return stringMismatch( schema, value.get<std::string>(), path );
            }

            if ( value.is_number() )
            {
                return numberMismatch( schema, value, path );
            }

            if ( value.is_array() )
            {
                return arrayMismatch( schema, value, path );
            }

            if ( value.is_object() )
            {
                return objectMismatch( schema, value, path );
            }

            return std::nullopt;
        }

        static std::optional<std::string> stringMismatch(
            const nlohmann::ordered_json& schema, const std::string& text, const std::string& path )
        {
            const std::size_t length = characters( text );

            if ( const auto least = schema.find( "minLength" ); least != schema.end() && length < least->get<std::size_t>() )
            {
                return std::format( "{} must be at least {}", where( path ), counted( *least, "character" ) );
            }

            if ( const auto most = schema.find( "maxLength" ); most != schema.end() && length > most->get<std::size_t>() )
            {
                return std::format( "{} must be at most {}", where( path ), counted( *most, "character" ) );
            }

            return std::nullopt;
        }

        static std::optional<std::string> numberMismatch(
            const nlohmann::ordered_json& schema, const nlohmann::ordered_json& value, const std::string& path )
        {
            if ( const auto bound = schema.find( "minimum" ); bound != schema.end() && value < *bound )
            {
                return std::format( "{} must be at least {}", where( path ), bound->dump() );
            }

            if ( const auto bound = schema.find( "maximum" ); bound != schema.end() && value > *bound )
            {
                return std::format( "{} must be at most {}", where( path ), bound->dump() );
            }

            if ( const auto bound = schema.find( "exclusiveMinimum" ); bound != schema.end() && value <= *bound )
            {
                return std::format( "{} must be greater than {}", where( path ), bound->dump() );
            }

            if ( const auto bound = schema.find( "exclusiveMaximum" ); bound != schema.end() && value >= *bound )
            {
                return std::format( "{} must be less than {}", where( path ), bound->dump() );
            }

            return std::nullopt;
        }

        static std::optional<std::string> arrayMismatch(
            const nlohmann::ordered_json& schema, const nlohmann::ordered_json& value, const std::string& path )
        {
            if ( const auto least = schema.find( "minItems" ); least != schema.end() && value.size() < least->get<std::size_t>() )
            {
                return std::format( "{} must have at least {}", where( path ), counted( *least, "item" ) );
            }

            if ( const auto most = schema.find( "maxItems" ); most != schema.end() && value.size() > most->get<std::size_t>() )
            {
                return std::format( "{} must have at most {}", where( path ), counted( *most, "item" ) );
            }

            if ( const auto items = schema.find( "items" ); items != schema.end() )
            {
                for ( std::size_t index = 0; index < value.size(); ++index )
                {
                    if ( auto found = mismatch( *items, value[ index ], std::format( "{}[{}]", path, index ) ) )
                    {
                        return found;
                    }
                }
            }

            return std::nullopt;
        }

        static std::optional<std::string> objectMismatch(
            const nlohmann::ordered_json& schema, const nlohmann::ordered_json& value, const std::string& path )
        {
            if ( const auto required = schema.find( "required" ); required != schema.end() )
            {
                for ( const auto& name : *required )
                {
                    if ( !value.contains( name.get<std::string>() ) )
                    {
                        return std::format( "'{}' is required", child( path, name.get<std::string>() ) );
                    }
                }
            }

            const auto properties = schema.find( "properties" );
            const auto additional = schema.find( "additionalProperties" );

            for ( const auto& item : value.items() )
            {
                const std::string& name = item.key();
                const nlohmann::ordered_json& argument = item.value();

                if ( properties != schema.end() && properties->contains( name ) )
                {
                    if ( auto found = mismatch( ( *properties )[ name ], argument, child( path, name ) ) )
                    {
                        return found;
                    }

                    continue;
                }

                if ( additional == schema.end() )
                {
                    continue;
                }

                if ( additional->is_boolean() )
                {
                    if ( !additional->get<bool>() )
                    {
                        return std::format( "'{}' is not a parameter of this tool", child( path, name ) );
                    }

                    continue;
                }

                if ( auto found = mismatch( *additional, argument, child( path, name ) ) )
                {
                    return found;
                }
            }

            return std::nullopt;
        }

        nlohmann::ordered_json schema_;
    };
}
