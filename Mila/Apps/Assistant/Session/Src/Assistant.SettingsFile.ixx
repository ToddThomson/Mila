/**
 * @file Assistant.SettingsFile.ixx
 * @brief Where the Mila app's settings are kept between launches: one small file in the app's own folder.
 */

export module Assistant.SettingsFile;

import std;
import nlohmann.json;
import Assistant.Settings;

namespace Mila::Assistant
{
    /**
     * @brief The person's settings, as JSON, beside the conversations folder (MilaApp.md 3.7).
     *
     * Written by the session whenever a setting changes and after each model that loads, never by hand on the
     * app's behalf, so a setting the file holds that cannot be read is said and passed over rather than ending the
     * app: the rest still apply.
     */
    export class SettingsFile
    {
    public:

        explicit SettingsFile( std::filesystem::path path )
            : path_( std::move( path ) )
        {
        }

        [[nodiscard]] const std::filesystem::path& path() const noexcept
        {
            return path_;
        }

        /**
         * @brief The settings kept, each one the file does not hold, or holds in a form it cannot be, at its default.
         *
         * @param problems Gains one sentence for each setting passed over, and one for a file that is not JSON.
         */
        [[nodiscard]] Settings read( std::vector<std::string>& problems ) const
        {
            Settings settings;
            std::ifstream file( path_, std::ios::binary );

            if ( !file )
            {
                return settings;
            }

            nlohmann::json stored;

            try
            {
                stored = nlohmann::json::parse( file );
            }
            catch ( const nlohmann::json::parse_error& )
            {
                problems.push_back( std::format( "The settings in {} could not be read, so every setting is at its default.",
                    pathText() ) );

                return settings;
            }

            if ( !stored.is_object() )
            {
                problems.push_back( std::format( "The settings in {} could not be read, so every setting is at its default.",
                    pathText() ) );

                return settings;
            }

            const auto passOver = [&]( std::string_view key )
                {
                    problems.push_back( std::format( "The setting {} in {} could not be read, so it is at its default.",
                        key, pathText() ) );
                };

            if ( const auto* model = find( stored, "model" ) )
            {
                if ( model->is_string() )
                    settings.model = model->get<std::string>();
                else
                    passOver( "model" );
            }

            if ( const auto* context = find( stored, "context_length" ) )
            {
                if ( context->is_number_unsigned() && context->get<std::uint64_t>() > 0
                    && context->get<std::uint64_t>() <= static_cast<std::uint64_t>( std::numeric_limits<std::int64_t>::max() ) )
                {
                    settings.context_length = context->get<std::int64_t>();
                } else if ( !( context->is_string() && context->get<std::string>() == "auto" ) ) {
                    passOver( "context_length" );
                }
            }

            if ( const auto* thinking = find( stored, "thinking" ) )
            {
                if ( thinking->is_boolean() )
                    settings.thinking = thinking->get<bool>();
                else
                    passOver( "thinking" );
            }

            if ( const auto* effort = find( stored, "effort" ) )
            {
                if ( effort->is_string() )
                    settings.effort = effort->get<std::string>();
                else
                    passOver( "effort" );
            }

            if ( const auto* detail = find( stored, "detail" ) )
            {
                if ( detail->is_string() && std::ranges::contains( kDetailLevels, detail->get<std::string>() ) )
                    settings.detail = detail->get<std::string>();
                else
                    passOver( "detail" );
            }

            if ( const auto* sampling = find( stored, "sampling" ) )
            {
                if ( sampling->is_object() )
                    readSampling( *sampling, settings, passOver );
                else
                    passOver( "sampling" );
            }

            return settings;
        }

        /**
         * @brief Keep these settings, replacing the file whole, so a write cut short leaves the last one standing.
         *
         * @throws std::runtime_error when the file cannot be written.
         */
        void write( const Settings& settings ) const
        {
            nlohmann::ordered_json sampling = nlohmann::ordered_json::object();
            sampling[ "temperature" ] = optional( settings.sampling.temperature );
            sampling[ "top_k" ] = optional( settings.sampling.top_k );
            sampling[ "top_p" ] = optional( settings.sampling.top_p );
            sampling[ "presence_penalty" ] = optional( settings.sampling.presence_penalty );

            nlohmann::ordered_json stored = nlohmann::ordered_json::object();
            stored[ "v" ] = 1;
            stored[ "model" ] = settings.model.empty() ? nlohmann::ordered_json( nullptr ) : nlohmann::ordered_json( settings.model );
            stored[ "context_length" ] = settings.context_length
                ? nlohmann::ordered_json( *settings.context_length ) : nlohmann::ordered_json( "auto" );
            stored[ "thinking" ] = settings.thinking;
            stored[ "effort" ] = settings.effort.empty() ? nlohmann::ordered_json( nullptr ) : nlohmann::ordered_json( settings.effort );
            stored[ "detail" ] = settings.detail;
            stored[ "sampling" ] = std::move( sampling );

            std::filesystem::create_directories( path_.parent_path() );

            std::filesystem::path partial = path_;
            partial += ".partial";

            {
                std::ofstream file( partial, std::ios::binary | std::ios::trunc );

                if ( !file || !( file << stored.dump( 2 ) << '\n' ) )
                    throw std::runtime_error( std::format( "the settings could not be saved in {}", pathText() ) );
            }

            std::error_code failed;
            std::filesystem::rename( partial, path_, failed );

            if ( failed )
            {
                std::filesystem::remove( partial, failed );

                throw std::runtime_error( std::format( "the settings could not be saved in {}", pathText() ) );
            }
        }

    private:

        static const nlohmann::json* find( const nlohmann::json& stored, std::string_view key )
        {
            const auto found = stored.find( key );

            if ( found == stored.end() || found->is_null() )
            {
                return nullptr;
            }

            return &*found;
        }

        template<typename TPassOver>
        static void readSampling( const nlohmann::json& stored, Settings& settings, const TPassOver& passOver )
        {
            for ( const std::string_view key : kSamplingKeys )
            {
                const auto* value = find( stored, key );

                if ( value == nullptr )
                {
                    continue;
                }

                const bool whole = key != "top_k" || value->is_number_integer();

                if ( !value->is_number() || !whole || !samplingValueAllowed( key, value->get<double>() ) )
                {
                    passOver( std::format( "sampling.{}", key ) );
                    continue;
                }

                if ( key == "temperature" )
                    settings.sampling.temperature = value->get<float>();
                else if ( key == "top_k" )
                    settings.sampling.top_k = value->get<int>();
                else if ( key == "top_p" )
                    settings.sampling.top_p = value->get<float>();
                else
                    settings.sampling.presence_penalty = value->get<float>();
            }
        }

        template<typename T>
        static nlohmann::ordered_json optional( const std::optional<T>& value )
        {
            return value ? nlohmann::ordered_json( *value ) : nlohmann::ordered_json( nullptr );
        }

        std::string pathText() const
        {
            const std::u8string text = path_.u8string();

            return std::string( text.begin(), text.end() );
        }

        std::filesystem::path path_;
    };
}
