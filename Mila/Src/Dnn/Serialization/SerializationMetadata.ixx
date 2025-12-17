/**
 * @file SerializationMetadata.ixx
 * @brief Type-safe metadata container for component serialization.
 *
 * Abstracts serialization format (JSON, XML, binary, etc.) from user code.
 * Provides fluent API for common metadata patterns used in component save/load.
 */

module;
#include <string>
#include <vector>
#include <map>
#include <variant>
#include <optional>
#include <stdexcept>
#include <format>
#include <cstdint>

export module Serialization.Metadata;

import nlohmann.json;
import Dnn.TensorTypes;

namespace Mila::Dnn::Serialization
{
    using json = nlohmann::json;

    /**
     * @brief Type-safe metadata value supporting common serialization types.
     *
     * Encapsulates primitive types, strings, and structured data without
     * exposing the underlying serialization format to user code.
     */
    export using MetadataValue = std::variant<
        std::string,
        int64_t,
        double,
        bool,
        std::vector<int64_t>,
        std::vector<double>,
        std::vector<std::string>
    >;

    /**
     * @brief Type-safe metadata container for component serialization.
     *
     * Provides fluent API for building and reading component metadata without
     * exposing JSON implementation details. Internally uses JSON for serialization
     * but can be swapped with other formats without breaking user code.
     *
     * Common Usage Patterns:
     *
     * **Writing metadata**:
     * @code
     * SerializationMetadata meta;
     * meta.set("type", "Linear")
     *     .set("version", 1)
     *     .set("name", component_name)
     *     .set("input_shape", shape_t{128, 64})
     *     .set("has_bias", true);
     * archive.writeMetadata("meta.json", meta);
     * @endcode
     *
     * **Reading metadata**:
     * @code
     * auto meta = archive.readMetadata("meta.json");
     * std::string type = meta.getString("type");
     * int64_t version = meta.getInt("version");
     * shape_t shape = meta.getShape("input_shape");
     * bool has_bias = meta.getBool("has_bias");
     * @endcode
     */
    export class SerializationMetadata
    {
    public:

        SerializationMetadata() = default;

        // ====================================================================
        // Fluent API for writing metadata
        // ====================================================================

        /**
         * @brief Set metadata value with automatic type deduction.
         *
         * @param key Metadata key
         * @param value Metadata value (string, int64_t, double, bool, vector, shape_t)
         * @return Reference to this for method chaining
         */
        SerializationMetadata& set( const std::string& key, MetadataValue value )
        {
            data_[ key ] = std::move( value );
            return *this;
        }

        /**
         * @brief Set string value.
         *
         * @param key Metadata key
         * @param value String value
         * @return Reference to this for method chaining
         */
        SerializationMetadata& set( const std::string& key, const std::string& value )
        {
            data_[ key ] = value;
            return *this;
        }

        /**
         * @brief Set integer value.
         *
         * @param key Metadata key
         * @param value Integer value
         * @return Reference to this for method chaining
         */
        SerializationMetadata& set( const std::string& key, int64_t value )
        {
            data_[ key ] = value;
            return *this;
        }

        /**
         * @brief Set floating-point value.
         *
         * @param key Metadata key
         * @param value Double value
         * @return Reference to this for method chaining
         */
        SerializationMetadata& set( const std::string& key, double value )
        {
            data_[ key ] = value;
            return *this;
        }

        /**
         * @brief Set boolean value.
         *
         * @param key Metadata key
         * @param value Boolean value
         * @return Reference to this for method chaining
         */
        SerializationMetadata& set( const std::string& key, bool value )
        {
            data_[ key ] = value;
            return *this;
        }

        /**
         * @brief Set double vector value.
         *
         * @param key Metadata key
         * @param value Double vector
         * @return Reference to this for method chaining
         */
        SerializationMetadata& set( const std::string& key, const std::vector<double>& value )
        {
            data_[ key ] = value;
            return *this;
        }

        /**
         * @brief Set string vector value.
         *
         * @param key Metadata key
         * @param value String vector
         * @return Reference to this for method chaining
         */
        SerializationMetadata& set( const std::string& key, const std::vector<std::string>& value )
        {
            data_[ key ] = value;
            return *this;
        }

        // ====================================================================
        // Query operations
        // ====================================================================

        /**
         * @brief Check if key exists in metadata.
         *
         * @param key Metadata key
         * @return true if key exists
         */
        bool has( const std::string& key ) const
        {
            return data_.find( key ) != data_.end();
        }

        /**
         * @brief Get all keys in metadata.
         *
         * @return Vector of metadata keys
         */
        std::vector<std::string> keys() const
        {
            std::vector<std::string> result;
            result.reserve( data_.size() );

            for ( const auto& pair : data_ )
            {
                result.push_back( pair.first );
            }

            return result;
        }

        /**
         * @brief Check if metadata is empty.
         *
         * @return true if no metadata entries
         */
        bool empty() const noexcept
        {
            return data_.empty();
        }

        /**
         * @brief Get number of metadata entries.
         *
         * @return Entry count
         */
        size_t size() const noexcept
        {
            return data_.size();
        }

        // ====================================================================
        // Type-safe getters
        // ====================================================================

        /**
         * @brief Get string value.
         *
         * @param key Metadata key
         * @return String value
         * @throws std::runtime_error if key not found or type mismatch
         */
        std::string getString( const std::string& key ) const
        {
            const auto& value = getValue( key );

            if ( const auto* str = std::get_if<std::string>( &value ) )
            {
                return *str;
            }

            throw std::runtime_error(
                std::format( "SerializationMetadata: key '{}' is not a string", key ) );
        }

        /**
         * @brief Get integer value.
         *
         * @param key Metadata key
         * @return Integer value
         * @throws std::runtime_error if key not found or type mismatch
         */
        int64_t getInt( const std::string& key ) const
        {
            const auto& value = getValue( key );

            if ( const auto* i = std::get_if<int64_t>( &value ) )
            {
                return *i;
            }

            throw std::runtime_error(
                std::format( "SerializationMetadata: key '{}' is not an integer", key ) );
        }

        /**
         * @brief Get double value.
         *
         * @param key Metadata key
         * @return Double value
         * @throws std::runtime_error if key not found or type mismatch
         */
        double getDouble( const std::string& key ) const
        {
            const auto& value = getValue( key );

            if ( const auto* d = std::get_if<double>( &value ) )
            {
                return *d;
            }

            throw std::runtime_error(
                std::format( "SerializationMetadata: key '{}' is not a double", key ) );
        }

        /**
         * @brief Get boolean value.
         *
         * @param key Metadata key
         * @return Boolean value
         * @throws std::runtime_error if key not found or type mismatch
         */
        bool getBool( const std::string& key ) const
        {
            const auto& value = getValue( key );

            if ( const auto* b = std::get_if<bool>( &value ) )
            {
                return *b;
            }

            throw std::runtime_error(
                std::format( "SerializationMetadata: key '{}' is not a boolean", key ) );
        }

        /**
         * @brief Get shape value (integer vector).
         *
         * @param key Metadata key
         * @return Shape vector
         * @throws std::runtime_error if key not found or type mismatch
         */
        shape_t getShape( const std::string& key ) const
        {
            return getIntVector( key );
        }

        /**
         * @brief Get integer vector value.
         *
         * @param key Metadata key
         * @return Integer vector
         * @throws std::runtime_error if key not found or type mismatch
         */
        std::vector<int64_t> getIntVector( const std::string& key ) const
        {
            const auto& value = getValue( key );

            if ( const auto* vec = std::get_if<std::vector<int64_t>>( &value ) )
            {
                return *vec;
            }

            throw std::runtime_error(
                std::format( "SerializationMetadata: key '{}' is not an integer vector", key ) );
        }

        /**
         * @brief Get double vector value.
         *
         * @param key Metadata key
         * @return Double vector
         * @throws std::runtime_error if key not found or type mismatch
         */
        std::vector<double> getDoubleVector( const std::string& key ) const
        {
            const auto& value = getValue( key );

            if ( const auto* vec = std::get_if<std::vector<double>>( &value ) )
            {
                return *vec;
            }

            throw std::runtime_error(
                std::format( "SerializationMetadata: key '{}' is not a double vector", key ) );
        }

        /**
         * @brief Get string vector value.
         *
         * @param key Metadata key
         * @return String vector
         * @throws std::runtime_error if key not found or type mismatch
         */
        std::vector<std::string> getStringVector( const std::string& key ) const
        {
            const auto& value = getValue( key );

            if ( const auto* vec = std::get_if<std::vector<std::string>>( &value ) )
            {
                return *vec;
            }

            throw std::runtime_error(
                std::format( "SerializationMetadata: key '{}' is not a string vector", key ) );
        }

        // ====================================================================
        // Optional getters (return std::optional instead of throwing)
        // ====================================================================

        /**
         * @brief Get optional string value.
         *
         * @param key Metadata key
         * @return String value or std::nullopt if not found or type mismatch
         */
        std::optional<std::string> tryGetString( const std::string& key ) const noexcept
        {
            auto it = data_.find( key );
            if ( it == data_.end() )
            {
                return std::nullopt;
            }

            if ( const auto* str = std::get_if<std::string>( &it->second ) )
            {
                return *str;
            }

            return std::nullopt;
        }

        /**
         * @brief Get optional integer value.
         *
         * @param key Metadata key
         * @return Integer value or std::nullopt if not found or type mismatch
         */
        std::optional<int64_t> tryGetInt( const std::string& key ) const noexcept
        {
            auto it = data_.find( key );
            if ( it == data_.end() )
            {
                return std::nullopt;
            }

            if ( const auto* i = std::get_if<int64_t>( &it->second ) )
            {
                return *i;
            }

            return std::nullopt;
        }

        /**
         * @brief Get optional boolean value.
         *
         * @param key Metadata key
         * @return Boolean value or std::nullopt if not found or type mismatch
         */
        std::optional<bool> tryGetBool( const std::string& key ) const noexcept
        {
            auto it = data_.find( key );
            if ( it == data_.end() )
            {
                return std::nullopt;
            }

            if ( const auto* b = std::get_if<bool>( &it->second ) )
            {
                return *b;
            }

            return std::nullopt;
        }

        /**
         * @brief Get optional shape value.
         *
         * @param key Metadata key
         * @return Shape or std::nullopt if not found or type mismatch
         */
        std::optional<shape_t> tryGetShape( const std::string& key ) const noexcept
        {
            auto it = data_.find( key );
            if ( it == data_.end() )
            {
                return std::nullopt;
            }

            if ( const auto* vec = std::get_if<std::vector<int64_t>>( &it->second ) )
            {
                return *vec;
            }

            return std::nullopt;
        }

        // ====================================================================
        // Internal conversion (ModelArchive implementation detail)
        // ====================================================================

        /**
         * @brief Convert to JSON representation (internal use only).
         *
         * Used by ModelArchive to serialize metadata. Not intended for
         * direct user consumption.
         *
         * @return JSON object
         */
        json toJson() const
        {
            json j = json::object();

            for ( const auto& pair : data_ )
            {
                const std::string& key = pair.first;
                const MetadataValue& value = pair.second;

                std::visit( [ &j, &key ]( const auto& val )
                    {
                        j[ key ] = val;
                    }, value );
            }

            return j;
        }

        /**
         * @brief Construct from JSON representation (internal use only).
         *
         * Used by ModelArchive to deserialize metadata. Not intended for
         * direct user consumption.
         *
         * @param j JSON object
         * @return SerializationMetadata instance
         */
        static SerializationMetadata fromJson( const json& j )
        {
            SerializationMetadata meta;

            for ( auto it = j.begin(); it != j.end(); ++it )
            {
                const std::string& key = it.key();
                const json& value = it.value();

                if ( value.is_string() )
                {
                    meta.set( key, value.get<std::string>() );
                }
                else if ( value.is_number_integer() )
                {
                    meta.set( key, value.get<int64_t>() );
                }
                else if ( value.is_number_float() )
                {
                    meta.set( key, value.get<double>() );
                }
                else if ( value.is_boolean() )
                {
                    meta.set( key, value.get<bool>() );
                }
                else if ( value.is_array() && !value.empty() )
                {
                    if ( value[ 0 ].is_number_integer() )
                    {
                        meta.set( key, value.get<std::vector<int64_t>>() );
                    }
                    else if ( value[ 0 ].is_number_float() )
                    {
                        meta.set( key, value.get<std::vector<double>>() );
                    }
                    else if ( value[ 0 ].is_string() )
                    {
                        meta.set( key, value.get<std::vector<std::string>>() );
                    }
                }
            }

            return meta;
        }

    private:
        std::map<std::string, MetadataValue> data_;

        const MetadataValue& getValue( const std::string& key ) const
        {
            auto it = data_.find( key );
            if ( it == data_.end() )
            {
                throw std::runtime_error(
                    std::format( "SerializationMetadata: key '{}' not found", key ) );
            }

            return it->second;
        }
    };
}