/**
 * @file Serializer.ixx
 * @brief Minimal base interface for all serialization backends.
 */

module;
#include <string>

export module Serialization.Serializer;

import Serialization.OpenMode;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Minimal base interface for model serialization backends.
     *
     * Defines only the essential lifecycle operations common to all serializers.
     * Derived classes implement format-specific APIs.
     */
    export class Serializer
    {
    public:
        virtual ~Serializer() = default;

        /**
         * @brief Open a file for reading or writing.
         */
        [[nodiscard]] virtual bool open( const std::string& filename, OpenMode mode ) = 0;

        /**
         * @brief Close the file.
         */
        [[nodiscard]] virtual bool close() = 0;

        /**
         * @brief Check if currently open.
         */
        [[nodiscard]] virtual bool isOpen() const noexcept = 0;

        /**
         * @brief Check if open for reading.
         */
        [[nodiscard]] virtual bool isOpenForRead() const noexcept = 0;

        /**
         * @brief Check if open for writing.
         */
        [[nodiscard]] virtual bool isOpenForWrite() const noexcept = 0;

        /**
         * @brief Get the filename.
         */
        [[nodiscard]] virtual const std::string& getFilename() const noexcept = 0;
    };
}