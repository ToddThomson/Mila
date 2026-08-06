/**
 * @file ExportArtifact.Fetch.Http.ixx
 * @brief The --fetch diagnostic, for a build that has an HTTP transport.
 *
 * One of two candidate files for Tools.ExportArtifact.Fetch; CMake compiles this one when
 * MILA_ENABLE_MODEL_HUB is ON. Unlike a pull, this deliberately works in URLs rather than
 * coordinates, because what it exists to diagnose is the transport itself.
 */

module;
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <string>

export module Tools.ExportArtifact.Fetch;

import Mila;

namespace Mila::Tools
{
    /**
     * @brief Fetch one URL through Mila's own HTTP client and report what arrived.
     *
     * Exists because a corrupted 6.33 GB download is an intolerable debugging loop. Pointed
     * at a small file from the same repository it reproduces the transport in seconds, and
     * reports the byte count and digest so the result can be diffed against a known-good
     * local copy.
     *
     * @return Process exit code.
     */
    export int runFetch( const std::string& url, const std::filesystem::path& destination )
    {
        std::unique_ptr<std::FILE, int( * )( std::FILE* )> output(
            std::fopen( destination.string().c_str(), "wb" ), &std::fclose );

        if ( output == nullptr )
        {
            std::cerr << "Cannot open " << destination.string() << " for writing\n";

            return 2;
        }

        Mila::Distribution::Sha256 hash;
        uint64_t written = 0;

        Mila::Distribution::HttpRequest request;
        request.url = url;
        request.token = Mila::Distribution::discoverHuggingFaceToken();

        std::cout << std::format( "Fetching {}\n", url );

        const auto result = Mila::Distribution::httpGet( request,
            [&]( const char* data, size_t length ) -> bool
            {
                if ( std::fwrite( data, 1, length, output.get() ) != length )
                {
                    return false;
                }

                hash.update( data, length );
                written += length;

                return true;
            } );

        output.reset();

        std::cout << std::format( "  status         {}\n", toString( result.status ) );
        std::cout << std::format( "  http code      {}\n", result.http_code );
        std::cout << std::format( "  final url      {}\n", result.final_url );
        std::cout << std::format( "  content-length {}\n", result.content_length );
        std::cout << std::format( "  bytes written  {}\n", written );
        std::cout << std::format( "  sha256         {}\n", hash.finish() );

        if ( !result.message.empty() )
        {
            std::cout << std::format( "  message        {}\n", result.message );
        }

        return result.ok() ? 0 : 1;
    }
}
