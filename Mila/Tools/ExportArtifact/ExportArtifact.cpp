/**
 * @file ExportArtifact.cpp
 * @brief Entry point. All Mila use lives in the module unit -- see the CMakeLists note.
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

import Tools.ExportArtifact;

namespace
{
    void printUsage()
    {
        std::cout <<
            "Usage: ExportArtifact <source.bin> <destination.safetensors> [options]\n"
            "\n"
            "  --quantization <fp4|fp8|none>  Weight quantization to apply on load\n"
            "                                 (default: fp4)\n"
            "  --context-length <n>           Build context length (default: 4096).\n"
            "                                 Only affects KV-cache allocation during\n"
            "                                 the load; the export runs no forward pass.\n";
    }
}

int main( int argc, char** argv )
{
    if ( argc < 3 )
    {
        printUsage();

        return 2;
    }

    Mila::Tools::ExportOptions options;
    options.source = argv[ 1 ];
    options.destination = argv[ 2 ];

    for ( int index = 3; index < argc; ++index )
    {
        const std::string_view argument = argv[ index ];

        if ( argument == "--quantization" && index + 1 < argc )
        {
            if ( !Mila::Tools::parseQuantization( argv[ ++index ], options.quantization ) )
            {
                std::cerr << "Unknown quantization: " << argv[ index ] << "\n";

                return 2;
            }
        }
        else if ( argument == "--context-length" && index + 1 < argc )
        {
            options.context_length = std::atoll( argv[ ++index ] );

            if ( options.context_length <= 0 )
            {
                std::cerr << "context-length must be positive\n";

                return 2;
            }
        }
        else
        {
            std::cerr << "Unknown argument: " << argument << "\n";
            printUsage();

            return 2;
        }
    }

    return Mila::Tools::runExport( options );
}
