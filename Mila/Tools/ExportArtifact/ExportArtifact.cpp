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
            "  --emit-manifest                Write mila.json beside the artifact for\n"
            "                                 publishing. Costs a re-read to hash it.\n"
            "  --tokenizer <path>             Tokenizer to record in the manifest.\n"
            "\n"
            "Diagnostics:\n"
            "  ExportArtifact <source> --fingerprint\n"
            "                                 Load only, and print a logits fingerprint for a\n"
            "                                 fixed prompt. Run it against two files that\n"
            "                                 should hold the same model and diff the output.\n";
    }
}

int main( int argc, char** argv )
{
    if ( argc < 3 )
    {
        printUsage();

        return 2;
    }

    // Transport probe: ExportArtifact --fetch <url> <destination>
    if ( std::string_view( argv[ 1 ] ) == "--fetch" )
    {
        if ( argc < 4 )
        {
            std::cerr << "Usage: ExportArtifact --fetch <url> <destination>\n";

            return 2;
        }

        return Mila::Tools::runFetch( argv[ 2 ], argv[ 3 ] );
    }

    Mila::Tools::ExportOptions options;
    options.source = argv[ 1 ];

    // --fingerprint loads and reports only, so it takes no destination.
    const bool fingerprint_only = ( std::string_view( argv[ 2 ] ) == "--fingerprint" );

    if ( fingerprint_only )
    {
        options.fingerprint_only = true;
    }
    else
    {
        options.destination = argv[ 2 ];
    }

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
        else if ( argument == "--fingerprint" )
        {
            options.fingerprint_only = true;
        }
        else if ( argument == "--emit-manifest" )
        {
            options.emit_manifest = true;
        }
        else if ( argument == "--tokenizer" && index + 1 < argc )
        {
            options.tokenizer = argv[ ++index ];
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
