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
            "  --quantization <fp4|fp8|plan|none>\n"
            "                                 Weight quantization to apply on load\n"
            "                                 (default: fp4). 'plan' is the family's own\n"
            "                                 per-role allocation and needs a source whose\n"
            "                                 fitted tensors are already packed -- for Qwen 3.8\n"
            "                                 that is pack_qwen.py's output, and 'cb2-3' is\n"
            "                                 accepted as the name of what comes out.\n"
            "  --package <dir>                Assemble the package directory around the\n"
            "                                 artifact: mila.json with real digests, plus\n"
            "                                 whatever supporting files are given. One package is\n"
            "                                 one model, so a directory that already describes\n"
            "                                 one is refused unless --replace is given.\n"
            "  --emit-manifest                Package in place, beside the artifact.\n"
            "  --replace                      Build over a package directory that already\n"
            "                                 describes a model.\n"
            "  --tokenizer <path>             Tokenizer to declare in the manifest.\n"
            "  --license <path>               Source model's license text.\n"
            "  --notice <path>                Attribution the license requires be carried in a\n"
            "                                 file of its own (Llama names a 'Notice' file).\n"
            "  --model-card <path>            Model card (README.md).\n"
            "\n"
            "Artifacts that already exist (no GPU, no model load):\n"
            "  ExportArtifact --transcode <source> <destination.safetensors>\n"
            "                                 Rewrite a model file as safetensors, tensor for\n"
            "                                 tensor. Family-agnostic: it changes the container,\n"
            "                                 never the numbers, so it works for every family\n"
            "                                 the reader can open.\n"
            "  ExportArtifact --package <dir> --weights <artifact> [options]\n"
            "                                 Assemble a package around a finished artifact.\n"
            "                                 Architecture, quantization and variant are read\n"
            "                                 from the artifact itself.\n"
            "      --tokenizer <path>         Tokenizer to declare.\n"
            "      --license <path>           Source model's license text.\n"
            "      --notice <path>            Attribution the license requires be carried in a\n"
            "                                 file of its own (Llama names a 'Notice' file).\n"
            "      --model-card <path>        Model card (README.md).\n"
            "      --as <name>                Model name. Defaults to the directory name.\n"
            "      --variant <name>           Override the derived variant name.\n"
            "      --base-model <id>          Lineage, published with the model.\n"
            "      --license-id <id>          License identifier, e.g. llama3.2.\n"
            "      --replace                  Build over a directory that already describes a\n"
            "                                 model. Refused without it.\n"
            "\n"
            "Packages:\n"
            "  ExportArtifact --validate <package-dir>\n"
            "                                 Read every declared byte and report whether the\n"
            "                                 package agrees with its own manifest.\n"
            "  ExportArtifact --rename <from> <to>\n"
            "                                 Rename an installed model. Rewrites one record;\n"
            "                                 no bytes move.\n"
            "  ExportArtifact --install <package-dir> [options]\n"
            "                                 Install into the local model store.\n"
            "      --as <name>                Name to install under. One name is one model;\n"
            "                                 defaults to the manifest's, then the directory's.\n"
            "      --replace                  Replace an existing record of that name.\n"
            "      --keep                     Copy rather than move, leaving the package\n"
            "                                 intact for a hub upload.\n"
            "\n"
            "Diagnostics:\n"
            "  ExportArtifact <source> --fingerprint\n"
            "                                 Load only, and print a logits fingerprint for a\n"
            "                                 fixed prompt. Run it against two files that\n"
            "                                 should hold the same model and diff the output.\n"
            "  ExportArtifact --fetch <url> <destination> [--resume]\n"
            "                                 Pull one URL through Mila's own HTTP client and\n"
            "                                 report transport, status, final URL, byte counts\n"
            "                                 and digest.\n"
            "      --resume                   Resume from whatever <destination> already holds,\n"
            "                                 replaying it into the digest and sending a Range\n"
            "                                 header. The offset comes from the file, because\n"
            "                                 that is the only offset the store can produce.\n"
            "                                 Truncate a verified copy and re-run: the digest\n"
            "                                 must come back identical.\n";
    }

    int runPackageCommand( int argc, char** argv )
    {
        if ( argc < 3 )
        {
            std::cerr << "Usage: ExportArtifact --package <dir> --weights <artifact> [options]\n";

            return 2;
        }

        Mila::Tools::PackageArtifactRequest request;
        request.directory = argv[ 2 ];

        for ( int index = 3; index < argc; ++index )
        {
            const std::string_view argument = argv[ index ];

            if ( argument == "--weights" && index + 1 < argc )
            {
                request.weights = argv[ ++index ];
            }
            else if ( argument == "--tokenizer" && index + 1 < argc )
            {
                request.tokenizer = argv[ ++index ];
            }
            else if ( argument == "--license" && index + 1 < argc )
            {
                request.license = argv[ ++index ];
            }
            else if ( argument == "--notice" && index + 1 < argc )
            {
                request.notice = argv[ ++index ];
            }
            else if ( argument == "--model-card" && index + 1 < argc )
            {
                request.model_card = argv[ ++index ];
            }
            else if ( argument == "--variant" && index + 1 < argc )
            {
                request.variant = argv[ ++index ];
            }
            else if ( argument == "--as" && index + 1 < argc )
            {
                request.name = argv[ ++index ];
            }
            else if ( argument == "--base-model" && index + 1 < argc )
            {
                request.base_model = argv[ ++index ];
            }
            else if ( argument == "--license-id" && index + 1 < argc )
            {
                request.license_id = argv[ ++index ];
            }
            else if ( argument == "--instruct" )
            {
                request.instruct = true;
            }
            else if ( argument == "--replace" )
            {
                request.replace = true;
            }
            else
            {
                std::cerr << "Unknown argument: " << argument << "\n";

                return 2;
            }
        }

        if ( request.weights.empty() )
        {
            std::cerr << "--package needs --weights <artifact>\n";

            return 2;
        }

        return Mila::Tools::runPackageArtifact( request );
    }

    int runInstallCommand( int argc, char** argv )
    {
        if ( argc < 3 )
        {
            std::cerr << "Usage: ExportArtifact --install <package-dir> [options]\n";

            return 2;
        }

        Mila::Tools::InstallRequest request;
        request.package_directory = argv[ 2 ];

        for ( int index = 3; index < argc; ++index )
        {
            const std::string_view argument = argv[ index ];

            if ( argument == "--as" && index + 1 < argc )
            {
                request.name = argv[ ++index ];
            }
            else if ( argument == "--replace" )
            {
                request.replace = true;
            }
            else if ( argument == "--keep" )
            {
                request.keep_package = true;
            }
            else
            {
                std::cerr << "Unknown argument: " << argument << "\n";

                return 2;
            }
        }

        return Mila::Tools::runInstall( request );
    }
}

int main( int argc, char** argv )
{
    if ( argc < 2 )
    {
        printUsage();

        return 2;
    }

    const std::string_view first = argv[ 1 ];

    // Package management: these take a directory, not a model, and load nothing.
    if ( first == "--validate" )
    {
        if ( argc < 3 )
        {
            std::cerr << "Usage: ExportArtifact --validate <package-dir>\n";

            return 2;
        }

        return Mila::Tools::runValidate( argv[ 2 ] );
    }

    if ( first == "--install" )
    {
        return runInstallCommand( argc, argv );
    }

    // A finished artifact in, a package out. --package is also an option in export mode; as
    // the first argument it is the standalone verb over a file that already exists.
    if ( first == "--package" )
    {
        return runPackageCommand( argc, argv );
    }

    if ( first == "--rename" )
    {
        if ( argc < 4 )
        {
            std::cerr << "Usage: ExportArtifact --rename <from> <to>\n";

            return 2;
        }

        return Mila::Tools::runRename( argv[ 2 ], argv[ 3 ] );
    }

    if ( first == "--compare" )
    {
        if ( argc < 4 )
        {
            std::cerr << "Usage: ExportArtifact --compare <left> <right>\n";

            return 2;
        }

        return Mila::Tools::runCompare( argv[ 2 ], argv[ 3 ] );
    }

    if ( first == "--transcode" )
    {
        if ( argc < 4 )
        {
            std::cerr << "Usage: ExportArtifact --transcode <source> <destination>\n";

            return 2;
        }

        return Mila::Tools::runTranscode( argv[ 2 ], argv[ 3 ] );
    }

    // Transport probe: ExportArtifact --fetch <url> <destination> [--resume]
    if ( first == "--fetch" )
    {
        if ( argc < 4 )
        {
            std::cerr << "Usage: ExportArtifact --fetch <url> <destination> [--resume]\n";

            return 2;
        }

        bool resume = false;

        for ( int index = 4; index < argc; ++index )
        {
            if ( std::string_view( argv[ index ] ) == "--resume" )
            {
                resume = true;
            }
            else
            {
                std::cerr << "Unknown option for --fetch: " << argv[ index ] << "\n";

                return 2;
            }
        }

        return Mila::Tools::runFetch( argv[ 2 ], argv[ 3 ], resume );
    }

    if ( argc < 3 )
    {
        printUsage();

        return 2;
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

    bool package_in_place = false;

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
        else if ( argument == "--package" && index + 1 < argc )
        {
            options.package_directory = argv[ ++index ];
        }
        else if ( argument == "--emit-manifest" )
        {
            package_in_place = true;
        }
        else if ( argument == "--tokenizer" && index + 1 < argc )
        {
            options.tokenizer = argv[ ++index ];
        }
        else if ( argument == "--license" && index + 1 < argc )
        {
            options.license = argv[ ++index ];
        }
        else if ( argument == "--notice" && index + 1 < argc )
        {
            options.notice = argv[ ++index ];
        }
        else if ( argument == "--model-card" && index + 1 < argc )
        {
            options.model_card = argv[ ++index ];
        }
        else if ( argument == "--replace" )
        {
            options.replace_package = true;
        }
        else
        {
            std::cerr << "Unknown argument: " << argument << "\n";
            printUsage();

            return 2;
        }
    }

    // Resolved after the loop: --emit-manifest means the artifact's own directory, and the
    // destination is not known to be parsed when the flag is read.
    if ( package_in_place && options.package_directory.empty() )
    {
        options.package_directory = options.destination.parent_path();

        // A bare filename has no parent, and an empty package directory reads as "no packaging".
        if ( options.package_directory.empty() )
        {
            options.package_directory = ".";
        }
    }

    return Mila::Tools::runExport( options );
}
