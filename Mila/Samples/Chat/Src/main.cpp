#include <string>
#include <vector>
#include <iostream>
#include <filesystem>

import Mila;
import Mila.Chat;

// Helper: path to GPT-2 weights under MODELS_DIR (set via CMake target_compile_definitions)
static std::filesystem::path gpt2_weights_path()
{
    std::filesystem::path models_dir = MODELS_DIR;
    return models_dir / "gpt2" / "gpt2_small_fp32.bin";
}

int main( int argc, char* argv[] )
{
    Mila::initialize();

    try
    {
        auto model_path = gpt2_weights_path();

        if ( argc > 1 )
        {
            model_path = argv[ 1 ];
        }

        if ( !std::filesystem::exists( model_path ) )
        {
            std::cerr << "Error: Model file not found: " << model_path << "\n";
            std::cerr << "Usage: " << argv[ 0 ] << " [model_path]\n";

            return 1;
        }

        Mila::ChatApp::Chat chat( model_path );
        chat.run();

        return 0;
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}