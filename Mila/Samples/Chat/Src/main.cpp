#include <string>
#include <vector>
#include <iostream>
#include <filesystem>

import Chat;

int main( int argc, char* argv[] )
{
    try {
        std::string modelPath = "models/gpt2.bin";

        if ( argc > 1 ) {
            modelPath = argv[ 1 ];
        }

        if ( !std::filesystem::exists( modelPath ) ) {
            std::cerr << "Error: Model file not found: " << modelPath << "\n";
            std::cerr << "Usage: " << argv[ 0 ] << " [model_path]\n";
            return 1;
        }

        Mila::ChatApp::Chat chat(modelPath);
        chat.run();

        return 0;
    }
    catch ( const std::exception& e ) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}