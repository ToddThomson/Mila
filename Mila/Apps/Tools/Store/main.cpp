/**
 * @file main.cpp
 * @brief Entry point for the `mila-store` command. All work is in Mila.Store.
 */

#include <cstddef>
#include <string_view>
#include <vector>

import Mila.Store;

int main( int argc, char** argv )
{
    std::vector<std::string_view> arguments;
    arguments.reserve( static_cast<std::size_t>( argc ) );

    for ( int index = 0; index < argc; ++index )
    {
        arguments.emplace_back( argv[ index ] );
    }

    return Mila::StoreApp::run( arguments );
}
