/**
 * @file main.cpp
 * @brief Smallest program that consumes Mila from a published tag via CPM.
 *
 * Mirrors the FetchContent consumer but reaches Mila over the network:
 * CPMAddPackage git-clones the repository at MILA_CPM_GIT_TAG and links it as a
 * subproject. Building it proves the real release-access path -- a downstream
 * project depending on a tagged Mila pulled from the remote.
 */

#include <iostream>

import Mila;

int main()
{
    Mila::initialize();

    std::cout << "Mila CPM consumer: fetched from a published tag via CPMAddPackage. "
              << "Version " << Mila::getAPIVersion().toString() << "\n";

    Mila::shutdown();

    return 0;
}
