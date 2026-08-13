/**
 * @file main.cpp
 * @brief Smallest program that embeds Mila as a subproject (FetchContent).
 *
 * Mirrors Samples/QuickStart/Cpp but reaches Mila via add_subdirectory rather than
 * find_package. Building it proves the subproject (FetchContent/CPM-by-semver)
 * consumption path works end to end.
 */

#include <iostream>

import Mila;

int main()
{
    Mila::initialize();

    std::cout << "Mila FetchContent consumer: embedded via add_subdirectory. "
              << "Version " << Mila::getAPIVersion().toString() << "\n";

    Mila::shutdown();

    return 0;
}
