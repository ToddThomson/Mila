/**
 * @file main.cpp
 * @brief Mila QuickStart: the smallest program that consumes Mila as a dependency.
 *
 * Pulls Mila in via FetchContent (see CMakeLists.txt) and uses its public API. The
 * consumer toolchain recompiles Mila's C++23 module units from source -- including the
 * CUDA operation modules and the kernel headers they include -- and links Mila::Mila.
 * That whole-module recompile is inherent to consuming a module library (BMIs are not
 * portable); FetchContent does it once, in this project's own toolchain.
 */

#include <iostream>

import Mila;

int main()
{
    // Silent by default (NullSink); pass a ConsoleSink to opt in to log output.
    Mila::initialize();

    // Also exercises the version path: getAPIVersion() reads PUBLIC compile
    // definitions that must survive install + export to reach this consumer.
    std::cout << "Mila QuickStart: framework initialized via find_package(Mila). "
              << "Version " << Mila::getAPIVersion().toString() << "\n";

    Mila::shutdown();

    return 0;
}
