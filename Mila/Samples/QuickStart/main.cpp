/**
 * @file main.cpp
 * @brief Mila QuickStart: the smallest program that consumes an installed Mila.
 *
 * Links an installed Mila via find_package and uses its public API. Building this
 * against an installed Mila also serves as the packaging gate: the consumer
 * toolchain recompiles Mila's installed module units -- including the CUDA
 * operation modules and the kernel headers they include -- which is precisely
 * where the install layout is exercised. A successful build proves the package
 * is consumable; Mila's own build cannot prove that, because it links the in-tree
 * target rather than the installed package.
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
