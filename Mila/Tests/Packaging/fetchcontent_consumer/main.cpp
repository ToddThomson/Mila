/**
 * @file main.cpp
 * @brief Smallest program that embeds Mila as a subproject (FetchContent).
 *
 * Mirrors Samples/QuickStart/Cpp but reaches Mila via add_subdirectory rather than
 * find_package. Building it proves the subproject (FetchContent/CPM-by-semver)
 * consumption path works end to end.
 *
 * It deliberately does more than print a version. A gate that only links Mila proves the
 * library builds, not that its module is USABLE from a consumer translation unit -- and the
 * three known module-consumption defects are all compile-time, so the fixture catches them
 * by compiling rather than by running. It needs no GPU and no installed model.
 */

#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

 // WORKAROUND 1, and it must stay BEFORE `import Mila;` -- importing first is a fatal MSVC
 // modules error (C1116), which is itself the third defect this fixture pins by ordering.
 // Instantiating a Mila model instantiates its vtable, and Component::toString() is virtual,
 // so the model's toString() body is compiled HERE; it uses std::ostringstream, whose
 // definition does not reach this translation unit through the module.
 // Without <sstream>: "'oss' uses undefined class std::basic_ostringstream".
 // Delete this include once the library stops requiring it, and this gate will say so.
#include <sstream>

import Mila;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;

namespace
{
    /**
     * @brief Compiled, never called -- on purpose.
     *
     * The defect this pins is compile-time: naming `fromPretrained` instantiates
     * `GemmaModel`, which emits its vtable and compiles the virtual `toString()` body into
     * this translation unit. Running it would need a GPU and an installed model; compiling
     * it needs neither, and the failure it guards against happens at compile time either way.
     *
     * A gate that only linked `Mila::Mila` reported success while this exact instantiation
     * was broken for consumers, which is why the fixture now carries it.
     */
    [[maybe_unused]] void instantiateModelFromConsumerTranslationUnit()
    {
        GemmaModelConfig model_config( 4096 );
        model_config.withFP4Quantization();

        auto model = GemmaModel<DeviceType::Cuda, TensorDataType::BF16>::fromPretrained(
            "unreachable-never-called", model_config );

        // Force the virtual through the base, so the vtable cannot be elided.
        std::cout << model->toString() << "\n";
    }
}

int main()
{
    try
    {
        Mila::initialize();

        std::cout << "Mila FetchContent consumer: embedded via add_subdirectory. "
            << "Version " << Mila::getAPIVersion().toString() << "\n";

        // WORKAROUND 2: std::fgets, not std::getline or std::cin.getline. Any C++ stream
        // INPUT in a translation unit that imports Mila fails to compile -- both forms
        // instantiate basic_istream machinery here and hit "'_Ok' uses undefined class
        // basic_istream::sentry". Output (std::cout) is fine. <cstdio> instantiates nothing.
        //
        // RE-VERIFIED 2026-08-30 on MSVC 14.51.36231, in this fixture: swapping to
        // std::getline fails exactly as above, so <sstream> above does NOT also repair
        // input. The two workarounds are independent and both are still load-bearing.
        //
        // The gate reads a line so the consumer story covers input at all; swap this for
        // std::getline once the library stops breaking it, and the compile will say so.
        char line[ 256 ]{};

        if ( std::fgets( line, sizeof( line ), stdin ) != nullptr )
        {
            std::cout << "Read " << std::string( line ).size() << " bytes from stdin.\n";
        }
        else
        {
            std::cout << "No stdin; input path compiled but not exercised.\n";
        }

        Mila::shutdown();

        return 0;
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Consumer failed: " << e.what() << "\n";

        return 1;
    }
}
