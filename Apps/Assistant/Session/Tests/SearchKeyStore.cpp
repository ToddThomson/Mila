/**
 * @file SearchKeyStore.cpp
 * @brief The web search key is kept by the operating system's credential store, read back, replaced and removed
 *        (AssistantInterface.md 8), under a name of the test's own so the person's key is never touched.
 */

#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <string>

import Assistant.SearchKeyStore;

namespace Mila::Tests::Apps::Assistant
{
    using Mila::Assistant::SearchKeyStore;

    TEST( SearchKeyStore, AKeyIsKeptReplacedAndRemoved )
    {
        const SearchKeyStore store( "Mila Assistant test key, safe to delete" );
        store.remove();

        EXPECT_FALSE( store.read().has_value() );

        // Where no credential store can be reached, as on a Linux computer with no keyring running, there is nothing to test.
        try
        {
            store.write( "tvly-first" );
        }
        catch ( const std::runtime_error& refused )
        {
            GTEST_SKIP() << refused.what();
        }

        EXPECT_EQ( store.read(), std::optional<std::string>( "tvly-first" ) );

        store.write( "tvly-second" );
        EXPECT_EQ( store.read(), std::optional<std::string>( "tvly-second" ) );

        store.remove();
        EXPECT_FALSE( store.read().has_value() );
    }
}
