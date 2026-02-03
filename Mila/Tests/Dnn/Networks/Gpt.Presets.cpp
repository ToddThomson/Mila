#include <gtest/gtest.h>

import Mila;

namespace CompositeComponents_Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Networks;

    TEST( TransformerPresets, GPT2SmallProperties )
    {
        // Verify primary dimensions exposed by the GPT-2 small preset.
        auto cfg = GPT2_Small();

        EXPECT_EQ( cfg.getEmbeddingSize(), 768 );
        EXPECT_EQ( cfg.getNumLayers(), 12 );
        EXPECT_EQ( cfg.getNumHeads(), 12 );

        // Default max sequence length is present at network-level
        EXPECT_EQ( cfg.getMaxSequenceLength(), static_cast<dim_t>(1024) );

        // Preset should validate without throwing.
        EXPECT_NO_THROW( cfg.validate() );
    }
}