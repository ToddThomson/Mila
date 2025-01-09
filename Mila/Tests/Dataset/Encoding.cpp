#include "gtest/gtest.h"
#include <iostream>
#include <vector>


import Data.CategoryVectorEncoder;

using namespace Mila::Dnn::Data;

namespace Mila::Test::Data
{
    TEST( Dataset, CategoryVector )
    {
        std::vector<int> input = { 2, 3, 1 };
        auto transformer = CategoryVectorEncoder<float>( 3, 1.0 );

        auto output = transformer.Convert( input );
        
        EXPECT_TRUE( output.size() == 9);
    };
}