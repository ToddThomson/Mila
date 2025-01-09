#include "gtest/gtest.h"
#include <iostream>

import Dnn.RnnModel;
import Dnn.RnnModelOptions;

using namespace Mila::Dnn;

namespace Mila::Tests::Dnn
{
    TEST( Dnn, Creates_RnnModel )
    {
        auto options = RnnModelOptions();
        auto model = RnnModel<float>( options );

        std::cout << "Created DNN Model" << std::endl;
    };
}