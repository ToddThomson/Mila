#include "gtest/gtest.h"
#include <iostream>

import Dnn.Model;

using namespace Mila::Dnn;

namespace Mila::Tests::Dnn
{
    TEST( Dnn, Creates_DnnModel )
    {
        DnnModel model = DnnModel();

        std::cout << "Created DNN Model" << std::endl;
    };
}