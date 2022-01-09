#include "gtest/gtest.h"

import CuDnn.Version;
import CuDnn.Utils;

using namespace Mila::Dnn::CuDNN;

namespace Mila::Test::Dnn
{
    TEST( CuDnn, Utils_GetVersion )
    {
        int numErrors = 0;

        auto ver = GetVersion();
        std::cout << "CUDNN: Version: " << ver.ToString() << std::endl;

        EXPECT_TRUE( numErrors == 0 );
    };
}
