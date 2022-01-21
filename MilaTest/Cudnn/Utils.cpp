#include "gtest/gtest.h"

import CuDnn.Utils;

using namespace Mila::Dnn::CuDnn;

namespace Mila::Test::Dnn
{
    TEST( CuDnn, Utils_GetVersion )
    {
        auto ver = GetCudnnVersion();
        std::cout << "CuDnn: Version: " << ver.ToString() << std::endl;

        EXPECT_TRUE( ver.getMajor() == 8);
        EXPECT_TRUE( ver.getMinor() == 3 );
    };
}
