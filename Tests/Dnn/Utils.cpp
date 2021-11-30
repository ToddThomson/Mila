#include "gtest/gtest.h"
#include "Mila.h"

using namespace Mila::Dnn;

namespace Mila::Test::Dnn
{
    TEST( CuDNN, Utils_GetVersion )
    {
        int numErrors = 0;

        auto ver = GetVersion();
        std::cout << "CUDNN: Version: " << ver.ToString() << std::endl;

        EXPECT_TRUE( numErrors == 0 );
    };
}
