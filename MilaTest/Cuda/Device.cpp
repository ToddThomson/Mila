#include "gtest/gtest.h"

import Cuda;

namespace Mila::Test::Cuda
{
    using namespace Mila::Dnn::Cuda;

    TEST( Cuda, Create_Device )
    {
        int numErrors = 0;

        CudaDevice device = CudaDevice( 0 );

        std::cout << "Create device" << std::endl;

        EXPECT_TRUE( numErrors == 0 );
    };
}