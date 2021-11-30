#include "gtest/gtest.h"
#include "Mila.h"

using namespace Mila::Dnn::Cuda;

namespace Mila::Test::Cuda
{
    TEST( Cuda, Create_Device )
    {
        int numErrors = 0;

        CudaDevice device = CudaDevice( 0 );

        std::cout << "Create device" << std::endl;

        EXPECT_TRUE( numErrors == 0 );
    };
}