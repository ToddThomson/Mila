#include "gtest/gtest.h"

import Cuda;

namespace Mila::Test::Cuda
{
    using namespace Mila::Dnn::Cuda;

    TEST( Cuda, CudaDevice_Constructor_Passes )
    {
        EXPECT_NO_THROW( CudaDevice device = CudaDevice( 0 ) );
    };

    TEST( Cuda, CudaDevice_Constructor_Fails_Out_Of_Range )
    {
        int device_count = GetDeviceCount();

        EXPECT_THROW( CudaDevice device = CudaDevice( device_count ), std::out_of_range ) ;
    };

    TEST( Cuda, CudaDevice_Constructor_Fails_Not_Valid )
    {
        int numErrors = 0;

        CudaDevice device = CudaDevice( 0 );

        EXPECT_TRUE( numErrors == 0 );
    };
}