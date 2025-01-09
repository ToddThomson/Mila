#include "gtest/gtest.h"

import Cuda;

namespace Mila::Test::Cuda
{
    using namespace Mila::Dnn::Cuda;

    TEST( Cuda, CudaHelpers_FindCudaDevice_Best_Passes )
    {
        int device_id;
        int device_count;

        EXPECT_NO_THROW( device_count = GetDeviceCount() );
        EXPECT_NO_THROW( device_id = FindCudaDevice() );

        EXPECT_TRUE( (device_id >= 0) && (device_id < device_count) );
    };

    TEST( Cuda, CudaHelpers_FindCudaDevice_With_Id_Passes )
    {
        int device_id;
        int device_count;

        EXPECT_NO_THROW( device_count = GetDeviceCount() );

        // TJT: Randomize or check all?
        EXPECT_NO_THROW( device_id = FindCudaDevice( 0 ) );

        EXPECT_TRUE( device_id == 0 );
    };
}