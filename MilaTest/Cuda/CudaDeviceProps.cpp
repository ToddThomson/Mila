#include "gtest/gtest.h"

import Cuda;

namespace Mila::Test::Cuda
{
    using namespace Mila::Dnn::Cuda;

    TEST( Cuda, CudaDeviceProps_Passes )
    {
        EXPECT_NO_THROW( CudaDeviceProps props = CudaDeviceProps( 0 ) );
    };

    TEST( Cuda, CudaDeviceProps_GetTotalGlobalMem_Passes )
    {
        CudaDeviceProps props = CudaDeviceProps( 0 );

        EXPECT_GT( props.GetTotalGlobalMem(), 0 );
    };

    TEST( Cuda, CudaDeviceProps_GetComputeCaps_Passes )
    {
        CudaDeviceProps props = CudaDeviceProps( 0 );
        std::pair<int,int> caps = props.GetComputeCaps();
        
        EXPECT_GT( caps.first, 4 );
        EXPECT_TRUE( caps.second >= 0 );
    };

    TEST( Cuda, CudaDeviceProps_ToString_Passes )
    {
        CudaDeviceProps props = CudaDeviceProps( 0 );

        EXPECT_NO_THROW( props.ToString() );
    };

}