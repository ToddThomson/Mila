#include "gtest/gtest.h"
#include "Common.h"
#include <iostream>

import Dnn.RnnModel;
import Dnn.DropoutDescriptor;
import CuDnn.Utils;

namespace Mila::Tests::Dnn
{
    class DropoutTestModel : public Mila::Dnn::RnnModel<float>
    {
    public:

        DropoutTestModel() : Mila::Dnn::RnnModel<float>( GetDefaultRnnOptions() )
        {
        }

        void OnModelBuilding( const DnnModelBuilder& builder ) override
        {
            dropout_ = builder.Create<Dropout>();

            dropout_.SetProbability( 0.1f )
                .SetSeed( 739134ull );

            dropout_.Finalize();
        }

        const Dropout& GetDropout()
        {
            return dropout_;
        }

    private:

        Dropout dropout_;
    };
    
    TEST( Dnn_Operations, Create_Dropout )
    {
        DropoutTestModel app = DropoutTestModel();

        app.BuildModel();

        const Dropout& dropout = app.GetDropout();
        auto status = dropout.get_status();

        std::cout << dropout.ToString() << std::endl;

        if ( status != CUDNN_STATUS_SUCCESS )
        {
            std::cout << std::endl << "Test Failed!" << std::endl
                << "Status: " << Mila::Dnn::CuDnn::to_string( status ) << std::endl
                << "Error: " << dropout.get_error();
        }

        EXPECT_TRUE( status == CUDNN_STATUS_SUCCESS );
    };
}