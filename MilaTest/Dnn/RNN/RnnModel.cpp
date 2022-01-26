#include "gtest/gtest.h"
#include "../Common.h"
#include <iostream>

import Dnn.RnnModel;
import CuDnn.Utils;

namespace Mila::Tests::Dnn
{
    TEST( Dnn_Model, Builds_Default_RnnModel_Float )
    {
        auto options = GetDefaultRnnOptions();
        auto model = Mila::Dnn::RnnModel<float>( options );

        model.BuildModel();
        model.Train();

        std::cout << "RnnModel::rnnOp member:" << std::endl
            << model.GetRnnOp().ToString() << std::endl;
        
        auto status = model.GetRnnOp().get_status();
        auto error = model.GetRnnOp().get_error();

        if ( status == CUDNN_STATUS_SUCCESS )
        {
            std::cout << std::endl << "Test passed successfully." << std::endl;
        }
        else
        {
            std::cout << std::endl << "Test Failed!" << std::endl
                << "Status: " << Mila::Dnn::CuDnn::to_string( status ) << std::endl
                << "Error: " << error << std::endl;
        }

        EXPECT_TRUE( status == CUDNN_STATUS_SUCCESS );
    };
}