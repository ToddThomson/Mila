#include "gtest/gtest.h"

#include <vector>
#include <iostream>
#include "Common.h"

import Dnn.RnnModel;
import Dnn.TensorDescriptor;
import CuDnn.Utils;

namespace Mila::Tests::Dnn
{
    TEST( Dnn_Descriptor, Creates_Tensor_Float )
    {
        auto options = GetDefaultRnnOptions();
        auto model = Mila::Dnn::RnnModel<float>( options );

        auto builder = model.GetModelBuilder();

        cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
        int dimensions = 3;
        std::vector<int> shape = { 1, 2, 3 };
        std::vector<int> strides = { 1, 2, 3 };

        Tensor tensor = builder.Create<Tensor>();

        tensor.SetDataType( dataType )
            .SetDimensions( 3, shape, strides );

        tensor.Finalize();

        auto status = tensor.get_status();

        if ( status != CUDNN_STATUS_SUCCESS )
        {
            std::cout << std::endl << "Test Failed!" << std::endl
                << "Status: " << Mila::Dnn::CuDnn::to_string( status ) << std::endl
                << "Error: " << tensor.get_error();
        }

        EXPECT_TRUE( status == CUDNN_STATUS_SUCCESS );
    };
}