#include "gtest/gtest.h"

#include <vector>
#include <iostream>
#include "Common.h"

import Dnn.Model;
import Dnn.TensorDescriptor;
import CuDnn.Utils;

namespace Mila::Tests::Dnn
{
    TEST( Dnn_Descriptor, Creates_Tensor_Float )
    {
        auto model = Mila::Dnn::DnnModel();

        auto builder = model.GetModelBuilder();

        auto options = GetDefaultRnnOptions();

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
                << "Status: " << Mila::Dnn::CuDNN::to_string( status ) << std::endl
                << "Error: " << tensor.get_error();
        }

        EXPECT_TRUE( status == CUDNN_STATUS_SUCCESS );
    };
}