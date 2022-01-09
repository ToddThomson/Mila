#include "gtest/gtest.h"
#include "../Common.h"
#include <iostream>
#include <vector>

import Dnn.Model;
import Dnn.RnnDataSet;
import CuDnn.Utils;

namespace Mila::Tests::Dnn
{
    TEST( Dnn_Descriptor, Create_RnnDataSet_Float )
    {
        auto model = Mila::Dnn::DnnModel();

        auto options = GetDefaultRnnOptions();

        cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

        double paddingFill = 0.0;
        std::vector<int> seqLengthArray;

        for ( int i = 0; i < options.batchSize; i++ )
        {
            seqLengthArray.push_back( options.sequenceLength );
        }

        auto builder = model.GetModelBuilder();

        RnnDataSet xDesc = builder.Create<RnnDataSet>();

        xDesc.SetDataType( dataType )
            .SetLayout( CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED )
            .SetBatchSize( options.batchSize )
            .SetMaxSequenceLength( options.sequenceLength )
            .SetVectorSize( options.vectorSize )
            .SetSequenceLengthArray( seqLengthArray )
            .SetPaddingFill( paddingFill );

        xDesc.Finalize();

        auto status = xDesc.get_status();

        std::cout << "RNN data set, input x tensor: " << std::endl
            << xDesc.ToString() << std::endl
            << "Status: " << Mila::Dnn::CuDNN::to_string( status ) << std::endl
            << "Error Msg: " << xDesc.get_error() << std::endl;

        EXPECT_TRUE( status == CUDNN_STATUS_SUCCESS );
    };
}