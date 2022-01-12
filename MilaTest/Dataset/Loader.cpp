#include "gtest/gtest.h"
#include <iostream>
#include <filesystem>

using std::filesystem::current_path;

import Dnn.Data.TextToH5;
import Dnn.Data.BatchSequenceLoader;
import Dnn.Data.OneOfK;

using namespace Mila::Dnn::Data;

namespace Mila::Test::Data
{
    TEST( Dataset, H5Converter )
    {
        std::cout << "Current working directory: " << current_path() << std::endl;

        auto filepath = current_path();
        filepath += "\\Data\\tiny-shakespeare.txt";

        TextToH5 h5Converter = TextToH5();

        h5Converter.Convert( filepath );

        int numErrors = 0;

        EXPECT_TRUE( numErrors != 0 );
    };

    TEST( Dataset, SequenceLoader )
    {
        std::cout << "Current working directory: " << current_path() << std::endl;

        auto filepath = current_path();
        filepath += "\\tiny-shakespeare.h5";

        int batch_size = 10;
        int sequence_length = 10;

        BatchSequenceLoader loader = BatchSequenceLoader(
            DatasetType::training, 
            filepath,
            batch_size, 
            sequence_length );

        int blocks_read = 0;
        while (!loader.EndOfDataset())
        {
            XYPair samples = loader.Next();

            blocks_read++;
        }

        EXPECT_EQ( blocks_read, loader.BlockCount() );
    };

    TEST( Dataset, OneOfK )
    {
        std::vector<int> input = { 2, 3, 1 };
        auto transformer = OneOfK<float>( 3, 1.0 );

        auto output = transformer.Convert( input );
        
        EXPECT_TRUE( output.size() == 9);
    };
}