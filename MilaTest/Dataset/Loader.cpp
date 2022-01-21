#include "gtest/gtest.h"
#include <iostream>
#include <filesystem>

using std::filesystem::current_path;

import Data.TextToDataset;
import Data.DatasetLoader;
import Data.CategoryToVectorEncoder;

using namespace Mila::Dnn::Data;

namespace Mila::Test::Data
{
    TEST( Dataset, TextToDataset )
    {
        std::cout << "Current working directory: " << current_path() << std::endl;

        auto filepath = current_path();
        filepath += "\\Data\\tiny-shakespeare.txt";

        TextToDataset h5Converter = TextToDataset( filepath );

        h5Converter.CreateDataset();

        //EXPECT_TRUE( numErrors != 0 );
    };

    TEST( Dataset, SequenceLoader )
    {
        std::cout << "Current working directory: " << current_path() << std::endl;

        auto filepath = current_path();
        filepath += "\\tiny-shakespeare.h5";

        int batch_size = 10;
        int sequence_length = 10;

        DatasetLoader loader = DatasetLoader(
            DatasetType::training, 
            filepath,
            batch_size, 
            sequence_length );

        int blocks_read = 0;
        while (!loader.EndOfDataset())
        {
            XYPair samples = loader.NextBlock();

            blocks_read++;
        }

        EXPECT_EQ( blocks_read, loader.BlockCount() );
    };

    TEST( Dataset, CategoryVector )
    {
        std::vector<int> input = { 2, 3, 1 };
        auto transformer = CategoryToVectorEncoder<float>( 3, 1.0 );

        auto output = transformer.Convert( input );
        
        EXPECT_TRUE( output.size() == 9);
    };
}