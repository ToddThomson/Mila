#include "gtest/gtest.h"
#include <iostream>
#include <filesystem>

using std::filesystem::current_path;

import Data.Dataset;
import Data.DatasetType;

using namespace Mila::Dnn::Data;

namespace Mila::Test::Data
{
    TEST( Dataset, Dataset_ReadsAllTrainingBlocks )
    {
        std::cout << "Current working directory: " << current_path() << std::endl;

        auto filepath = current_path();
        filepath += "\\tiny-shakespeare.h5";

        int batch_size = 10;
        int sequence_length = 10;

        Dataset dataset = Dataset( filepath, batch_size, sequence_length );

        dataset.Load( DatasetType::training );

        int blocks_read = 0;
        while ( !dataset.EndOfDataset() )
        {
            XYPair samples = dataset.NextBlock();

            blocks_read++;
        }

        EXPECT_EQ( blocks_read, dataset.BlockCount() );
    };
}