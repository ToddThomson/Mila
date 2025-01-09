#include "gtest/gtest.h"
#include <iostream>
#include <filesystem>

using std::filesystem::current_path;

import Data.CharDatasetGenerator;
import Data.Dataset;
import Data.DatasetType;

using namespace Mila::Dnn::Data;

namespace Mila::Test::Data
{
    TEST( Dataset, GenerateDataset )
    {
        std::cout << "Current working directory: " << current_path() << std::endl;

        auto filepath = current_path();
        filepath += "\\Data\\tiny-shakespeare.txt";

        CharDatasetGenerator dataset_generator = CharDatasetGenerator( filepath );

        dataset_generator.GenerateDataset();
    };
}