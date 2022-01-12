#include <iostream>

import Mila;

int main()
{
    std::cout << "Mila API Version: "
        << Mila::GetAPIVersion().ToString() << std::endl;
    std::cout << "CuDNN Version: "
        << Mila::Dnn::CuDNN::GetVersion().ToString() << std::endl;
}