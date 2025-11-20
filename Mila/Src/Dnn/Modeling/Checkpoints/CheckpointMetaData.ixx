module;
#include <cstddef>
#include <chrono>
#include <filesystem>

export module Modeling.CheckpointMetaData;

namespace Mila::Dnn::Modeling
{
    export struct CheckpointMetadata
    {
        std::size_t epoch;
        double train_loss;
        double val_loss;
        std::chrono::system_clock::time_point timestamp;
        std::filesystem::path filepath;
    };
}