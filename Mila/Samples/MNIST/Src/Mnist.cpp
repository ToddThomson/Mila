#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <format>
#include <limits>
#include <type_traits>
#include <memory>
#include <ios>
#include <exception>
#include <cstdint>

import Mila;

import Mnist.Classifier;
import Mnist.DataLoader;

namespace fs = std::filesystem;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;
using namespace Mila::Mnist;

struct MnistConfig
{
    std::string data_directory = "./Data/Mnist";
    size_t batch_size = 128;
    size_t epochs = 5;
    float learning_rate = 0.01f;
    DeviceType compute_device = DeviceType::Cuda;
    ComputePrecision::Policy precisionPolicy = ComputePrecision::Policy::Auto;
};

void printUsage()
{
    std::cout << "Usage: mnist [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --data-dir <path>     Path to MNIST data directory (default: ./Data/Mnist)\n";
    std::cout << "  --batch-size <int>    Batch size (default: 128)\n";
    std::cout << "  --epochs <int>        Number of epochs (default: 5)\n";
    std::cout << "  --learning-rate <float> Learning rate (default: 0.01)\n";
    std::cout << "  --device <string>     Compute device (cpu or cuda, default: cuda)\n";
    std::cout << "  --precision <string>  Precision policy (auto, performance, accuracy, disabled, default: auto)\n";
    std::cout << "  --help                Show this help message\n";
}

bool parseCommandLine( int argc, char** argv, MnistConfig& config )
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--help")
        {
            printUsage();
            return false;
        }
        else if (arg == "--data-dir" && i + 1 < argc)
        {
            config.data_directory = argv[++i];
        }
        else if (arg == "--batch-size" && i + 1 < argc)
        {
            config.batch_size = std::stoi( argv[++i] );
        }
        else if (arg == "--epochs" && i + 1 < argc)
        {
            config.epochs = std::stoi( argv[++i] );
        }
        else if (arg == "--learning-rate" && i + 1 < argc)
        {
            config.learning_rate = std::stof( argv[++i] );
        }
        else if (arg == "--device" && i + 1 < argc)
        {
            std::string device = argv[++i];
            if (device == "cpu")
            {
                config.compute_device = DeviceType::Cpu;
            }
            else if (device == "cuda")
            {
                config.compute_device = DeviceType::Cuda;
            }
            else
            {
                std::cerr << "Unknown device type: " << device << ". Using default: cuda" << std::endl;
            }
        }
        else if (arg == "--precision" && i + 1 < argc)
        {
            std::string precision = argv[++i];
            if (precision == "auto")
            {
                config.precisionPolicy = ComputePrecision::Policy::Auto;
            }
            else if (precision == "performance")
            {
                config.precisionPolicy = ComputePrecision::Policy::Performance;
            }
            else if (precision == "accuracy")
            {
                config.precisionPolicy = ComputePrecision::Policy::Accuracy;
            }
            else if (precision == "disabled")
            {
                config.precisionPolicy = ComputePrecision::Policy::Native;
            }
            else
            {
                std::cerr << "Unknown precision policy: " << precision << ". Using default: auto" << std::endl;
            }
        }
        else if (arg.substr( 0, 2 ) == "--")
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage();
            return false;
        }
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Data directory: " << config.data_directory << std::endl;
    std::cout << "  Batch size: " << config.batch_size << std::endl;
    std::cout << "  Epochs: " << config.epochs << std::endl;
    std::cout << "  Learning rate: " << config.learning_rate << std::endl;
    std::cout << "  Device: " << (config.compute_device == DeviceType::Cuda ? "CUDA" : "CPU") << std::endl;
    std::cout << "  Precision policy: ";

    switch (config.precisionPolicy)
    {
        case ComputePrecision::Policy::Auto:
            std::cout << "Auto"; break;
        case ComputePrecision::Policy::Performance:
            std::cout << "Performance"; break;
        case ComputePrecision::Policy::Accuracy:
            std::cout << "Accuracy"; break;
        case ComputePrecision::Policy::Native:
            std::cout << "Disabled"; break;
    }
    std::cout << std::endl << std::endl;

    if (!fs::exists( config.data_directory ))
    {
        std::cerr << "MNIST data directory not found: " << config.data_directory << std::endl;
        std::cerr << "Please download the MNIST dataset from http://yann.lecun.com/exdb/mnist/" << std::endl;
        std::cerr << "and extract the files to " << config.data_directory << std::endl;
        return false;
    }

    return true;
}

template<TensorDataType TDataType>
float softmaxCrossEntropyLoss( const Tensor<TDataType, CpuMemoryResource>& logits,
    const Tensor<TDataType, CpuMemoryResource>& targets )
{
    size_t batch_size = logits.shape()[0];
    size_t num_classes = logits.shape()[1];
    float loss = 0.0f;

    for (size_t i = 0; i < batch_size; ++i)
    {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < num_classes; ++j)
        {
            max_logit = std::max( max_logit, static_cast<float>( logits.data()[i * num_classes + j] ) );
        }

        float denom = 0.0f;
        for (size_t j = 0; j < num_classes; ++j)
        {
            float exp_val = std::exp( static_cast<float>( logits.data()[i * num_classes + j] ) - max_logit );
            denom += exp_val;
        }

        for (size_t j = 0; j < num_classes; ++j)
        {
            float target = static_cast<float>( targets.data()[i * num_classes + j] );
            if (target > 0.0f)
            {
                float prob = std::exp( static_cast<float>( logits.data()[i * num_classes + j] ) - max_logit ) / denom;
                loss += -std::log( prob ) * target;
            }
        }
    }

    return loss / batch_size;
}

template<TensorDataType TDataType, typename MR>
float computeAccuracy( const Tensor<TDataType, MR>& logits,
    const Tensor<TDataType, MR>& targets )
{
    size_t batch_size = logits.shape()[0];
    size_t num_classes = logits.shape()[1];
    size_t correct = 0;

    for (size_t i = 0; i < batch_size; ++i)
    {
        size_t pred_class = 0;
        float max_logit = static_cast<float>( logits.data()[i * num_classes] );

        for (size_t j = 1; j < num_classes; ++j)
        {
            float logit = static_cast<float>( logits.data()[i * num_classes + j] );
            if (logit > max_logit)
            {
                max_logit = logit;
                pred_class = j;
            }
        }

        size_t true_class = 0;
        for (size_t j = 0; j < num_classes; ++j)
        {
            if (targets.data()[i * num_classes + j] > 0.5f)
            {
                true_class = j;
                break;
            }
        }

        if (pred_class == true_class)
        {
            correct++;
        }
    }

    return static_cast<float>(correct) / batch_size;
}

template<TensorDataType TDataType, typename MR>
void sgdUpdate( Tensor<TDataType, MR>& param, Tensor<TDataType, MR>& grad, float learning_rate )
{
    for (size_t i = 0; i < param.size(); ++i)
    {
        param.data()[i] -= learning_rate * grad.data()[i];
    }
}

template<DeviceType TDeviceType, TensorDataType TDataType, typename THostMR>
    requires PrecisionSupportedOnDevice<TDataType, TDeviceType> &&
    (std::is_same_v<THostMR, CudaPinnedMemoryResource> || std::is_same_v<THostMR, CpuMemoryResource>)
void runMnistTrainingLoop(
    std::shared_ptr<MnistClassifier<TDeviceType, TDataType>>& model,
    const MnistConfig& config )
{
    using DeviceMR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;

    // Get device from model's execution context
    auto device = model->getDevice();

    MnistDataLoader<TensorDataType::FP32, THostMR> train_loader( config.data_directory, config.batch_size, true, device );
    MnistDataLoader<TensorDataType::FP32, THostMR> test_loader( config.data_directory, config.batch_size, false, device );

    // Build the model with the input shape from the data loader
    shape_t input_shape = { static_cast<int64_t>(train_loader.batchSize()), MNIST_IMAGE_SIZE };
    model->build( input_shape );

    std::cout << "Model built successfully!" << std::endl;
    std::cout << model->toString() << std::endl;

    Tensor<TDataType, DeviceMR> input_batch( device, input_shape );
    Tensor<TDataType, CpuMemoryResource> target_batch( "CPU", { static_cast<int64_t>(train_loader.batchSize()), MNIST_NUM_CLASSES } );

    Tensor<TDataType, CpuMemoryResource> logits( "CPU", { static_cast<int64_t>(train_loader.batchSize()), MNIST_NUM_CLASSES } );
    Tensor<TDataType, DeviceMR> output( device, { static_cast<int64_t>(train_loader.batchSize()), MNIST_NUM_CLASSES } );

    std::cout << "Starting training for " << config.epochs << " epochs..." << std::endl;

    for (size_t epoch = 0; epoch < config.epochs; ++epoch)
    {
        train_loader.reset();

        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        size_t batches = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        while (train_loader.hasNext())
        {
            train_loader.nextBatch();

            copy( train_loader.inputs(), input_batch );
            copy( train_loader.targets(), target_batch );

            model->forward( input_batch, output );

            copy( output, logits );

            float batch_loss = softmaxCrossEntropyLoss( logits, target_batch );
            float batch_acc = computeAccuracy( logits, target_batch );

            epoch_loss += batch_loss;
            epoch_acc += batch_acc;
            batches++;

            if (batches % 100 == 0 || batches == train_loader.numBatches())
            {
                std::cout << "Epoch " << (epoch + 1) << " [" << batches << "/"
                    << train_loader.numBatches() << "] - Loss: " << std::fixed
                    << std::setprecision( 4 ) << batch_loss << " - Accuracy: "
                    << std::setprecision( 2 ) << (batch_acc * 100.0f) << "%" << std::endl;
            }
        }

        epoch_loss /= batches;
        epoch_acc /= batches;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

        test_loader.reset();
        float test_loss = 0.0f;
        float test_acc = 0.0f;
        size_t test_batches = 0;

        model->setTraining( false );

        while (test_loader.hasNext())
        {
            test_loader.nextBatch();

            copy( test_loader.inputs(), input_batch );
            copy( test_loader.targets(), target_batch );

            model->forward( input_batch, output );

            copy( output, logits );

            test_loss += softmaxCrossEntropyLoss( logits, target_batch );
            test_acc += computeAccuracy( logits, target_batch );

            test_batches++;
        }

        test_loss /= test_batches;
        test_acc /= test_batches;

        model->setTraining( true );

        std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs
            << " - Time: " << epoch_time << "s - Loss: " << std::fixed << std::setprecision( 4 ) << epoch_loss
            << " - Accuracy: " << std::setprecision( 2 ) << (epoch_acc * 100.0f) << "%"
            << " - Test Loss: " << std::setprecision( 4 ) << test_loss
            << " - Test Accuracy: " << std::setprecision( 2 ) << (test_acc * 100.0f) << "%"
            << std::endl;
    }
}

int main( int argc, char** argv )
{
    try
    {
        std::cout << "MNIST Classification Example using Mila MLP" << std::endl;
        std::cout << "===========================================" << std::endl;

        Mila::initialize();

        MnistConfig config;
        if (!parseCommandLine( argc, argv, config ))
        {
            return 1;
        }

        if (config.compute_device == DeviceType::Cuda)
        {
            try
            {
                std::cout << "Using CUDA device" << std::endl;

                // Create execution context
                auto exec_context = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

                // Create model with execution context
                auto model = std::make_shared<CudaMnistClassifier<TensorDataType::FP32>>(
                    exec_context,
                    "MnistMLP",
                    static_cast<int64_t>(config.batch_size) );

                model->setTraining( true );

                runMnistTrainingLoop<DeviceType::Cuda, TensorDataType::FP32, CudaPinnedMemoryResource>(
                    model, config );
            }
            catch (const std::exception& e)
            {
                std::cerr << "CUDA error: " << e.what() << ", falling back to CPU" << std::endl;
                config.compute_device = DeviceType::Cpu;
            }
        }

        if (config.compute_device == DeviceType::Cpu)
        {
            std::cout << "Using CPU device" << std::endl;

            // Create execution context
            auto exec_context = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

            // Create model with execution context
            auto model = std::make_shared<CpuMnistClassifier<TensorDataType::FP32>>(
                exec_context,
                "MnistMLP",
                static_cast<int64_t>(config.batch_size) );

            model->setTraining( true );

            runMnistTrainingLoop<DeviceType::Cpu, TensorDataType::FP32, CpuMemoryResource>(
                model, config );
        }

        std::cout << "Training complete!" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}