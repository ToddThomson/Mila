#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <format>

import Mila;

import Mnist.Classifier;
import Mnist.DataLoader;

namespace fs = std::filesystem;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;
using namespace Mila::Mnist;

struct MnistConfig {
    std::string dataDir = "./Data/Mnist";
    size_t batchSize = 128;
    size_t epochs = 5;
    float learningRate = 0.01f;
    DeviceType computeDeviceType = DeviceType::Cuda;
};

void printUsage() {
    std::cout << "Usage: mnist [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --data-dir <path>     Path to MNIST data directory (default: ./Data/Mnist)\n";
    std::cout << "  --batch-size <int>    Batch size (default: 128)\n";
    std::cout << "  --epochs <int>        Number of epochs (default: 5)\n";
    std::cout << "  --learning-rate <float> Learning rate (default: 0.01)\n";
    std::cout << "  --device <string>     Compute device (cpu or cuda, default: cuda)\n";
    std::cout << "  --help                Show this help message\n";
}

// Parse command line arguments and fill the configuration
bool parseCommandLine( int argc, char** argv, MnistConfig& config ) {
    for ( int i = 1; i < argc; i++ ) {
        std::string arg = argv[ i ];

        if ( arg == "--help" ) {
            printUsage();
            return false;
        }
        else if ( arg == "--data-dir" && i + 1 < argc ) {
            config.dataDir = argv[ ++i ];
        }
        else if ( arg == "--batch-size" && i + 1 < argc ) {
            config.batchSize = std::stoi( argv[ ++i ] );
        }
        else if ( arg == "--epochs" && i + 1 < argc ) {
            config.epochs = std::stoi( argv[ ++i ] );
        }
        else if ( arg == "--learning-rate" && i + 1 < argc ) {
            config.learningRate = std::stof( argv[ ++i ] );
        }
        else if ( arg == "--device" && i + 1 < argc ) {
            std::string device = argv[ ++i ];
            if ( device == "cpu" ) {
                config.computeDeviceType = DeviceType::Cpu;
            }
            else if ( device == "cuda" ) {
                config.computeDeviceType = DeviceType::Cuda;
            }
            else {
                std::cerr << "Unknown device type: " << device << ". Using default: cuda" << std::endl;
            }
        }
        else if ( arg.substr( 0, 2 ) == "--" ) {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage();
            return false;
        }
    }

    // Print configuration
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Data directory: " << config.dataDir << std::endl;
    std::cout << "  Batch size: " << config.batchSize << std::endl;
    std::cout << "  Epochs: " << config.epochs << std::endl;
    std::cout << "  Learning rate: " << config.learningRate << std::endl;
    std::cout << "  Device: " << (config.computeDeviceType == DeviceType::Cuda ? "CUDA" : "CPU") << std::endl;
    std::cout << std::endl;

    // Validate data directory
    if ( !fs::exists( config.dataDir ) ) {
        std::cerr << "MNIST data directory not found: " << config.dataDir << std::endl;
        std::cerr << "Please download the MNIST dataset from http://yann.lecun.com/exdb/mnist/" << std::endl;
        std::cerr << "and extract the files to " << config.dataDir << std::endl;
        return false;
    }

    return true;
}

// Cross-entropy loss function for classification
template<typename TPrecision>
float computeCrossEntropyLoss( const Tensor<TPrecision, HostMemoryResource>& logits, const Tensor<TPrecision, HostMemoryResource>& targets ) {

    // Apply softmax and compute cross-entropy loss
    size_t batch_size = logits.shape()[ 0 ];
    size_t num_classes = logits.shape()[ 1 ];
    float loss = 0.0f;

    for ( size_t i = 0; i < batch_size; ++i ) {
        // Find max logit for numerical stability
        float max_logit = -std::numeric_limits<float>::infinity();
        for ( size_t j = 0; j < num_classes; ++j ) {
            max_logit = std::max( max_logit, static_cast<float>( logits.data()[ i * num_classes + j ] ) );
        }

        // Compute softmax denominator (sum of exp(logit - max_logit))
        float denom = 0.0f;
        for ( size_t j = 0; j < num_classes; ++j ) {
            float exp_val = std::exp( static_cast<float>( logits.data()[ i * num_classes + j ] ) - max_logit );
            denom += exp_val;
        }

        // Compute cross-entropy loss for each sample
        for ( size_t j = 0; j < num_classes; ++j ) {
            float target = static_cast<float>( targets.data()[ i * num_classes + j ] );
            if ( target > 0.0f ) {  // Only calculate for the true class (one-hot encoding)
                float prob = std::exp( static_cast<float>( logits.data()[ i * num_classes + j ] ) - max_logit ) / denom;
                loss += -std::log( prob ) * target;
            }
        }
    }

    return loss / batch_size;
}

// Compute accuracy for classification
template<typename TPrecision, typename MR>
float computeAccuracy( const Tensor<TPrecision, MR>& logits,
    const Tensor<TPrecision, MR>& targets ) {
    size_t batch_size = logits.shape()[ 0 ];
    size_t num_classes = logits.shape()[ 1 ];
    size_t correct = 0;

    for ( size_t i = 0; i < batch_size; ++i ) {
        // Find predicted class (max logit)
        size_t pred_class = 0;
        float max_logit = static_cast<float>( logits.data()[ i * num_classes ] );

        for ( size_t j = 1; j < num_classes; ++j ) {
            float logit = static_cast<float>( logits.data()[ i * num_classes + j ] );
            if ( logit > max_logit ) {
                max_logit = logit;
                pred_class = j;
            }
        }

        // Find true class (one-hot encoded)
        size_t true_class = 0;
        for ( size_t j = 0; j < num_classes; ++j ) {
            if ( targets.data()[ i * num_classes + j ] > 0.5f ) {
                true_class = j;
                break;
            }
        }

        // Check if prediction was correct
        if ( pred_class == true_class ) {
            correct++;
        }
    }

    return static_cast<float>(correct) / batch_size;
}

// Simple SGD optimizer for demonstration
template<typename TPrecision, typename MR>
void sgdUpdate( Tensor<TPrecision, MR>& param, Tensor<TPrecision, MR>& grad, float learning_rate ) {
    for ( size_t i = 0; i < param.size(); ++i ) {
        param.data()[ i ] -= learning_rate * grad.data()[ i ];
    }
}

template<typename TPrecision, typename THostMR, DeviceType TDeviceType>
    requires ValidFloatTensorType<TPrecision> &&
        (std::is_same_v<THostMR, CudaPinnedMemoryResource> || std::is_same_v<THostMR, CpuMemoryResource>)
void runMnistTrainingLoop(
    std::shared_ptr<MnistClassifier<TPrecision, TDeviceType>>& model,
    const MnistConfig& config ) {
	
    using DeviceMR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, HostMemoryResource>;

    MnistDataLoader<float, THostMR> train_loader( config.dataDir, config.batchSize, true );
    MnistDataLoader<float, THostMR> test_loader( config.dataDir, config.batchSize, false );

    Tensor<TPrecision, DeviceMR> input_batch( { train_loader.batchSize(), static_cast<size_t>(MNIST_IMAGE_SIZE) } );
    Tensor<TPrecision, HostMemoryResource> target_batch( { train_loader.batchSize(), static_cast<size_t>(MNIST_NUM_CLASSES) } );

	Tensor<float, HostMemoryResource> logits( { train_loader.batchSize(), static_cast<size_t>(MNIST_NUM_CLASSES) } );
	Tensor<float, DeviceMR> output( { train_loader.batchSize(), static_cast<size_t>(MNIST_NUM_CLASSES) } );

    std::cout << "Starting training for " << config.epochs << " epochs..." << std::endl;

    for ( size_t epoch = 0; epoch < config.epochs; ++epoch ) {
        train_loader.reset();

        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        size_t batches = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        while ( batches < train_loader.numBatches() ) {
            train_loader.nextBatch();
            
            input_batch.copyFrom( train_loader.inputs() );
            target_batch.copyFrom( train_loader.targets() );

            model->forward( input_batch, output );

			logits.copyFrom( output );

            float batch_loss = computeCrossEntropyLoss( logits, target_batch );
            float batch_acc = computeAccuracy( logits, target_batch );

            epoch_loss += batch_loss;
            epoch_acc += batch_acc;
            batches++;

            if ( batches % 100 == 0 || batches == train_loader.numBatches() ) {
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

        // Switch to evaluation mode
        model->setTraining( false );

        while ( test_batches < test_loader.numBatches() ) {
            test_loader.nextBatch();

            input_batch.copyFrom( test_loader.inputs() );
            target_batch.copyFrom( test_loader.targets() );

            model->forward( input_batch, output );

            logits.copyFrom( output );

            test_loss += computeCrossEntropyLoss( logits, target_batch );
            test_acc += computeAccuracy( logits, target_batch );
            
            test_batches++;
        }

        // Calculate test statistics
        test_loss /= test_batches;
        test_acc /= test_batches;

        model->setTraining( true );

        // Print epoch summary
        std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs
            << " - Time: " << epoch_time << "s - Loss: " << std::fixed << std::setprecision( 4 ) << epoch_loss
            << " - Accuracy: " << std::setprecision( 2 ) << (epoch_acc * 100.0f) << "%"
            << " - Test Loss: " << std::setprecision( 4 ) << test_loss
            << " - Test Accuracy: " << std::setprecision( 2 ) << (test_acc * 100.0f) << "%"
            << std::endl;
    }
}

int main( int argc, char** argv ) {
    try {
        std::cout << "MNIST Classification Example using Mila MLP" << std::endl;
        std::cout << "===========================================" << std::endl;

        Mila::initialize();

        MnistConfig config;
        if ( !parseCommandLine( argc, argv, config ) ) {
            return 1;
        }

        if ( config.computeDeviceType == DeviceType::Cuda ) {
            try {
                std::string deviceName = "CUDA:0";
                std::cout << "Using CUDA device" << std::endl;

                auto model = std::make_shared<MnistClassifier<float, DeviceType::Cuda>>(
                    "MnistMLP", deviceName, config.batchSize );

                model->setTraining( true );

                std::cout << model->toString() << std::endl;
                
                runMnistTrainingLoop<float, CudaPinnedMemoryResource, DeviceType::Cuda>( model, config );
            }
            catch ( const std::exception& e ) {
                std::cerr << "CUDA error: " << e.what() << ", falling back to CPU" << std::endl;
                config.computeDeviceType = DeviceType::Cpu;
            }
        }

        if ( config.computeDeviceType == DeviceType::Cpu ) {
            std::string deviceName = "CPU";
            std::cout << "Using CPU device" << std::endl;

            std::cout << "Creating model..." << std::endl;
            auto model = std::make_shared<MnistClassifier<float, DeviceType::Cpu>>(
                "MnistMLP", deviceName, config.batchSize );
            
            model->setTraining( true );

            std::cout << model->toString() << std::endl;
            
            runMnistTrainingLoop<float, HostMemoryResource, DeviceType::Cpu>(
                model, config );
        }

        std::cout << "Training complete!" << std::endl;
    }
    catch ( const std::exception& e ) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}