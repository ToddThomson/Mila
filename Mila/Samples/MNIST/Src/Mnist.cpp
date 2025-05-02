#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>

import Mila;

import Mnist.Classifier;
import Mnist.DataLoader;

namespace fs = std::filesystem;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;
using namespace Mila::Mnist;

// Cross-entropy loss function for classification
template<typename TPrecision, typename MR>
float computeCrossEntropyLoss( const Tensor<TPrecision, MR>& logits,
    const Tensor<TPrecision, MR>& targets ) {
    
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

int main( int argc, char** argv ) {
    try {
        std::cout << "MNIST Classification Example using Mila MLP" << std::endl;
        std::cout << "===========================================" << std::endl;

		Mila::initialize();

        const std::string MNIST_DIR = "./Data/Mnist";
        const size_t BATCH_SIZE = 128;
        const size_t EPOCHS = 5;
        const float LEARNING_RATE = 0.01f;
		const DeviceType COMPUTE_DEVICE_TYPE = DeviceType::Cuda;

        if ( !fs::exists( MNIST_DIR ) ) {
            std::cerr << "MNIST data directory not found: " << MNIST_DIR << std::endl;
            std::cerr << "Please download the MNIST dataset from http://yann.lecun.com/exdb/mnist/" << std::endl;
            std::cerr << "and extract the files to " << MNIST_DIR << std::endl;
            return 1;
        }

        // Select device type (CPU or CUDA)
    #ifdef CUDA_AVAILABLE
        constexpr DeviceType DEVICE_TYPE = DeviceType::Cuda;
        const std::string DEVICE_NAME = "CUDA:0";
        using MemResourceType = CudaMemoryResource;
        std::cout << "Using CUDA device" << std::endl;
    #else
        constexpr DeviceType DEVICE_TYPE = DeviceType::Cpu;
        const std::string DEVICE_NAME = "CPU";
        using MemResourceType = HostMemoryResource;
        std::cout << "Using CPU device" << std::endl;
    #endif

        // Create data loaders
        std::cout << "Loading MNIST dataset..." << std::endl;
        MnistDataLoader<float, DEVICE_TYPE> train_loader( MNIST_DIR, BATCH_SIZE, true );
        MnistDataLoader<float, DEVICE_TYPE> test_loader( MNIST_DIR, BATCH_SIZE, false );

        // Create model
        std::cout << "Creating model..." << std::endl;
        auto model = std::make_shared<MnistClassifier<float, DEVICE_TYPE>>(
            "MnistMLP", DEVICE_NAME, BATCH_SIZE );

        // Enable training mode
        model->setTraining( true );

        // Print model architecture
        std::cout << model->toString() << std::endl;

        // Prepare tensors for training
        Tensor<float, MemResourceType> input_batch;
        Tensor<float, MemResourceType> target_batch;
        Tensor<float, MemResourceType> output_batch;

        // Training loop
        std::cout << "Starting training for " << EPOCHS << " epochs..." << std::endl;

        for ( size_t epoch = 0; epoch < EPOCHS; ++epoch ) {
            // Reset loaders
            train_loader.reset();

            // Track statistics
            float epoch_loss = 0.0f;
            float epoch_acc = 0.0f;
            size_t batches = 0;

            auto start_time = std::chrono::high_resolution_clock::now();

            // Training batches
            while ( train_loader.nextBatch( input_batch, target_batch ) ) {
                if ( batches == 469 ) {
                    std::cout << "Last batch" << std::endl;
                }

                // Reset the output tensor with appropriate shape
                output_batch = Tensor<float, MemResourceType>( { input_batch.shape()[ 0 ], static_cast<size_t>(MNIST_NUM_CLASSES) } );

                // Forward pass
                model->forward( input_batch, output_batch );

                // Compute loss and accuracy
                float batch_loss = computeCrossEntropyLoss( output_batch, target_batch );
                float batch_acc = computeAccuracy( output_batch, target_batch );

                epoch_loss += batch_loss;
                epoch_acc += batch_acc;
                batches++;

                // Here in a real implementation, you would:
                // 1. Compute gradients with backward pass
                // 2. Update parameters using optimizer

                // Print progress every 100 batches or for the last batch
                if ( batches % 100 == 0 || batches + 1 == train_loader.numBatches() ) {
                    std::cout << "Epoch " << (epoch + 1) << " [" << batches << "/"
                        << train_loader.numBatches() << "] - Loss: " << std::fixed
                        << std::setprecision( 4 ) << batch_loss << " - Accuracy: "
                        << std::setprecision( 2 ) << (batch_acc * 100.0f) << "%" << std::endl;
                }
            }

            std::cout << "Finished training batches" << std::endl;

            // Calculate epoch statistics
            epoch_loss /= batches;
            epoch_acc /= batches;

            auto end_time = std::chrono::high_resolution_clock::now();
            auto epoch_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

            // Evaluation on test set
            test_loader.reset();
            float test_loss = 0.0f;
            float test_acc = 0.0f;
            size_t test_batches = 0;

            // Switch to evaluation mode
            model->setTraining( false );

            // Test loop
            while ( test_loader.nextBatch( input_batch, target_batch ) ) {
                output_batch = Tensor<float, MemResourceType>( { input_batch.shape()[ 0 ], static_cast<size_t>(MNIST_NUM_CLASSES) } );

                // Forward pass only
                model->forward( input_batch, output_batch );

                // Compute metrics
                test_loss += computeCrossEntropyLoss( output_batch, target_batch );
                test_acc += computeAccuracy( output_batch, target_batch );
                test_batches++;
            }

            // Calculate test statistics
            test_loss /= test_batches;
            test_acc /= test_batches;

            // Switch back to training mode
            model->setTraining( true );

            // Print epoch summary
            std::cout << "Epoch " << (epoch + 1) << "/" << EPOCHS
                << " - Time: " << epoch_time << "s - Loss: " << std::fixed << std::setprecision( 4 ) << epoch_loss
                << " - Accuracy: " << std::setprecision( 2 ) << (epoch_acc * 100.0f) << "%"
                << " - Test Loss: " << std::setprecision( 4 ) << test_loss
                << " - Test Accuracy: " << std::setprecision( 2 ) << (test_acc * 100.0f) << "%"
                << std::endl;
        }

        std::cout << "Training complete!" << std::endl;

    }
    catch ( const std::exception& e ) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}