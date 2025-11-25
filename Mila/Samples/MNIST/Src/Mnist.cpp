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
#include <stdexcept>

import Mila;

import Mnist.Classifier;
import Mnist.DataLoader;

namespace fs = std::filesystem;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;
using namespace Mila::Dnn::Optimizers;
using namespace Mila::Mnist;

struct MnistConfig
{
    std::string data_directory = "./Data/DataSets/Mnist";
    int64_t batch_size = 128;
    size_t epochs = 5;
    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.01f;
    DeviceType compute_device = DeviceType::Cuda;
	TensorDataType precision = TensorDataType::FP32;
    ComputePrecision::Policy precisionPolicy = ComputePrecision::Policy::Auto;
};

void printUsage()
{
    std::cout << "Usage: mnist [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --data-dir <path>            Path to MNIST data directory (default: ./Data/DataSets/Mnist)\n";
    std::cout << "  --batch-size <int>           Batch size (default: 128)\n";
    std::cout << "  --epochs <int>               Number of epochs (default: 5)\n";
    std::cout << "  --learning-rate <float>      Learning rate (default: 0.001)\n";
    std::cout << "  --beta1 <float>              Adam beta1 parameter (default: 0.9)\n";
    std::cout << "  --beta2 <float>              Adam beta2 parameter (default: 0.999)\n";
    std::cout << "  --weight-decay <float>       Weight decay (default: 0.01)\n";
    std::cout << "  --device <string>            Compute device (cpu or cuda, default: cuda)\n";
	std::cout << "  --precision <string>         Precision ( FP32, FP16, etc.)\n";
    std::cout << "  --precision-policy <string>  Precision policy (auto, performance, accuracy, disabled, default: auto)\n";
    std::cout << "  --help                       Show this help message\n";
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
        else if (arg == "--beta1" && i + 1 < argc)
        {
            config.beta1 = std::stof( argv[++i] );
        }
        else if (arg == "--beta2" && i + 1 < argc)
        {
            config.beta2 = std::stof( argv[++i] );
        }
        else if (arg == "--weight-decay" && i + 1 < argc)
        {
            config.weight_decay = std::stof( argv[++i] );
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
        else if (arg == "--precision-policy" && i + 1 < argc)
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
    std::cout << "  Beta1: " << config.beta1 << std::endl;
    std::cout << "  Beta2: " << config.beta2 << std::endl;
    std::cout << "  Weight decay: " << config.weight_decay << std::endl;
    std::cout << "  Device: " << (config.compute_device == DeviceType::Cuda ? "CUDA" : "CPU") << std::endl;
    // FIXME: std::cout << "  Precision policy: " << config.precisionPolicy.toString() << std::endl;

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
void softmaxCrossEntropyGradient(
    const Tensor<TDataType, CpuMemoryResource>& logits,
    const Tensor<TDataType, CpuMemoryResource>& targets,
    Tensor<TDataType, CpuMemoryResource>& output_grad )
{
    using HostType = typename TensorHostTypeMap<TDataType>::host_type;

    size_t batch_size = logits.shape()[0];
    size_t num_classes = logits.shape()[1];

    for (size_t i = 0; i < batch_size; ++i)
    {
        // Numerical stability: subtract max
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < num_classes; ++j)
        {
            max_logit = std::max( max_logit,
                static_cast<float>( logits.data()[i * num_classes + j] ) );
        }

        // Compute softmax denominator
        float denom = 0.0f;
        for (size_t j = 0; j < num_classes; ++j)
        {
            float exp_val = std::exp(
                static_cast<float>( logits.data()[i * num_classes + j] ) - max_logit );
            denom += exp_val;
        }

        // Gradient: softmax(logits) - targets
        for (size_t j = 0; j < num_classes; ++j)
        {
            float prob = std::exp(
                static_cast<float>( logits.data()[i * num_classes + j] ) - max_logit ) / denom;
            float target = static_cast<float>( targets.data()[i * num_classes + j] );

            // dL/dlogit = (prob - target) / batch_size
            output_grad.data()[i * num_classes + j] =
                static_cast<HostType>( (prob - target) / static_cast<float>( batch_size ) );
        }
    }
}

template<TensorDataType TDataType>
float softmaxCrossEntropyLoss( 
    const Tensor<TDataType, CpuMemoryResource>& logits,
    const Tensor<TDataType, CpuMemoryResource>& targets )
{
    size_t batch_size = logits.shape()[0];
    size_t num_classes = logits.shape()[1];
    float loss = 0.0f;

    for (size_t i = 0; i < batch_size; ++i)
    {
        // ============================================================
        // SOFTMAX: Numerical stability with max subtraction
        // ============================================================

        // Find max logit for numerical stability (prevents overflow in exp)
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < num_classes; ++j)
        {
            max_logit = std::max( max_logit, static_cast<float>( logits.data()[i * num_classes + j] ) );
        }

        // ============================================================
        // SOFTMAX: Compute denominator (sum of exp(logit - max))
        // ============================================================

        // Compute sum of exp(logit - max_logit) for softmax denominator
        float denom = 0.0f;
        for (size_t j = 0; j < num_classes; ++j)
        {
            float exp_val = std::exp( static_cast<float>( logits.data()[i * num_classes + j] ) - max_logit );
            denom += exp_val;
        }

        // ============================================================
        // CROSS-ENTROPY: Compute -sum(target * log(softmax(logit)))
        // ============================================================

        // For each class, compute cross-entropy contribution
        for (size_t j = 0; j < num_classes; ++j)
        {
            float target = static_cast<float>( targets.data()[i * num_classes + j] );
            
            if (target > 0.0f)
            {
                // SOFTMAX: prob = exp(logit - max) / denom
                float prob = std::exp( static_cast<float>(logits.data()[i * num_classes + j]) - max_logit ) / denom;

                // CROSS-ENTROPY: loss += -target * log(prob)
                loss += -std::log( prob ) * target;
            }
        }
    }

    // Average loss over batch
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

template<DeviceType TDeviceType, TensorDataType TPrecision, typename THostMR>
    requires PrecisionSupportedOnDevice<TPrecision, TDeviceType> &&
(std::is_same_v<THostMR, CudaPinnedMemoryResource> || std::is_same_v<THostMR, CpuMemoryResource>)
void trainMnist( const MnistConfig& config )
{
	// REVIEW: Get canonical device memory resource type for device
    using DeviceMR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;

    // ============================================================
    // Model setup
    // ============================================================

    std::shared_ptr<ExecutionContext<TDeviceType>> exec_context;
    if constexpr (TDeviceType == DeviceType::Cuda)
    {
        exec_context = std::make_shared<ExecutionContext<TDeviceType>>( 0 );
    }
    else
    {
        exec_context = std::make_shared<ExecutionContext<TDeviceType>>();
    }

    auto net = std::make_unique<MnistClassifier<TDeviceType, TPrecision>>(
        exec_context,
        "MnistClassifier",
        config.batch_size );

    //mnist_net->setTraining( true );

    // Get device from model's execution context
    auto device = net->getDevice();

    MnistDataLoader<TensorDataType::FP32, THostMR> train_loader( config.data_directory, config.batch_size, true, device );
    MnistDataLoader<TensorDataType::FP32, THostMR> test_loader( config.data_directory, config.batch_size, false, device );

    // Build the model with the input shape from the data loader
    shape_t input_shape = { train_loader.batchSize(), MNIST_IMAGE_SIZE };
    net->build( input_shape );

    // In Mnist.cpp, right after model->build():
    //exec_context->synchronize();

    std::cout << "Mnist Network built successfully!" << std::endl;
    std::cout << net->toString() << std::endl;

    // ============================================================
    // AdamW optimizer setup
    // ============================================================

    // Create AdamW optimizer configuration
    auto adamw_config = AdamWConfig()
        .withLearningRate( config.learning_rate )
        .withBeta1( config.beta1 )
        .withBeta2( config.beta2 )
        .withEpsilon( config.epsilon )
        .withWeightDecay( config.weight_decay )
        .withName( "AdamW" );

    // Validate configuration
    adamw_config.validate();

    /* TODO: AdamW config object constructor
    auto optimizer = std::make_shared<AdamWOptimizer<TDeviceType, TDataType>>(
        exec_context,
        adamw_config );*/

    auto optimizer = std::make_unique<AdamWOptimizer<TDeviceType, TPrecision>>(
        exec_context, adamw_config );
        //config.learning_rate,
        //config.beta1,
        //config.beta2,
        //config.epsilon,
        //config.weight_decay );

    // Register all model parameters and gradients with the optimizer
	
    // TJT: BUG: If the model parameters or gradients change after this point,
	// the optimizer will have invalid references.

	// TJT: This should be done via a model method to avoid exposing internal details.

    auto params = net->getParameters();
    auto param_grads = net->getGradients();

    if (params.size() != param_grads.size())
    {
        throw std::runtime_error( "Parameter count mismatch between parameters and gradients" );
    }

    for (size_t i = 0; i < params.size(); ++i)
    {
        optimizer->addParameter( params[i], param_grads[i] );
    }

    std::cout << "Optimizer initialized with " << optimizer->getParameterCount()
        << " parameter groups" << std::endl;

	// TJT: TODO: Loss function abstraction
    //std::unique_ptr<Loss<TDeviceType, TPrecision>> loss_fn{ nullptr };

    // Convert MnistConfig into the library ModelConfig required by Model ctor.
    ModelConfig model_config;
    model_config
        .epochs( static_cast<std::size_t>( config.epochs ) )
        .batchSize( static_cast<std::size_t>( config.batch_size ) )
        .learningRate( static_cast<double>( config.learning_rate ) );

    Model<TDeviceType, TPrecision> model(
        std::move( net ),
        std::move( optimizer ),
        model_config
    );

    // Allocate tensors for training
    Tensor<TPrecision, DeviceMR> input_batch( device, input_shape );
    Tensor<TPrecision, CpuMemoryResource> target_batch( "CPU", { train_loader.batchSize(), MNIST_NUM_CLASSES } );

    Tensor<TPrecision, CpuMemoryResource> logits( "CPU", { train_loader.batchSize(), MNIST_NUM_CLASSES } );
    Tensor<TPrecision, DeviceMR> output( device, { train_loader.batchSize(), MNIST_NUM_CLASSES } );

    // Allocate gradient tensors for backward pass
    Tensor<TPrecision, CpuMemoryResource> output_grad_cpu( "CPU", { train_loader.batchSize(), MNIST_NUM_CLASSES } );
    Tensor<TPrecision, DeviceMR> output_grad( device, { train_loader.batchSize(), MNIST_NUM_CLASSES } );
    Tensor<TPrecision, DeviceMR> input_grad( device, input_shape );

	//model.train( train_loader, test_loader );


	// Review: General training loop is now in Model::train()

    //std::cout << "Starting training for " << config.epochs << " epochs..." << std::endl;

    //for (size_t epoch = 0; epoch < config.epochs; ++epoch)
    //{
    //    train_loader.reset();

    //    float epoch_loss = 0.0f;
    //    float epoch_acc = 0.0f;
    //    size_t batches = 0;

    //    auto start_time = std::chrono::high_resolution_clock::now();

    //    while (train_loader.hasNext())
    //    {
    //        train_loader.nextBatch();

    //        // Copy batch data
    //        copy( train_loader.inputs(), input_batch );
    //        copy( train_loader.targets(), target_batch );

    //        // Forward pass
    //        mnist_net->forward( input_batch, output );
    //        exec_context->synchronize();

    //        // Copy output to CPU for loss computation
    //        // REVIEW: Without passing the exec_context, we are implicitly synchronizing here
    //        copy( output, logits );
    //        
    //        // Compute loss and accuracy
    //        float batch_loss = softmaxCrossEntropyLoss( logits, target_batch );
    //        float batch_acc = computeAccuracy( logits, target_batch );

    //        // ============================================================
    //        // Backward pass and optimization step
    //        // ============================================================

    //        // 1. Compute loss gradient on CPU
    //        // TJT: TODO: To be done on TDevice
    //        zeros( output_grad_cpu );
    //        softmaxCrossEntropyGradient( logits, target_batch, output_grad_cpu );

    //        // 2. Copy gradient to device
    //        copy( output_grad_cpu, output_grad );

    //        // 3. Zero gradients before backward pass
    //        optimizer->zeroGrad();
    //        zeros( input_grad );

    //        // 4. Backward pass through model to compute gradients
    //        mnist_net->backward( input_batch, output_grad, input_grad );

    //        // 5. Update parameters using computed gradients
    //        optimizer->step();

    //        // ============================================================

    //        epoch_loss += batch_loss;
    //        epoch_acc += batch_acc;
    //        batches++;

    //        if (batches % 100 == 0 || batches == train_loader.numBatches())
    //        {
    //            std::cout << "Epoch " << (epoch + 1) << " [" << batches << "/"
    //                << train_loader.numBatches() << "] - Loss: " << std::fixed
    //                << std::setprecision( 4 ) << batch_loss << " - Accuracy: "
    //                << std::setprecision( 2 ) << (batch_acc * 100.0f) << "%" << std::endl;
    //        }
    //    }

    //    epoch_loss /= batches;
    //    epoch_acc /= batches;

    //    auto end_time = std::chrono::high_resolution_clock::now();
    //    auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    //    double epoch_time_sec = epoch_duration.count() / 1000.0;

    //    // Evaluation on test set
    //    mnist_net->setTraining( false );

    //    test_loader.reset();
    //    float test_loss = 0.0f;
    //    float test_acc = 0.0f;
    //    size_t test_batches = 0;

    //    while (test_loader.hasNext())
    //    {
    //        test_loader.nextBatch();

    //        copy( test_loader.inputs(), input_batch );
    //        copy( test_loader.targets(), target_batch );

    //        mnist_net->forward( input_batch, output );

    //        copy( output, logits, exec_context.get() );

    //        test_loss += softmaxCrossEntropyLoss( logits, target_batch );
    //        test_acc += computeAccuracy( logits, target_batch );

    //        test_batches++;
    //    }

    //    test_loss /= test_batches;
    //    test_acc /= test_batches;

    //    std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs
    //        << " - Time: " << std::fixed << std::setprecision( 2 ) << epoch_time_sec << "s"
    //        << " - Loss: " << std::fixed << std::setprecision( 4 ) << epoch_loss
    //        << " - Accuracy: " << std::setprecision( 2 ) << (epoch_acc * 100.0f) << "%"
    //        << " - Test Loss: " << std::setprecision( 4 ) << test_loss
    //        << " - Test Accuracy: " << std::setprecision( 2 ) << (test_acc * 100.0f) << "%"
    //        << " - LR: " << std::scientific << std::setprecision( 3 ) << optimizer->getLearningRate()
    //        << std::endl;

    //    // Back to training mode
    //    mnist_net->setTraining( true );
    //}
}

int main( int argc, char** argv )
{
    try
    {
        std::cout << "MNIST Classification Example using Mila" << std::endl;
        std::cout << "=======================================" << std::endl;

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

                // TODO: Configurable dtype_t. Add FP16 and BF16 first
                trainMnist<DeviceType::Cuda, TensorDataType::FP32, CudaPinnedMemoryResource>( config );
            }
            catch (const std::exception& e)
            {
                std::cerr << "CUDA error: " << e.what() << std::endl;
                throw;
            }
        }

        if (config.compute_device == DeviceType::Cpu)
        {
            std::cout << "Using CPU device" << std::endl;
            trainMnist<DeviceType::Cpu, TensorDataType::FP32, CpuMemoryResource>( config );
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