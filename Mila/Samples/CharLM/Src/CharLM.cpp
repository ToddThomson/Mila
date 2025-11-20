#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <cstdint>
#include <stdexcept>
#include <memory>
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

import CharLM.Transformer;
import CharLM.CharDataLoader;

namespace fs = std::filesystem;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;
using namespace Mila::Dnn::Optimizers;
using namespace Mila::CharLM;

struct CharLMConfig
{
    std::string data_file = "./Data/DataSets/TinyShakespeare/input.txt";
    int64_t batch_size = 32;
    int64_t seq_length = 128;
    size_t epochs = 10;
    float learning_rate = 3e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.01f;
    DeviceType compute_device = DeviceType::Cuda;
    ComputePrecision::Policy precisionPolicy = ComputePrecision::Policy::Auto;
    
    // Model hyperparameters
    int64_t vocab_size = 256;         // ASCII extended (will be overridden by actual vocab)
    int64_t embedding_dim = 256;
    int64_t num_heads = 4;
    int64_t num_layers = 4;
    int64_t mlp_hidden_dim = 1024;
};

void printUsage()
{
    std::cout << "Usage: charlm [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --data-file <path>      Path to token, vocab files (default: ./Data/DataSets/TinyShakespeare/input.txt )\n";
    std::cout << "  --batch-size <int>      Batch size (default: 32)\n";
    std::cout << "  --seq-length <int>      Sequence length (default: 128)\n";
    std::cout << "  --epochs <int>          Number of epochs (default: 10)\n";
    std::cout << "  --learning-rate <float> Learning rate (default: 3e-4)\n";
    std::cout << "  --beta1 <float>         Adam beta1 parameter (default: 0.9)\n";
    std::cout << "  --beta2 <float>         Adam beta2 parameter (default: 0.999)\n";
    std::cout << "  --weight-decay <float>  Weight decay (default: 0.01)\n";
    std::cout << "  --device <string>       Compute device (cpu or cuda, default: cuda)\n";
    std::cout << "  --precision <string>    Precision policy (auto, performance, accuracy, disabled, default: auto)\n";
    std::cout << "  --embedding-dim <int>   Embedding dimension (default: 256)\n";
    std::cout << "  --num-heads <int>       Number of attention heads (default: 4)\n";
    std::cout << "  --num-layers <int>      Number of transformer layers (default: 4)\n";
    std::cout << "  --help                  Show this help message\n";
}

bool parseCommandLine( int argc, char** argv, CharLMConfig& config )
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--help")
        {
            printUsage();
            return false;
        }
        else if (arg == "--data-file" && i + 1 < argc)
        {
            config.data_file = argv[++i];
        }
        else if (arg == "--batch-size" && i + 1 < argc)
        {
            config.batch_size = std::stoi( argv[++i] );
        }
        else if (arg == "--seq-length" && i + 1 < argc)
        {
            config.seq_length = std::stoi( argv[++i] );
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
        else if (arg == "--embedding-dim" && i + 1 < argc)
        {
            config.embedding_dim = std::stoi( argv[++i] );
        }
        else if (arg == "--num-heads" && i + 1 < argc)
        {
            config.num_heads = std::stoi( argv[++i] );
        }
        else if (arg == "--num-layers" && i + 1 < argc)
        {
            config.num_layers = std::stoi( argv[++i] );
        }
        else if (arg.substr( 0, 2 ) == "--")
        {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage();
            return false;
        }
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Data File: " << config.data_file<< std::endl;
    std::cout << "  Batch size: " << config.batch_size << std::endl;
    std::cout << "  Sequence length: " << config.seq_length << std::endl;
    std::cout << "  Epochs: " << config.epochs << std::endl;
    std::cout << "  Learning rate: " << config.learning_rate << std::endl;
    std::cout << "  Beta1: " << config.beta1 << std::endl;
    std::cout << "  Beta2: " << config.beta2 << std::endl;
    std::cout << "  Weight decay: " << config.weight_decay << std::endl;
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
    std::cout << std::endl;

    std::cout << "  Embedding dimension: " << config.embedding_dim << std::endl;
    std::cout << "  Number of heads: " << config.num_heads << std::endl;
    std::cout << "  Number of layers: " << config.num_layers << std::endl;
    std::cout << std::endl;

    if (!fs::exists( config.data_file ))
    {
        std::cerr << "Data file not found: " << config.data_file<< std::endl;
        std::cerr << "Please provide a text file for character-level language modeling." << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief Compute cross-entropy gradient for sequence prediction.
 */
template<TensorDataType TDataType>
void sequenceCrossEntropyGradient(
    const Tensor<TDataType, CpuMemoryResource>& logits,
    const Tensor<TDataType, CpuMemoryResource>& targets,
    Tensor<TDataType, CpuMemoryResource>& output_grad )
{
    using HostType = typename TensorHostTypeMap<TDataType>::host_type;

    size_t batch_size = logits.shape()[0];
    size_t seq_length = logits.shape()[1];
    size_t vocab_size = logits.shape()[2];

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t t = 0; t < seq_length; ++t)
        {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t v = 0; v < vocab_size; ++v)
            {
                size_t idx = b * seq_length * vocab_size + t * vocab_size + v;
                max_logit = std::max( max_logit, static_cast<float>( logits.data()[idx] ) );
            }

            float denom = 0.0f;
            for (size_t v = 0; v < vocab_size; ++v)
            {
                size_t idx = b * seq_length * vocab_size + t * vocab_size + v;
                float exp_val = std::exp( static_cast<float>( logits.data()[idx] ) - max_logit );
                denom += exp_val;
            }

            size_t target_idx = static_cast<size_t>( targets.data()[b * seq_length + t] );
            for (size_t v = 0; v < vocab_size; ++v)
            {
                size_t idx = b * seq_length * vocab_size + t * vocab_size + v;
                float prob = std::exp( static_cast<float>( logits.data()[idx] ) - max_logit ) / denom;
                float target = (v == target_idx) ? 1.0f : 0.0f;

                output_grad.data()[idx] = static_cast<HostType>(
                    (prob - target) / static_cast<float>( batch_size * seq_length ) );
            }
        }
    }
}

/**
 * @brief Compute cross-entropy loss for sequence prediction.
 */
template<TensorDataType TDataType>
float sequenceCrossEntropyLoss(
    const Tensor<TDataType, CpuMemoryResource>& logits,
    const Tensor<TDataType, CpuMemoryResource>& targets )
{
    size_t batch_size = logits.shape()[0];
    size_t seq_length = logits.shape()[1];
    size_t vocab_size = logits.shape()[2];
    float loss = 0.0f;

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t t = 0; t < seq_length; ++t)
        {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t v = 0; v < vocab_size; ++v)
            {
                size_t idx = b * seq_length * vocab_size + t * vocab_size + v;
                max_logit = std::max( max_logit, static_cast<float>( logits.data()[idx] ) );
            }

            float denom = 0.0f;
            for (size_t v = 0; v < vocab_size; ++v)
            {
                size_t idx = b * seq_length * vocab_size + t * vocab_size + v;
                float exp_val = std::exp( static_cast<float>( logits.data()[idx] ) - max_logit );
                denom += exp_val;
            }

            size_t target_idx = static_cast<size_t>( targets.data()[b * seq_length + t] );
            size_t logit_idx = b * seq_length * vocab_size + t * vocab_size + target_idx;
            float prob = std::exp( static_cast<float>( logits.data()[logit_idx] ) - max_logit ) / denom;
            loss += -std::log( prob );
        }
    }

    return loss / (batch_size * seq_length);
}

/**
 * @brief Compute perplexity from loss.
 */
float computePerplexity( float loss )
{
    return std::exp( loss );
}

template<DeviceType TDeviceType, TensorDataType TDataType, typename THostMR>
    requires PrecisionSupportedOnDevice<TDataType, TDeviceType> &&
             (std::is_same_v<THostMR, CudaPinnedMemoryResource> || std::is_same_v<THostMR, CpuMemoryResource>)
void train( const CharLMConfig& config )
{
    using DeviceMR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;

    // ============================================================
    // Execution context setup
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

    auto device = exec_context->getDevice();

    // ============================================================
    // Data loader setup
    // ============================================================
    
    CharDataLoader<TensorDataType::FP32, THostMR> train_loader(
        config.data_file,
        config.batch_size,
        config.seq_length,
        true,  // is_training (shuffle)
        device,
        config.seq_length / 2  // stride (50% overlap for more training data)
    );

    // Update vocab_size from actual data
    size_t actual_vocab_size = train_loader.vocabSize();
    std::cout << "Actual vocabulary size from data: " << actual_vocab_size << std::endl;

    // ============================================================
    // Model setup
    // ============================================================

    CharTransformerConfig model_config;
    model_config.vocab_size = actual_vocab_size;
    model_config.max_seq_length = config.seq_length;
    model_config.embedding_dim = config.embedding_dim;
    model_config.num_heads = config.num_heads;
    model_config.num_layers = config.num_layers;
    model_config.mlp_hidden_dim = config.mlp_hidden_dim;
    model_config.name = "CharTransformer";

    auto model = std::make_shared<CharTransformer<TDeviceType, TDataType>>(
        exec_context,
        model_config );

    model->setTraining( true );

    // Build the model with the input shape
    shape_t input_shape = { config.batch_size, config.seq_length };
    model->build( input_shape );

    std::cout << "Model built successfully!" << std::endl;
    std::cout << model->toString() << std::endl;

    // ============================================================
    // AdamW optimizer setup
    // ============================================================

    auto adamw_config = AdamWConfig()
        .withLearningRate( config.learning_rate )
        .withBeta1( config.beta1 )
        .withBeta2( config.beta2 )
        .withEpsilon( config.epsilon )
        .withWeightDecay( config.weight_decay )
        .withName( "AdamW" );

    adamw_config.validate();

    auto optimizer = std::make_shared<AdamWOptimizer<TDeviceType, TDataType>>(
        exec_context, adamw_config );
        //config.learning_rate,
        //config.beta1,
        //config.beta2,
        //config.epsilon,
        //config.weight_decay );

    // Register all model parameters with the optimizer
    auto params = model->getParameters();
    auto param_grads = model->getGradients();

    // Debug
	auto param_count = params.size();
	auto grad_count = param_grads.size();

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

    // ============================================================
    // Allocate tensors for training
    // ============================================================

    shape_t sequence_shape = { config.batch_size, config.seq_length };
    shape_t logits_shape = { config.batch_size, config.seq_length, static_cast<int64_t>(actual_vocab_size) };

    Tensor<TDataType, DeviceMR> input_batch( device, sequence_shape );
    Tensor<TDataType, DeviceMR> target_batch( device, sequence_shape );
    Tensor<TDataType, DeviceMR> output( device, logits_shape );

    // CPU tensors for loss computation
    Tensor<TDataType, CpuMemoryResource> logits_cpu( "CPU", logits_shape );
    Tensor<TDataType, CpuMemoryResource> targets_cpu( "CPU", sequence_shape );

    // Allocate gradient tensors for backward pass
    Tensor<TDataType, CpuMemoryResource> output_grad_cpu( "CPU", logits_shape );
    Tensor<TDataType, DeviceMR> output_grad( device, logits_shape );
    Tensor<TDataType, DeviceMR> input_grad( device, sequence_shape );

    std::cout << "\nStarting training for " << config.epochs << " epochs..." << std::endl;
    std::cout << "Total batches per epoch: " << train_loader.numBatches() << std::endl;
    std::cout << std::endl;

    // ============================================================
    // Training loop
    // ============================================================

    for (size_t epoch = 0; epoch < config.epochs; ++epoch)
    {
        train_loader.reset();

        float epoch_loss = 0.0f;
        size_t batches = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        while (train_loader.hasNext())
        {
            train_loader.nextBatch();

            copy( train_loader.inputs(), input_batch );
            copy( train_loader.targets(), target_batch );

            model->forward( input_batch, output );
            exec_context->synchronize();

            // Copy output to CPU for loss computation
            copy( output, logits_cpu );
            copy( target_batch, targets_cpu );

            // Compute loss
            float batch_loss = sequenceCrossEntropyLoss( logits_cpu, targets_cpu );
            float batch_perplexity = computePerplexity( batch_loss );

            // ============================================================
            // Backward pass and optimization step
            // ============================================================

            // 1. Compute loss gradient on CPU
            zeros( output_grad_cpu );
            sequenceCrossEntropyGradient( logits_cpu, targets_cpu, output_grad_cpu );

            // 2. Copy gradient to device
            copy( output_grad_cpu, output_grad );

            // 3. Zero gradients before backward pass
            optimizer->zeroGrad();
            zeros( input_grad );

            // 4. Backward pass through model (computes parameter gradients)
            model->backward( input_batch, output_grad, input_grad );

            // 5. Update parameters using computed gradients
            optimizer->step();

            // ============================================================

            epoch_loss += batch_loss;
            batches++;

            // Print progress every 50 batches or at end of epoch
            if (batches % 50 == 0 || batches == train_loader.numBatches())
            {
                std::cout << "Epoch " << (epoch + 1) << " [" << batches << "/"
                    << train_loader.numBatches() << "] - Loss: " << std::fixed
                    << std::setprecision( 4 ) << batch_loss 
                    << " - Perplexity: " << std::setprecision( 2 ) << batch_perplexity
                    << std::endl;
            }
        }

        epoch_loss /= batches;
        float epoch_perplexity = computePerplexity( epoch_loss );

        auto end_time = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double epoch_time_sec = epoch_duration.count() / 1000.0;

        std::cout << "\nEpoch " << (epoch + 1) << "/" << config.epochs
            << " - Time: " << std::fixed << std::setprecision( 2 ) << epoch_time_sec << "s"
            << " - Avg Loss: " << std::setprecision( 4 ) << epoch_loss
            << " - Avg Perplexity: " << std::setprecision( 2 ) << epoch_perplexity
            << " - LR: " << std::scientific << std::setprecision( 3 ) << optimizer->getLearningRate()
            << std::endl << std::endl;
    }

    std::cout << "Training complete!" << std::endl;
}

int main( int argc, char** argv )
{
    try
    {
        std::cout << "Character Language Model Example using Mila" << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << std::endl;

        Mila::initialize();

        CharLMConfig config;
        if (!parseCommandLine( argc, argv, config ))
        {
            return 1;
        }

        if (config.compute_device == DeviceType::Cuda)
        {
            try
            {
                std::cout << "Using CUDA device" << std::endl;
                train<DeviceType::Cuda, TensorDataType::FP32, CudaPinnedMemoryResource>( config );
            }
            catch (const std::exception& e)
            {
                std::cerr << "CUDA error: " << e.what();
                throw;
            }
        }

        if (config.compute_device == DeviceType::Cpu)
        {
            std::cout << "Using CPU device" << std::endl;
            //trainCharLM<DeviceType::Cpu, TensorDataType::FP32, CpuMemoryResource>( config );
        }

        std::cout << "\nComplete!" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}