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
#include <iomanip>
#include <chrono>
#include <format>
#include <limits>
#include <ios>
#include <exception>
#include <cmath>

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
    TensorDataType precision = TensorDataType::FP32;
    ComputePrecision::Policy precisionPolicy = ComputePrecision::Policy::Auto;

    int64_t embedding_dim = 256;
    int64_t num_heads = 4;
    int64_t num_layers = 4;
    int64_t mlp_hidden_dim = 1024;
};

void printUsage()
{
    std::cout << "Usage: charlm [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --data-file <path>       Path to text file (default: ./Data/DataSets/TinyShakespeare/input.txt)\n";
    std::cout << "  --batch-size <int>       Batch size (default: 32)\n";
    std::cout << "  --seq-length <int>       Sequence length (default: 128)\n";
    std::cout << "  --epochs <int>           Number of epochs (default: 10)\n";
    std::cout << "  --learning-rate <float>  Learning rate (default: 3e-4)\n";
    std::cout << "  --beta1 <float>          Adam beta1 parameter (default: 0.9)\n";
    std::cout << "  --beta2 <float>          Adam beta2 parameter (default: 0.999)\n";
    std::cout << "  --weight-decay <float>   Weight decay (default: 0.01)\n";
    std::cout << "  --device <string>        Compute device (cpu or cuda, default: cuda)\n";
    std::cout << "  --precision-policy <string> Precision policy (auto, performance, accuracy, disabled, default: auto)\n";
    std::cout << "  --embedding-dim <int>    Embedding dimension (default: 256)\n";
    std::cout << "  --num-heads <int>        Number of attention heads (default: 4)\n";
    std::cout << "  --num-layers <int>       Number of transformer layers (default: 4)\n";
    std::cout << "  --help                   Show this help message\n";
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
            if (device == "CPU")
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
    std::cout << "  Data file: " << config.data_file << std::endl;
    std::cout << "  Batch size: " << config.batch_size << std::endl;
    std::cout << "  Sequence length: " << config.seq_length << std::endl;
    std::cout << "  Epochs: " << config.epochs << std::endl;
    std::cout << "  Learning rate: " << config.learning_rate << std::endl;
    std::cout << "  Beta1: " << config.beta1 << std::endl;
    std::cout << "  Beta2: " << config.beta2 << std::endl;
    std::cout << "  Weight decay: " << config.weight_decay << std::endl;
    std::cout << "  Device: " << (config.compute_device == DeviceType::Cuda ? "CUDA" : "CPU") << std::endl;
    std::cout << "  Embedding dimension: " << config.embedding_dim << std::endl;
    std::cout << "  Number of heads: " << config.num_heads << std::endl;
    std::cout << "  Number of layers: " << config.num_layers << std::endl;

    if (!fs::exists( config.data_file ))
    {
        std::cerr << "Data file not found: " << config.data_file << std::endl;
        std::cerr << "Please provide a text file for character-level language modeling." << std::endl;
        return false;
    }

    return true;
}

template<TensorDataType TDataType>
void sequenceCrossEntropyGradient(
    const Tensor<TDataType, CpuMemoryResource>& logits,
    const Tensor<TensorDataType::INT32, CpuMemoryResource>& targets,
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

template<TensorDataType TDataType>
float sequenceCrossEntropyLoss(
    const Tensor<TDataType, CpuMemoryResource>& logits,
    const Tensor<TensorDataType::INT32, CpuMemoryResource>& targets )
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

float computePerplexity( float loss )
{
    return std::exp( loss );
}

template<DeviceType TDeviceType, TensorDataType TDataType, typename THostMR>
    requires PrecisionSupportedOnDevice<TDataType, TDeviceType> &&
             (std::is_same_v<THostMR, CudaPinnedMemoryResource> || std::is_same_v<THostMR, CpuMemoryResource>)
void trainCharLM( const CharLMConfig& config )
{
    using DeviceMR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
    DeviceId device_id = Device::getDeviceId<TDeviceType>(0);

    // Data loader should produce INT32 token indices (tokens/targets) on host memory resource
    CharDataLoader<THostMR> train_loader(
        config.data_file,
        config.batch_size,
        config.seq_length,
        true,
        device_id,
        config.seq_length / 2
    );

    size_t actual_vocab_size = train_loader.vocabSize();
    std::cout << "Actual vocabulary size from data: " << actual_vocab_size << std::endl;

    CharTransformerConfig model_config;
    model_config.vocab_size = actual_vocab_size;
    model_config.max_seq_length = config.seq_length;
    model_config.embedding_dim = config.embedding_dim;
    model_config.num_heads = config.num_heads;
    model_config.num_layers = config.num_layers;
    model_config.mlp_hidden_dim = config.mlp_hidden_dim;

    auto model = std::make_unique<CharTransformer<TDeviceType, TDataType>>(
        "CharTransformer",
        model_config,
        device_id );

    shape_t input_shape = { config.batch_size, config.seq_length };
    model->build( input_shape );

    std::cout << "Model built successfully!" << std::endl;
    std::cout << model->toString() << std::endl;

    // Create optimizer (automatic training mode + registration)
    auto optimizer = model->createOptimizer<AdamWOptimizer<TDeviceType, TDataType>>(
        AdamWConfig()
            .withLearningRate( config.learning_rate )
            .withBeta1( config.beta1 )
            .withBeta2( config.beta2 )
            .withEpsilon( config.epsilon )
            .withWeightDecay( config.weight_decay )
    );

    std::cout << "Optimizer initialized with " << optimizer->getParameterCount()
        << " parameter groups" << std::endl;

    shape_t sequence_shape = { config.batch_size, config.seq_length };
    shape_t logits_shape = { config.batch_size, config.seq_length, static_cast<int64_t>(actual_vocab_size) };

    // Inputs/targets are token indices (INT32) on device
    Tensor<TensorDataType::INT32, DeviceMR> input_batch( device_id, sequence_shape );
    Tensor<TensorDataType::INT32, DeviceMR> target_batch( device_id, sequence_shape );

    // Model output / logits use the model numeric type
    Tensor<TDataType, DeviceMR> output( device_id, logits_shape );

    // Host copies for loss computation: logits host in model numeric type, targets host as INT32
    Tensor<TDataType, CpuMemoryResource> logits_cpu( Device::Cpu(), logits_shape );
    Tensor<TensorDataType::INT32, CpuMemoryResource> targets_cpu( Device::Cpu(), sequence_shape );

    Tensor<TDataType, CpuMemoryResource> output_grad_cpu( Device::Cpu(), logits_shape );
    Tensor<TDataType, DeviceMR> output_grad( device_id, logits_shape );
    Tensor<TDataType, DeviceMR> input_grad( device_id, sequence_shape );

    std::cout << "Starting training for " << config.epochs << " epochs..." << std::endl;
    std::cout << "Total batches per epoch: " << train_loader.numBatches() << std::endl;

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
            model->synchronize();

            copy( output, logits_cpu );
            copy( target_batch, targets_cpu );

            float batch_loss = sequenceCrossEntropyLoss( logits_cpu, targets_cpu );
            float batch_perplexity = computePerplexity( batch_loss );

            zeros( output_grad_cpu );
            sequenceCrossEntropyGradient( logits_cpu, targets_cpu, output_grad_cpu );

            copy( output_grad_cpu, output_grad );

            optimizer->zeroGrad();
            zeros( input_grad );

            model->backward( input_batch, output_grad, input_grad );

            optimizer->step();

            epoch_loss += batch_loss;
            batches++;

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

        std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs
            << " - Time: " << std::fixed << std::setprecision( 2 ) << epoch_time_sec << "s"
            << " - Avg Loss: " << std::setprecision( 4 ) << epoch_loss
            << " - Avg Perplexity: " << std::setprecision( 2 ) << epoch_perplexity
            << " - LR: " << std::scientific << std::setprecision( 3 ) << optimizer->getLearningRate()
            << std::endl;
    }

    std::cout << "Training complete!" << std::endl;
}

int main( int argc, char** argv )
{
    try
    {
        std::cout << "Character Language Model Example using Mila" << std::endl;
        std::cout << "===========================================" << std::endl;

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
                trainCharLM<DeviceType::Cuda, TensorDataType::FP32, CudaPinnedMemoryResource>( config );
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
            trainCharLM<DeviceType::Cpu, TensorDataType::FP32, CpuMemoryResource>( config );
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