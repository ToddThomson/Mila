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
import CharLM.Vocabulary;

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
    size_t epochs = 200;
    float learning_rate = 0.0003f; // 1e-4f; // we've tried 3e-4f; 0.001f, 0.0005f
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.001f; // we've tried 0.01f, 0.001f, 0.0001f
    DeviceType compute_device = DeviceType::Cuda;
    TensorDataType precision = TensorDataType::FP32;
    ComputePrecision::Policy precisionPolicy = ComputePrecision::Policy::Auto;

    int64_t embedding_dim = 384;
    int64_t num_heads = 6;
    int64_t num_layers = 6;
    int64_t mlp_hidden_dim = 1536;

    float sample_temperature = 0.8f;
    int64_t sample_length = 300;
    size_t sample_every_n_epochs = 10;
    std::string sample_prompt = "ROMEO:\n";

    // Simple epoch-based learning-rate scheduler:
    // lr_decay: multiplicative factor applied when schedule triggers (e.g., 0.9)
    // lr_decay_every_n_epochs: apply decay every N epochs (0 = disable)
    float lr_decay = 1.0f;
    size_t lr_decay_every_n_epochs = 0;
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
    std::cout << "  --sample-temperature <float> Sampling temperature (default: 0.8)\n";
    std::cout << "  --sample-length <int>    Length of generated samples (default: 300)\n";
    std::cout << "  --sample-every <int>     Generate sample every N epochs (default: 1, 0=disable)\n";
    std::cout << "  --lr-decay <float>       Multiplicative LR decay factor (e.g. 0.9). 1.0 means no decay.\n";
    std::cout << "  --lr-decay-every <int>   Apply LR decay every N epochs (0 = disable)\n";
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
        else if (arg == "--sample-temperature" && i + 1 < argc)
        {
            config.sample_temperature = std::stof( argv[++i] );
        }
        else if (arg == "--sample-length" && i + 1 < argc)
        {
            config.sample_length = std::stoi( argv[++i] );
        }
        else if (arg == "--sample-every" && i + 1 < argc)
        {
            config.sample_every_n_epochs = std::stoi( argv[++i] );
        }
        else if (arg == "--lr-decay" && i + 1 < argc)
        {
            config.lr_decay = std::stof( argv[++i] );
        }
        else if (arg == "--lr-decay-every" && i + 1 < argc)
        {
            config.lr_decay_every_n_epochs = static_cast<size_t>( std::stoll( argv[++i] ) );
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
    std::cout << "  Sample temperature: " << config.sample_temperature << std::endl;
    std::cout << "  Sample length: " << config.sample_length << std::endl;
    std::cout << "  Sample every N epochs: " << config.sample_every_n_epochs << std::endl;
    std::cout << "  LR decay factor: " << config.lr_decay << std::endl;
    std::cout << "  LR decay every N epochs: " << config.lr_decay_every_n_epochs << std::endl;

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

    size_t batch_size = logits.shape()[ 0 ];
    size_t seq_length = logits.shape()[ 1 ];
    size_t vocab_size = logits.shape()[ 2 ];

    for ( size_t b = 0; b < batch_size; ++b )
    {
        for ( size_t t = 0; t < seq_length; ++t )
        {
            float max_logit = -std::numeric_limits<float>::infinity();
            for ( size_t v = 0; v < vocab_size; ++v )
            {
                size_t idx = b * seq_length * vocab_size + t * vocab_size + v;
                max_logit = std::max( max_logit, static_cast<float>( logits.data()[ idx ] ) );
            }

            float denom = 0.0f;
            for ( size_t v = 0; v < vocab_size; ++v )
            {
                size_t idx = b * seq_length * vocab_size + t * vocab_size + v;
                float exp_val = std::exp( static_cast<float>( logits.data()[ idx ] ) - max_logit );
                denom += exp_val;
            }

            size_t target_idx = static_cast<size_t>( targets.data()[ b * seq_length + t ] );
            for ( size_t v = 0; v < vocab_size; ++v )
            {
                size_t idx = b * seq_length * vocab_size + t * vocab_size + v;
                float prob = std::exp( static_cast<float>( logits.data()[ idx ] ) - max_logit ) / denom;
                float target = (v == target_idx) ? 1.0f : 0.0f;

                output_grad.data()[ idx ] = static_cast<HostType>(
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

template<TensorDataType TDataType>
int32_t sampleFromLogits(
    const typename TensorHostTypeMap<TDataType>::host_type* logits_ptr,
    size_t vocab_size,
    float temperature,
    std::mt19937& rng )
{
    using HostType = typename TensorHostTypeMap<TDataType>::host_type;

    float max_logit = -std::numeric_limits<float>::infinity();
    for ( size_t v = 0; v < vocab_size; ++v )
    {
        max_logit = std::max( max_logit, static_cast<float>( logits_ptr[ v ] ) );
    }

    std::vector<float> probs( vocab_size );
    float sum = 0.0f;

    for ( size_t v = 0; v < vocab_size; ++v )
    {
        float scaled_logit = (static_cast<float>( logits_ptr[ v ] ) - max_logit) / temperature;
        probs[ v ] = std::exp( scaled_logit );
        sum += probs[ v ];
    }

    for ( size_t v = 0; v < vocab_size; ++v )
    {
        probs[ v ] /= sum;
    }

    std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
    float sample = dist( rng );
    float cumsum = 0.0f;

    for ( size_t v = 0; v < vocab_size; ++v )
    {
        cumsum += probs[ v ];
        if ( sample < cumsum )
        {
            return static_cast<int32_t>( v );
        }
    }

    return static_cast<int32_t>( vocab_size - 1 );
}


// -----------------------------------------------------------------------------
// DEBUG:
// Small parameter / gradient norm logger
// - Non-invasive: logs a per-epoch summary (mean and max L2 norms).
// - Uses dynamic_cast to typed Tensor; ignores tensors that don't match expected
//   device/precision types.
// -----------------------------------------------------------------------------
template<DeviceType TDeviceType, TensorDataType TDataType>
void logParameterAndGradientNorms( CharTransformer<TDeviceType, TDataType>* model )
{
    using DeviceMR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
    using DeviceTensor = Tensor<TDataType, DeviceMR>;

    std::vector<ITensor*> params;
    std::vector<ITensor*> grads;

    try
    {
        params = model->getParameters();
    }
    catch ( const std::exception& )
    {
        // If getParameters throws for some reason, skip logging
        return;
    }

    try
    {
        grads = model->getGradients();
    }
    catch ( const std::exception& )
    {
        // Gradients may not be available; continue with parameters only
        grads.clear();
    }

    double sum_param_norm = 0.0;
    double max_param_norm = 0.0;
    size_t param_count = 0;

    for ( ITensor* p : params )
    {
        auto* dp = dynamic_cast<DeviceTensor*>( p );
        if ( !dp ) continue;

        auto host = toHost<TDataType>( *dp );

        double s = 0.0;
        for ( size_t i = 0; i < host.size(); ++i )
        {
            double v = static_cast<double>( host.data()[ i ] );
            s += v * v;
        }

        double norm = std::sqrt( s );
        sum_param_norm += norm;
        max_param_norm = std::max( max_param_norm, norm );
        param_count++;
    }

    double sum_grad_norm = 0.0;
    double max_grad_norm = 0.0;
    size_t grad_count = 0;

    for ( ITensor* g : grads )
    {
        auto* dg = dynamic_cast<DeviceTensor*>( g );
        if ( !dg ) continue;

        auto host = toHost<TDataType>( *dg );

        double s = 0.0;
        for ( size_t i = 0; i < host.size(); ++i )
        {
            double v = static_cast<double>( host.data()[ i ] );
            s += v * v;
        }

        double norm = std::sqrt( s );
        sum_grad_norm += norm;
        max_grad_norm = std::max( max_grad_norm, norm );
        grad_count++;
    }

    std::cout << "  [NORM] params: count=" << param_count
              << " mean=" << (param_count ? (sum_param_norm / param_count) : 0.0)
              << " max=" << max_param_norm
              << " | grads: count=" << grad_count
              << " mean=" << (grad_count ? (sum_grad_norm / grad_count) : 0.0)
              << " max=" << max_grad_norm
              << std::endl;
}

/// <brief>Clip gradients in-place on device by copying to host, scaling, and copying back.
/// <param name="model">Model whose gradients are clipped.</param>
/// <param name="max_norm">Maximum allowed L2 norm across all gradient tensors.</param>
template<DeviceType TDeviceType, TensorDataType TDataType>
void clipGradients( CharTransformer<TDeviceType, TDataType>* model, float max_norm )
{
    using DeviceMR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
    using DeviceTensor = Tensor<TDataType, DeviceMR>;
    using HostTensor = Tensor<TDataType, CpuMemoryResource>;

    std::vector<ITensor*> grads;

    try
    {
        grads = model->getGradients();
    }
    catch ( const std::exception& )
    {
        // Gradients may not be available; nothing to clip
        return;
    }

    // Gather host copies and compute global norm
    double sumsq = 0.0;
    std::vector<std::pair<DeviceTensor*, HostTensor>> host_pairs;
    host_pairs.reserve( grads.size() );

    for ( ITensor* g : grads )
    {
        auto* dg = dynamic_cast<DeviceTensor*>( g );
        if ( !dg ) continue;

        HostTensor host = toHost<TDataType>( *dg ); // copy to host
        for ( size_t i = 0; i < host.size(); ++i )
        {
            double v = static_cast<double>( host.data()[ i ] );
            sumsq += v * v;
        }

        host_pairs.emplace_back( dg, std::move(host) );
    }

    double total_norm = std::sqrt( sumsq );

    if ( total_norm <= static_cast<double>( max_norm ) || total_norm == 0.0 )
    {
        return;
    }

    const float scale = static_cast<float>( max_norm / total_norm );

    // Scale host copies and copy back to device
    for ( auto &pair : host_pairs )
    {
        HostTensor &host = pair.second;
        for ( size_t i = 0; i < host.size(); ++i )
        {
            host.data()[ i ] = ( host.data()[ i ] * scale );
        }

        // Copy scaled host gradient back to device tensor
        copy( host, *pair.first );
    }
}

template<DeviceType TDeviceType, TensorDataType TDataType, typename THostMR>
    requires PrecisionSupportedOnDevice<TDataType, TDeviceType> &&
             (std::is_same_v<THostMR, CudaPinnedMemoryResource> || std::is_same_v<THostMR, CpuMemoryResource>)
std::string generateSample(
    CharTransformer<TDeviceType, TDataType>* model,
    const CharVocabulary& vocab,
    DeviceId device_id,
    const CharLMConfig& config,
    size_t epoch )
{
    using DeviceMR = typename DeviceTypeTraits<TDeviceType>::memory_resource;

    // DEBUG: temporarily set model to eval mode
    //bool was_training = model->isTraining();
    //model->setTraining( false );

    std::vector<int32_t> prompt_tokens;
    for ( char c : config.sample_prompt )
    {
        prompt_tokens.push_back( vocab.charToIndex( c ) );
    }

    std::vector<int32_t> generated_tokens = prompt_tokens;
    std::mt19937 rng( static_cast<unsigned int>(epoch) );

    size_t vocab_size = vocab.size();

    int64_t model_batch_size = config.batch_size;
    int64_t model_seq_length = config.seq_length;

    shape_t model_shape = { model_batch_size, model_seq_length };
    shape_t logits_shape = { model_batch_size, model_seq_length, static_cast<int64_t>(vocab_size) };

    Tensor<TensorDataType::INT32, DeviceMR> context_device( device_id, model_shape );
    Tensor<TDataType, DeviceMR> logits_device( device_id, logits_shape );
    Tensor<TDataType, CpuMemoryResource> logits_cpu( Device::Cpu(), logits_shape );
    Tensor<TensorDataType::INT32, CpuMemoryResource> context_cpu( Device::Cpu(), model_shape );

    for ( int64_t i = 0; i < config.sample_length; ++i )
    {
        int64_t context_start = std::max( int64_t( 0 ),
            static_cast<int64_t>( generated_tokens.size() ) - model_seq_length );
        int64_t actual_context_len = static_cast<int64_t>( generated_tokens.size() ) - context_start;

        std::fill_n( context_cpu.data(), model_batch_size * model_seq_length, 0 );

        int64_t pad_left = model_seq_length - actual_context_len;
        for ( int64_t j = 0; j < actual_context_len; ++j )
        {
            context_cpu.data()[ pad_left + j ] = generated_tokens[ context_start + j ];
        }

        copy( context_cpu, context_device );

        model->forward( context_device, logits_device );
        model->synchronize();

        copy( logits_device, logits_cpu );

        size_t batch_0_last_token_offset = (model_seq_length - 1) * vocab_size;

        int32_t next_token = sampleFromLogits<TDataType>(
            logits_cpu.data() + batch_0_last_token_offset,
            vocab_size,
            config.sample_temperature,
            rng );

        generated_tokens.push_back( next_token );
    }

    std::string generated_text;
    for ( int32_t token : generated_tokens )
    {
        generated_text += vocab.indexToChar( token );
    }

    //if ( was_training )
    //{
    //    model->setTraining( true );
    //}

    return generated_text;
}

template<DeviceType TDeviceType, TensorDataType TDataType, typename THostMR>
    requires PrecisionSupportedOnDevice<TDataType, TDeviceType> &&
             (std::is_same_v<THostMR, CudaPinnedMemoryResource> || std::is_same_v<THostMR, CpuMemoryResource>)
void trainCharLM( const CharLMConfig& config )
{
    using DeviceMR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
    DeviceId device_id = Device::getDeviceId<TDeviceType>(0);

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

    std::string vocab_file = config.data_file + ".vocab";
    CharVocabulary vocab;
    vocab.load( vocab_file );
    std::cout << "Loaded vocabulary from: " << vocab_file << std::endl;

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

    // After building, set training mode
    model->setTraining( true );

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

    if (config.lr_decay != 1.0f && config.lr_decay_every_n_epochs == 0)
    {
        // apply decay every epoch by default when a decay factor is set
        const_cast<CharLMConfig&>(config).lr_decay_every_n_epochs = 1;
    }

    shape_t logits_shape = { config.batch_size, config.seq_length, static_cast<int64_t>(actual_vocab_size) };

    Tensor<TensorDataType::INT32, DeviceMR> input_batch( device_id, input_shape );
    Tensor<TensorDataType::INT32, DeviceMR> target_batch( device_id, input_shape );

    Tensor<TDataType, DeviceMR> output( device_id, logits_shape );
    Tensor<TDataType, CpuMemoryResource> logits_cpu( Device::Cpu(), logits_shape );
    Tensor<TensorDataType::INT32, CpuMemoryResource> targets_cpu( Device::Cpu(), input_shape );
    Tensor<TDataType, CpuMemoryResource> output_grad_cpu( Device::Cpu(), logits_shape );
    Tensor<TDataType, DeviceMR> output_grad( device_id, logits_shape );

    std::cout << "Starting training for " << config.epochs << " epochs..." << std::endl;
    std::cout << "Total batches per epoch: " << train_loader.numBatches() << std::endl;

    for (size_t epoch = 0; epoch < config.epochs; ++epoch)
    {
        if (config.lr_decay != 1.0f &&
            config.lr_decay_every_n_epochs > 0 &&
            epoch > 0 &&
            ( (epoch % config.lr_decay_every_n_epochs) == 0 ) )
        {
            float old_lr = optimizer->getLearningRate();
            float new_lr = old_lr * config.lr_decay;
            optimizer->setLearningRate( new_lr );

            std::cout << "Adjusted learning rate: " << std::scientific << std::setprecision(3)
                      << old_lr << " -> " << new_lr << std::endl;
        }

        train_loader.reset();

        float epoch_loss = 0.0f;
        size_t batches = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        while (train_loader.hasNext())
        {
            train_loader.nextBatch();

            // Load input and target batches to device
            // NOTE: moving the targets to device required for when loss is computed on device ( not yet implemented )
            
            copy( train_loader.inputs(), input_batch );
            copy( train_loader.targets(), target_batch );

            // DEBUG: print first input batch
            //std::clog << "Inputs batch:\n" << train_loader.inputs().toString( true ) << std::endl;
            //std::clog << "Targets batch:\n" << train_loader.targets().toString( true ) << std::endl;

            // Run forward to get logits on device
            model->forward( input_batch, output );
            model->synchronize();

            // Move logits and targets to host and compute loss + output gradients on CPU
            copy( output, logits_cpu );
            copy( target_batch, targets_cpu );

            float batch_loss = sequenceCrossEntropyLoss( logits_cpu, targets_cpu );
            float batch_perplexity = computePerplexity( batch_loss );

            zero( output_grad_cpu );
            sequenceCrossEntropyGradient( logits_cpu, targets_cpu, output_grad_cpu );

            //std::clog << "Output Grad:\n" << output_grad_cpu.toString( true ) << std::endl;

            //// DEBUG: Gradient verification
            //std::cout << "\n=== GRADIENT VERIFICATION ===\n";

            //// Check first position
            //int b = 0, t = 0;
            //int target_token = targets_cpu.data()[ b * 128 + t ];
            //std::cout << "Position [0,0] target token: " << target_token << "\n";

            //// Print gradients for first position (all 66 vocab entries)
            //std::cout << "Gradients for position [0,0]:\n";
            //for ( int v = 0; v < 66; ++v ) {
            //    size_t idx = 0 * 128 * 66 + 0 * 66 + v;
            //    float grad = output_grad_cpu.data()[ idx ];
            //    if ( v == target_token ) {
            //        std::cout << "  [" << v << "] (TARGET): " << grad << "\n";
            //    }
            //    else if ( v < 5 || v > 60 ) {  // Show first 5 and last 5
            //        std::cout << "  [" << v << "]: " << grad << "\n";
            //    }
            //}

            //// Verify: sum of all gradients for one position should be close to 0
            //float sum = 0.0f;
            //for ( int v = 0; v < 66; ++v ) {
            //    size_t idx = 0 * 128 * 66 + 0 * 66 + v;
            //    sum += output_grad_cpu.data()[ idx ];
            //}
            //std::cout << "Sum of gradients (should be ~0): " << sum << "\n";

            //// Count how many are negative
            //int neg_count = 0;
            //for ( size_t i = 0; i < 32 * 128 * 66; ++i ) {
            //    if ( output_grad_cpu.data()[ i ] < 0 ) neg_count++;
            //}
            //std::cout << "Number of negative gradients: " << neg_count
            //    << " (should be " << (32 * 128) << ")\n";

            copy( output_grad_cpu, output_grad );

            model->zeroGradients();

            model->backward( input_batch, output_grad );
            
            // Ensure all gradients are ready before optimizer step
            model->synchronize();

            // Clip gradients to avoid spikes
            //clipGradients<TDeviceType, TDataType>( model.get(), 10.0f );

            // TEMPORARY DIAGNOSTIC: Check if gradients are actually being set
            /*if ( batches == 1 && epoch == 0 )
            {
                try
                {
                    auto grads = model->getGradients();
                    std::cout << "\n=== CHECKING MODEL GRADIENTS AFTER BACKWARD ===\n";
                    std::cout << "Total gradient tensors: " << grads.size() << "\n";

                    size_t zero_count = 0;
                    size_t nonzero_count = 0;

                    for ( size_t i = 0; i < std::min( grads.size(), size_t( 5 ) ); ++i )
                    {
                        auto* grad_fp32 = dynamic_cast<Tensor<TDataType, DeviceMR>*>( grads[ i ] );
                        if ( grad_fp32 )
                        {
                            auto host_grad = toHost<TDataType>( *grad_fp32 );

                            double sum = 0.0;
                            size_t zeros = 0;
                            for ( size_t j = 0; j < host_grad.size(); ++j )
                            {
                                double val = static_cast<double>( host_grad.data()[ j ] );
                                sum += std::abs( val );
                                if ( val == 0.0 ) zeros++;
                            }

                            std::cout << "Gradient " << i << ": size=" << host_grad.size()
                                << " zeros=" << zeros
                                << " mean_abs=" << (sum / host_grad.size())
                                << "\n";

                            if ( zeros == host_grad.size() ) zero_count++;
                            else nonzero_count++;
                        }
                    }

                    std::cout << "Summary: " << nonzero_count << " non-zero, "
                        << zero_count << " all-zero gradient tensors (first 5)\n";
                    std::cout << "=== END GRADIENT CHECK ===\n\n";
                }
                catch ( const std::exception& e )
                {
                    std::cerr << "Gradient check failed: " << e.what() << "\n";
                }
            }*/

            optimizer->step();
            model->synchronize();

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

        // Log parameter and gradient norms (per-epoch summary)
        try
        {
            logParameterAndGradientNorms<TDeviceType, TDataType>( model.get() );
        }
        catch ( const std::exception& e )
        {
            std::cerr << "Warning: failed to log parameter/gradient norms: " << e.what() << std::endl;
        }

        // Generate sample every N epochs (or never if sample_every_n_epochs == 0)
        if (config.sample_every_n_epochs > 0 && 
            ((epoch + 1) % config.sample_every_n_epochs == 0 || epoch + 1 == config.epochs))
        {
            std::cout << "\n--- Generated Sample (Epoch " << (epoch + 1) << ") ---" << std::endl;
            
            try
            {
                std::string sample = generateSample<TDeviceType, TDataType, THostMR>(
                    model.get(),
                    vocab,
                    device_id,
                    config,
                    epoch );
                std::cout << sample << std::endl;
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error generating sample: " << e.what() << std::endl;
            }
            
            std::cout << "--- End Sample ---\n" << std::endl;
        }
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
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}