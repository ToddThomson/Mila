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
    float learning_rate = 1e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.01f;
    DeviceType compute_device = DeviceType::Cuda;
    TensorDataType precision = TensorDataType::FP32;
    ComputePrecision::Policy precisionPolicy = ComputePrecision::Policy::Auto;

    int64_t embedding_dim = 384;
    int64_t num_heads = 6;
    int64_t num_layers = 6;
    int64_t mlp_hidden_dim = 1536;

    float sample_temperature = 0.8f;
    int64_t sample_length = 300;
    size_t sample_every_n_epochs = 1;
    std::string sample_prompt = "ROMEO:\n";

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
    for ( int i = 1; i < argc; i++ )
    {
        std::string arg = argv[ i ];

        if ( arg == "--help" )
        {
            printUsage();
            return false;
        }

        if ( arg == "--data-file" && i + 1 < argc )
        {
            config.data_file = argv[ ++i ];
            continue;
        }

        if ( arg == "--batch-size" && i + 1 < argc )
        {
            config.batch_size = std::stoi( argv[ ++i ] );
            continue;
        }

        if ( arg == "--seq-length" && i + 1 < argc )
        {
            config.seq_length = std::stoi( argv[ ++i ] );
            continue;
        }

        if ( arg == "--epochs" && i + 1 < argc )
        {
            config.epochs = std::stoi( argv[ ++i ] );
            continue;
        }

        if ( arg == "--learning-rate" && i + 1 < argc )
        {
            config.learning_rate = std::stof( argv[ ++i ] );
            continue;
        }

        if ( arg == "--beta1" && i + 1 < argc )
        {
            config.beta1 = std::stof( argv[ ++i ] );
            continue;
        }

        if ( arg == "--beta2" && i + 1 < argc )
        {
            config.beta2 = std::stof( argv[ ++i ] );
            continue;
        }

        if ( arg == "--weight-decay" && i + 1 < argc )
        {
            config.weight_decay = std::stof( argv[ ++i ] );
            continue;
        }

        if ( arg == "--device" && i + 1 < argc )
        {
            std::string device = argv[ ++i ];
            if ( device == "CPU" )
            {
                config.compute_device = DeviceType::Cpu;
            }
            else if ( device == "cuda" )
            {
                config.compute_device = DeviceType::Cuda;
            }
            else
            {
                std::cerr << "Unknown device type: " << device << ". Using default: cuda" << std::endl;
            }
            continue;
        }

        if ( arg == "--precision-policy" && i + 1 < argc )
        {
            std::string precision = argv[ ++i ];
            if ( precision == "auto" )
            {
                config.precisionPolicy = ComputePrecision::Policy::Auto;
            }
            else if ( precision == "performance" )
            {
                config.precisionPolicy = ComputePrecision::Policy::Performance;
            }
            else if ( precision == "accuracy" )
            {
                config.precisionPolicy = ComputePrecision::Policy::Accuracy;
            }
            else if ( precision == "disabled" )
            {
                config.precisionPolicy = ComputePrecision::Policy::Native;
            }
            else
            {
                std::cerr << "Unknown precision policy: " << precision << ". Using default: auto" << std::endl;
            }
            continue;
        }

        if ( arg == "--embedding-dim" && i + 1 < argc )
        {
            config.embedding_dim = std::stoi( argv[ ++i ] );
            continue;
        }

        if ( arg == "--num-heads" && i + 1 < argc )
        {
            config.num_heads = std::stoi( argv[ ++i ] );
            continue;
        }

        if ( arg == "--num-layers" && i + 1 < argc )
        {
            config.num_layers = std::stoi( argv[ ++i ] );
            continue;
        }

        if ( arg == "--sample-temperature" && i + 1 < argc )
        {
            config.sample_temperature = std::stof( argv[ ++i ] );
            continue;
        }

        if ( arg == "--sample-length" && i + 1 < argc )
        {
            config.sample_length = std::stoi( argv[ ++i ] );
            continue;
        }

        if ( arg == "--sample-every" && i + 1 < argc )
        {
            config.sample_every_n_epochs = std::stoi( argv[ ++i ] );
            continue;
        }

        if ( arg == "--lr-decay" && i + 1 < argc )
        {
            config.lr_decay = std::stof( argv[ ++i ] );
            continue;
        }

        if ( arg == "--lr-decay-every" && i + 1 < argc )
        {
            config.lr_decay_every_n_epochs = static_cast<size_t>(std::stoll( argv[ ++i ] ));
            continue;
        }

        std::cerr << "Unknown option: " << arg << std::endl;
        printUsage();
        return false;
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

    if ( !fs::exists( config.data_file ) )
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
    Tensor<TDataType, CpuMemoryResource>& output_grad,
    int pad_id )
{
    using HostType = typename TensorHostTypeMap<TDataType>::host_type;

    size_t batch_size = logits.shape()[ 0 ];
    size_t seq_length = logits.shape()[ 1 ];
    size_t vocab_size = logits.shape()[ 2 ];

    // Count non-pad tokens
    size_t valid_count = 0;
    for ( size_t b = 0; b < batch_size; ++b )
    {
        for ( size_t t = 0; t < seq_length; ++t )
        {
            int32_t tgt = targets.data()[ b * seq_length + t ];
            if ( pad_id < 0 || tgt != pad_id )
            {
                ++valid_count;
            }
        }
    }

    if ( valid_count == 0 )
    {
        std::fill_n( output_grad.data(), batch_size * seq_length * vocab_size, HostType(0) );
        return;
    }

    const float norm = 1.0f / static_cast<float>( valid_count );

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

            int32_t target_idx = targets.data()[ b * seq_length + t ];

            if ( pad_id >= 0 && target_idx == pad_id )
            {
                for ( size_t v = 0; v < vocab_size; ++v )
                {
                    size_t idx = b * seq_length * vocab_size + t * vocab_size + v;
                    output_grad.data()[ idx ] = HostType(0);
                }
                continue;
            }

            for ( size_t v = 0; v < vocab_size; ++v )
            {
                size_t idx = b * seq_length * vocab_size + t * vocab_size + v;
                float prob = std::exp( static_cast<float>( logits.data()[ idx ] ) - max_logit ) / denom;
                float target = ( static_cast<int32_t>( v ) == target_idx ) ? 1.0f : 0.0f;
                output_grad.data()[ idx ] = static_cast<HostType>( (prob - target) * norm );
            }
        }
    }
}

template<TensorDataType TDataType>
float sequenceCrossEntropyLoss(
    const Tensor<TDataType, CpuMemoryResource>& logits,
    const Tensor<TensorDataType::INT32, CpuMemoryResource>& targets,
    int pad_id )
{
    size_t batch_size = logits.shape()[ 0 ];
    size_t seq_length = logits.shape()[ 1 ];
    size_t vocab_size = logits.shape()[ 2 ];

    double total_nll = 0.0;
    size_t valid_count = 0;

    for ( size_t b = 0; b < batch_size; ++b )
    {
        for ( size_t t = 0; t < seq_length; ++t )
        {
            int32_t target_idx = targets.data()[ b * seq_length + t ];
            if ( pad_id >= 0 && target_idx == pad_id )
            {
                continue;
            }

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

            size_t logit_idx = b * seq_length * vocab_size + t * vocab_size + static_cast<size_t>( target_idx );
            float prob = std::exp( static_cast<float>( logits.data()[ logit_idx ] ) - max_logit ) / denom;
            total_nll += -std::log( std::max( prob, 1e-12f ) );

            ++valid_count;
        }
    }

    if ( valid_count == 0 )
    {
        return 0.0f;
    }

    return static_cast<float>( total_nll / static_cast<double>( valid_count ) );
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

    bool was_training = model->isTraining();
    model->setTraining( false );

    std::vector<int32_t> prompt_tokens;
    for ( char c : config.sample_prompt )
    {
        int idx = vocab.charToIndex( c );
        if ( idx < 0 )
        {
            if ( vocab.hasSpecialTokens() )
            {
                idx = vocab.unkTokenId();
            }
            else
            {
                throw std::runtime_error( "Prompt contains out-of-vocab character and no UNK token is defined." );
            }
        }
        prompt_tokens.push_back( idx );
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

    int32_t pad_value = vocab.hasSpecialTokens() ? vocab.padTokenId() : 0;

    for ( int64_t i = 0; i < config.sample_length; ++i )
    {
        int64_t context_start = std::max( int64_t( 0 ),
            static_cast<int64_t>( generated_tokens.size() ) - model_seq_length );
        int64_t actual_context_len = static_cast<int64_t>( generated_tokens.size() ) - context_start;

        std::fill_n( context_cpu.data(), static_cast<size_t>( model_batch_size * model_seq_length ), pad_value );

        for ( int64_t j = 0; j < actual_context_len; ++j )
        {
            context_cpu.data()[ j ] = generated_tokens[ context_start + j ];
        }

        copy( context_cpu, context_device );

        // DEBUG: Dump context_cpu
        // std::cout << context_cpu.toString( true );

        auto& logits_dev = model->forward( context_device );
        model->synchronize();

        copy( logits_dev, logits_cpu );

        int64_t last_real_pos = (actual_context_len > 0) ? (actual_context_len - 1) : (model_seq_length - 1);
        if ( last_real_pos < 0 ) last_real_pos = 0;

        size_t batch_0_last_token_offset = static_cast<size_t>( last_real_pos ) * vocab_size;

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

    if ( was_training )
    {
        model->setTraining( true );
    }

    return generated_text;
}

template<DeviceType TDeviceType, TensorDataType TDataType, typename THostMR>
    requires PrecisionSupportedOnDevice<TDataType, TDeviceType> &&
(std::is_same_v<THostMR, CudaPinnedMemoryResource> || std::is_same_v<THostMR, CpuMemoryResource>)
void trainCharLM( const CharLMConfig& config )
{
    using DeviceMR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
    DeviceId device_id = Device::getDeviceId<TDeviceType>( 0 );

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

    shape_t logits_shape = { config.batch_size, config.seq_length, static_cast<int64_t>(actual_vocab_size) };

    Tensor<TensorDataType::INT32, DeviceMR> input_batch( device_id, input_shape );
    Tensor<TensorDataType::INT32, DeviceMR> target_batch( device_id, input_shape );

    Tensor<TDataType, CpuMemoryResource> logits_cpu( Device::Cpu(), logits_shape );
    Tensor<TensorDataType::INT32, CpuMemoryResource> targets_cpu( Device::Cpu(), input_shape );
    Tensor<TDataType, CpuMemoryResource> output_grad_cpu( Device::Cpu(), logits_shape );
    Tensor<TDataType, DeviceMR> output_grad( device_id, logits_shape );

    std::cout << "Starting training for " << config.epochs << " epochs..." << std::endl;
    std::cout << "Total batches per epoch: " << train_loader.numBatches() << std::endl;

    for ( size_t epoch = 0; epoch < config.epochs; ++epoch )
    {
        train_loader.reset();

        float epoch_loss = 0.0f;
        size_t batches = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        while ( train_loader.hasNext() )
        {
            train_loader.nextBatch();

            copy( train_loader.inputs(), input_batch );
            copy( train_loader.targets(), target_batch );

            auto& output = model->forward( input_batch );
            model->synchronize();

            copy( output, logits_cpu );
            copy( target_batch, targets_cpu );

            int pad_id = train_loader.padTokenId();

            float batch_loss = sequenceCrossEntropyLoss( logits_cpu, targets_cpu, pad_id );
            float batch_perplexity = computePerplexity( batch_loss );

            zero( output_grad_cpu );
            sequenceCrossEntropyGradient( logits_cpu, targets_cpu, output_grad_cpu, pad_id );

            copy( output_grad_cpu, output_grad );

            model->zeroGradients();

            model->backward( input_batch, output_grad );

            model->synchronize();

            optimizer->step();
            model->synchronize();

            epoch_loss += batch_loss;
            batches++;

            if ( batches % 50 == 0 || batches == train_loader.numBatches() )
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

        if ( config.sample_every_n_epochs > 0 &&
            ((epoch + 1) % config.sample_every_n_epochs == 0 || epoch + 1 == config.epochs) )
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
            catch ( const std::exception& e )
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
        if ( !parseCommandLine( argc, argv, config ) )
        {
            return 1;
        }

        if ( config.compute_device == DeviceType::Cuda )
        {
            try
            {
                std::cout << "Using CUDA device" << std::endl;
                trainCharLM<DeviceType::Cuda, TensorDataType::FP32, CudaPinnedMemoryResource>( config );
            }
            catch ( const std::exception& e )
            {
                std::cerr << "CUDA error: " << e.what() << std::endl;
                throw;
            }
        }

        if ( config.compute_device == DeviceType::Cpu )
        {
            std::cout << "Using CPU device" << std::endl;
            trainCharLM<DeviceType::Cpu, TensorDataType::FP32, CpuMemoryResource>( config );
        }
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}