#include <iostream>
#include <string>
#include <filesystem>

import Mila;

import Bard.Config;
import Bard.Trainer;
//import Data.TokenizerType;

namespace fs = std::filesystem;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Compute;
using namespace Mila::Data;
using namespace Bard;

void printUsage()
{
    std::cout << "Usage: bard [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --data-dir <path>        Root data directory containing dataset subfolders (default: ./Data/Shakespeare)\n";
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
    std::cout << "  --tokenizer <bpe|char>   Tokenizer type to use for encoding/decoding (default: bpe)\n";
    std::cout << "  --help                   Show this help message\n";
}

bool parseCommandLine( int argc, char** argv, BardConfig& config )
{
    for ( int i = 1; i < argc; i++ )
    {
        std::string arg = argv[ i ];

        if ( arg == "--help" )
        {
            printUsage();
            return false;
        }

        if ( arg == "--data-dir" && i + 1 < argc )
        {
            config.data_dir = argv[ ++i ];
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

        if ( arg == "--tokenizer" && i + 1 < argc )
        {
            std::string tokenizer_arg = argv[ ++i ];
            auto tokenizer_type = stringToTokenizerType( tokenizer_arg );
            
            if ( tokenizer_type == TokenizerType::Unknown )
            {
                std::cerr << "Unknown tokenizer: " << tokenizer_arg << ". Using default: bpe" << std::endl;
            }
            else if ( tokenizer_type != TokenizerType::Bpe && tokenizer_type != TokenizerType::Char )
            {
                std::cerr << "Tokenizer \"" << tokenizer_arg << "\" is not supported by this sample. Using default: bpe" << std::endl;
            }
            else
            {
                config.tokenizer = tokenizer_type;
            }
            
            continue;
        }

        std::cerr << "Unknown option: " << arg << std::endl;
        printUsage();
        return false;
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Data dir: " << config.data_dir << std::endl;
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
    std::cout << "  Tokenizer: " << tokenizerTypeToString( config.tokenizer ) << std::endl;

    if ( !fs::exists( config.data_dir ) || !fs::is_directory( config.data_dir ) )
    {
        std::cerr << "Data directory not found: " << config.data_dir << std::endl;
        std::cerr << "Please provide the root data directory containing dataset subfolders (e.g. encoded/, vocabularies/)." << std::endl;

        return false;
    }

    return true;
}

int main( int argc, char** argv )
{
    try
    {
        std::cout << "Bard Language Model sample using Mila" << std::endl;
        std::cout << "=====================================" << std::endl;

        Mila::initialize();

        BardConfig config;
        if ( !parseCommandLine( argc, argv, config ) )
        {
            return 1;
        }

        if ( config.compute_device == DeviceType::Cuda )
        {
            std::cout << "Using CUDA device" << std::endl;
            train<DeviceType::Cuda, TensorDataType::FP32, CudaPinnedMemoryResource>( config );
        }
        else if ( config.compute_device == DeviceType::Cpu )
        {
            std::cout << "Using CPU device" << std::endl;
            train<DeviceType::Cpu, TensorDataType::FP32, CpuMemoryResource>( config );
        }
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Bard training error: " << e.what() << std::endl;
        return 0;
    }

    return 0;
}