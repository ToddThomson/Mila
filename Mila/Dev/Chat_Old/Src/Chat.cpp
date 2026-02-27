#include <time.h>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <format>
#include <iomanip>

import Mila;

import Gpt2.Transformer;
import Gpt2.CheckpointReader;
import Gpt2App.Gpt2DataLoader;
import Gpt2App.Gpt2Tokenizer;
import Gpt2App.Gpt2Config;

namespace
{
    struct Gpt2TrainConfig
    {
        std::string train_data = "data/datasets/tinyshakespeare/tiny_shakespeare_train.bin";
        std::string val_data = "data/datasets/tinyshakespeare/tiny_shakespeare_val.bin";
        size_t batch_size = 4;
        size_t seq_length = 64;
        size_t epochs = 1;
        float learning_rate = 3e-4f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float weight_decay = 0.0f;
        std::string device = "CUDA";
        int sample_every = 20;
    };

    // Cross-entropy loss and gradient (host CPU, float)
    static float sequenceCrossEntropyLossCpu(
        const Mila::Dnn::Tensor<Mila::Dnn::dtype_t::FP32, Mila::Dnn::CpuMemoryResource>& logits,
        const Mila::Dnn::Tensor<Mila::Dnn::dtype_t::INT32, Mila::Dnn::CpuMemoryResource>& targets )
    {
        using namespace Mila::Dnn;
        size_t B = logits.shape()[0];
        size_t T = logits.shape()[1];
        size_t V = logits.shape()[2];

        double total_nll = 0.0;
        size_t count = 0;

        const float* lptr = reinterpret_cast<const float*>(logits.rawData());
        const int32_t* tptr = reinterpret_cast<const int32_t*>(targets.rawData());

        for ( size_t b = 0; b < B; ++b )
        {
            for ( size_t t = 0; t < T; ++t )
            {
                size_t base = (b * T + t) * V;

                // find max for numerical stability
                float mx = lptr[ base ];
                for ( size_t v = 1; v < V; ++v ) mx = std::max( mx, lptr[ base + v ] );

                double denom = 0.0;
                for ( size_t v = 0; v < V; ++v ) denom += std::exp( static_cast<double>( lptr[ base + v ] - mx ) );

                int32_t target = tptr[ b * T + t ];
                double num = std::exp( static_cast<double>( lptr[ base + target ] - mx ) );

                double prob = num / denom;
                total_nll += -std::log( std::max( prob, 1e-12 ) );
                ++count;
            }
        }

        if ( count == 0 ) return 0.0f;

        return static_cast<float>( total_nll / static_cast<double>( count ) );
    }

    static void sequenceCrossEntropyGradientCpu(
        const Mila::Dnn::Tensor<Mila::Dnn::dtype_t::FP32, Mila::Dnn::CpuMemoryResource>& logits,
        const Mila::Dnn::Tensor<Mila::Dnn::dtype_t::INT32, Mila::Dnn::CpuMemoryResource>& targets,
        Mila::Dnn::Tensor<Mila::Dnn::dtype_t::FP32, Mila::Dnn::CpuMemoryResource>& out_grad )
    {
        using namespace Mila::Dnn;
        size_t B = logits.shape()[0];
        size_t T = logits.shape()[1];
        size_t V = logits.shape()[2];

        const float* lptr = reinterpret_cast<const float*>(logits.rawData());
        int32_t* tptr = const_cast<int32_t*>( reinterpret_cast<const int32_t*>(targets.rawData()) );
        float* gptr = reinterpret_cast<float*>(out_grad.rawData());

        // compute per-position softmax then subtract 1 on target index, normalize by total tokens
        size_t total_positions = B * T;
        const float norm = 1.0f / static_cast<float>( total_positions );

        for ( size_t b = 0; b < B; ++b )
        {
            for ( size_t t = 0; t < T; ++t )
            {
                size_t base = (b * T + t) * V;

                // numerically stable softmax
                float mx = lptr[ base ];
                for ( size_t v = 1; v < V; ++v ) mx = std::max( mx, lptr[ base + v ] );

                double denom = 0.0;
                for ( size_t v = 0; v < V; ++v ) denom += std::exp( static_cast<double>( lptr[ base + v ] - mx ) );

                int32_t target = tptr[ b * T + t ];

                for ( size_t v = 0; v < V; ++v )
                {
                    double prob = std::exp( static_cast<double>( lptr[ base + v ] - mx ) ) / denom;
                    double grad = prob - ( static_cast<int32_t>( v ) == target ? 1.0 : 0.0 );
                    gptr[ base + v ] = static_cast<float>( grad * norm );
                }
            }
        }
    }
} // namespace

int main( int argc, char* argv[] )
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Gpt2;

    try
    {
        Mila::initialize();

        // Simple CLI (reuse defaults or minimal parsing)
        Gpt2TrainConfig cfg;

        for ( int i = 1; i + 1 < argc; i += 2 )
        {
            std::string opt = argv[ i ];
            std::string val = argv[ i + 1 ];
            if ( opt == "-b" ) cfg.batch_size = static_cast<size_t>( std::stoi( val ) );
            else if ( opt == "-t" ) cfg.seq_length = static_cast<size_t>( std::stoi( val ) );
            else if ( opt == "-c" ) cfg.device = val;
            else if ( opt == "-i" ) cfg.train_data = val;
        }

        std::cout << "Gpt2 training (minimal) - using device: " << cfg.device << std::endl;

        // Choose device
        DeviceType dev_type = Compute::toDeviceType( cfg.device );

        if ( dev_type == DeviceType::Cuda )
        {
            try
            {
                // run on CUDA
                DeviceId device_id = Device::getDeviceId<DeviceType::Cuda>( 0 );

                // Data loader
                auto train_loader = Gpt2App::Gpt2DataLoader<int>( cfg.train_data, cfg.batch_size, cfg.seq_length, 0, 1, true );

                // Tokenizer (load vocab)
                Gpt2App::Tokenizer tokenizer( "data/models/gpt2/gpt2_tokenizer.bin" );

                // Build transformer config (match fields used in GptTransformer)
                GptTransformerConfig model_cfg;
                model_cfg.vocab_size = tokenizer.getVocabularySize();
                model_cfg.max_seq_length = static_cast<int>( cfg.seq_length );
                model_cfg.embedding_dim = 768; // choose reasonable default or expose via CLI
                model_cfg.num_heads = 12;
                model_cfg.num_layers = 12;
                model_cfg.mlp_hidden_dim = 3072;

                // Create model
                auto model = std::make_unique< GptTransformer<DeviceType::Cuda, dtype_t::FP32> >(
                    "gpt2", model_cfg, device_id );

                shape_t input_shape = { static_cast<int64_t>( cfg.batch_size ), static_cast<int64_t>( cfg.seq_length ) };
                model->build( input_shape );

                std::cout << "Built GPT-2 network:" << std::endl;
                std::cout << model->toString() << std::endl;

                model->setTraining( true );

                auto optimizer = model->createOptimizer< Mila::Dnn::Optimizers::AdamWOptimizer<DeviceType::Cuda, dtype_t::FP32> >(
                    Mila::Dnn::Optimizers::AdamWConfig()
                        .withLearningRate( cfg.learning_rate )
                        .withBeta1( cfg.beta1 )
                        .withBeta2( cfg.beta2 )
                        .withEpsilon( cfg.eps )
                        .withWeightDecay( cfg.weight_decay )
                );

                std::cout << "Optimizer parameter groups: " << optimizer->getParameterCount() << std::endl;

                // Prepare tensors
                using DeviceMR = typename DeviceTypeTraits<DeviceType::Cuda>::memory_resource;
                Tensor<dtype_t::INT32, DeviceMR> input_batch( device_id, input_shape );
                Tensor<dtype_t::INT32, DeviceMR> target_batch( device_id, input_shape );

                shape_t logits_shape = { static_cast<int64_t>( cfg.batch_size ), static_cast<int64_t>( cfg.seq_length ), static_cast<int64_t>( model_cfg.vocab_size ) };

                Tensor<dtype_t::FP32, CpuMemoryResource> logits_cpu( Device::Cpu(), logits_shape );
                Tensor<dtype_t::INT32, CpuMemoryResource> targets_cpu( Device::Cpu(), input_shape );
                Tensor<dtype_t::FP32, CpuMemoryResource> output_grad_cpu( Device::Cpu(), logits_shape );
                Tensor<dtype_t::FP32, DeviceMR> output_grad_dev( device_id, logits_shape );

                std::cout << "Starting training loop (epochs=" << cfg.epochs << ")..." << std::endl;

                for ( size_t epoch = 0; epoch < cfg.epochs; ++epoch )
                {
                    train_loader.reset();
                    size_t batch_idx = 0;
                    double epoch_loss = 0.0;

                    auto epoch_start = std::chrono::high_resolution_clock::now();

                    while ( train_loader.hasNext() )
                    {
                        train_loader.next_batch();

                        // copy data from loader to device tensors
                        copy( train_loader.inputs(), input_batch );
                        copy( train_loader.targets(), target_batch );

                        auto& logits_dev = model->forward( input_batch );
                        model->getExecutionContext()->synchronize();

                        // Move logits & targets to CPU
                        copy( logits_dev, logits_cpu );
                        copy( target_batch, targets_cpu );

                        // Compute loss on CPU
                        float batch_loss = sequenceCrossEntropyLossCpu( logits_cpu, targets_cpu );
                        epoch_loss += batch_loss;

                        // Compute gradient on CPU and copy to device
                        std::memset( output_grad_cpu.rawData(), 0, output_grad_cpu.getStorageSize() );
                        sequenceCrossEntropyGradientCpu( logits_cpu, targets_cpu, output_grad_cpu );

                        copy( output_grad_cpu, output_grad_dev );

                        // Backprop
                        model->zeroGradients();
                        model->backward( input_batch, output_grad_dev );
                        model->getExecutionContext()->synchronize();

                        optimizer->step();
                        model->getExecutionContext()->synchronize();

                        ++batch_idx;

                        if ( batch_idx % 50 == 0 )
                        {
                            std::cout << std::format( "Epoch {} batch {} - loss {:.4f}\n", epoch + 1, batch_idx, batch_loss );
                        }
                    }

                    auto epoch_end = std::chrono::high_resolution_clock::now();
                    double epoch_time = std::chrono::duration<double>( epoch_end - epoch_start ).count();

                    std::cout << std::format( "Epoch {}/{} finished - avg loss {:.4f} - time {:.2f}s\n",
                        epoch + 1, cfg.epochs, epoch_loss / std::max<size_t>( 1, batch_idx ), epoch_time );
                }
            }
            catch ( const std::exception& e )
            {
                std::cerr << "CUDA run error: " << e.what() << std::endl;
                return 1;
            }
        }
        else
        {
            // CPU path (mirror CUDA but using DeviceType::Cpu)
            DeviceId device_id = Device::getDeviceId<DeviceType::Cpu>( 0 );

            auto train_loader = Gpt2App::Gpt2DataLoader<int>( cfg.train_data, cfg.batch_size, cfg.seq_length, 0, 1, true );
            Gpt2App::Tokenizer tokenizer( "data/models/gpt2/gpt2_tokenizer.bin" );

            GptTransformerConfig model_cfg;
            model_cfg.vocab_size = tokenizer.getVocabularySize();
            model_cfg.max_seq_length = static_cast<int>( cfg.seq_length );
            model_cfg.embedding_dim = 512;
            model_cfg.num_heads = 8;
            model_cfg.num_layers = 6;
            model_cfg.mlp_hidden_dim = 2048;

            auto model = std::make_unique< GptTransformer<DeviceType::Cpu, dtype_t::FP32> >(
                "gpt2", model_cfg, device_id );

            shape_t input_shape = { static_cast<int64_t>( cfg.batch_size ), static_cast<int64_t>( cfg.seq_length ) };
            model->build( input_shape );

            model->setTraining( true );

            auto optimizer = model->createOptimizer< Mila::Dnn::Optimizers::AdamWOptimizer<DeviceType::Cpu, dtype_t::FP32> >(
                Mila::Dnn::Optimizers::AdamWConfig()
                    .withLearningRate( cfg.learning_rate )
                    .withBeta1( cfg.beta1 )
                    .withBeta2( cfg.beta2 )
                    .withEpsilon( cfg.eps )
                    .withWeightDecay( cfg.weight_decay )
            );

            using DeviceMR = typename DeviceTypeTraits<DeviceType::Cpu>::memory_resource;
            Tensor<dtype_t::INT32, DeviceMR> input_batch( device_id, input_shape );
            Tensor<dtype_t::INT32, DeviceMR> target_batch( device_id, input_shape );

            shape_t logits_shape = { static_cast<int64_t>( cfg.batch_size ), static_cast<int64_t>( cfg.seq_length ), static_cast<int64_t>( model_cfg.vocab_size ) };

            Tensor<dtype_t::FP32, CpuMemoryResource> logits_cpu( Device::Cpu(), logits_shape );
            Tensor<dtype_t::INT32, CpuMemoryResource> targets_cpu( Device::Cpu(), input_shape );
            Tensor<dtype_t::FP32, CpuMemoryResource> output_grad_cpu( Device::Cpu(), logits_shape );
            Tensor<dtype_t::FP32, DeviceMR> output_grad_dev( device_id, logits_shape );

            for ( size_t epoch = 0; epoch < cfg.epochs; ++epoch )
            {
                train_loader.reset();
                size_t batch_idx = 0;

                while ( train_loader.hasNext() )
                {
                    train_loader.next_batch();

                    copy( train_loader.inputs(), input_batch );
                    copy( train_loader.targets(), target_batch );

                    auto& logits_dev = model->forward( input_batch );
                    model->getExecutionContext()->synchronize();

                    copy( logits_dev, logits_cpu );
                    copy( target_batch, targets_cpu );

                    float batch_loss = sequenceCrossEntropyLossCpu( logits_cpu, targets_cpu );

                    std::memset( output_grad_cpu.rawData(), 0, output_grad_cpu.getStorageSize() );
                    sequenceCrossEntropyGradientCpu( logits_cpu, targets_cpu, output_grad_cpu );

                    copy( output_grad_cpu, output_grad_dev );

                    model->zeroGradients();
                    model->backward( input_batch, output_grad_dev );
                    model->getExecutionContext()->synchronize();

                    optimizer->step();
                    model->getExecutionContext()->synchronize();

                    ++batch_idx;
                }

                std::cout << std::format( "Epoch {}/{} done\n", epoch + 1, cfg.epochs );
            }
        }
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}