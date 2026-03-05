module;
#include <string>

export module Bard.Config;

import Mila;

namespace Bard
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Data;

    /**
     * @brief Configuration for Bard training CLI and trainer.
     */
    export struct BardConfig
    {
        // Root data directory containing the dataset subfolders (encoded, vocabularies, ...)
        // Defaults to the repository layout under Data/Shakespeare
        std::string data_dir = "./Data/Shakespeare";
        TokenizerType tokenizer = TokenizerType::Char;
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
}