
export module Bard.Config;

import std;
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
        // Root data directory containing the dataset subfolders (encoded, vocabularies, ...).
        // Defaults to the repository layout under Data/Datasets/Shakespeare; main() overrides
        // this with the absolute MILA_DATASETS_DIR path so the sample finds data regardless of
        // the working directory it is launched from.
        std::string data_dir = "./Data/Datasets/Shakespeare";
        TokenizerType tokenizer = TokenizerType::Char;
        std::int64_t batch_size = 32;
        std::int64_t seq_length = 128;
        std::size_t epochs = 200;
        float learning_rate = 1e-4f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
        float weight_decay = 0.01f;
        TensorDataType precision = TensorDataType::FP32;

        std::int64_t embedding_dim = 384;
        std::int64_t num_heads = 6;
        std::int64_t num_layers = 6;
        std::int64_t mlp_hidden_dim = 1536;

        float sample_temperature = 0.8f;
        std::int64_t sample_length = 300;
        std::size_t sample_every_n_epochs = 1;
        std::string sample_prompt = "ROMEO:\n";

        float lr_decay = 1.0f;
        std::size_t lr_decay_every_n_epochs = 0;
    };
}