module;

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

export module CharLM.CharDataLoader;

import Mila;

namespace Mila::CharLM
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Data;

    namespace fs = std::filesystem;

    /**
     * @brief Character-level text data loader for language modeling.
     *
     * Loads preprocessed token sequences. Token indices are always INT32.
     * Supports CPU and pinned host memory resources for batched I/O.
     */
    export template<typename TMemoryResource>
        requires (std::is_same_v<TMemoryResource, CudaPinnedMemoryResource> ||
                  std::is_same_v<TMemoryResource, CpuMemoryResource>)
    class CharDataLoader : public DatasetReader<TensorDataType::INT32, TensorDataType::INT32, TMemoryResource>
    {
    public:
        using BaseLoader = DatasetReader<TensorDataType::INT32, TensorDataType::INT32, TMemoryResource>;
        using HostType = typename TensorHostTypeMap<TensorDataType::INT32>::host_type;
        using TensorType = Tensor<TensorDataType::INT32, TMemoryResource>;

        /**
         * @brief Constructs character-level data loader from preprocessed files.
         *
         * @param text_file_base Base path to preprocessed files (without extensions)
         * @param batch_size Number of sequences per batch
         * @param seq_length Length of each sequence (context window)
         * @param is_training Whether to shuffle sequences (training mode)
         * @param device Compute device for tensor allocation
         * @param stride Step size for sliding window (default: seq_length for non-overlapping)
         */
        CharDataLoader(
            const std::string& text_file_base,
            int64_t batch_size,
            int64_t seq_length,
            bool is_training,
            DeviceId device,
            int64_t stride = -1 )
            : BaseLoader( batch_size ),
            seq_length_( seq_length ),
            is_training_( is_training ),
            device_( validateDeviceId( device ) ),
            stride_( stride > 0 ? stride : seq_length )
        {
            if (seq_length_ <= 0)
            {
                throw std::invalid_argument( "Sequence length must be positive" );
            }

            vocab_file_ = text_file_base + ".vocab";
            tokens_file_ = text_file_base + ".tokens";

            // Verify preprocessed files exist
            if (!fs::exists( vocab_file_ ))
            {
                throw std::runtime_error(
                    "Vocabulary file not found: " + vocab_file_ + "\n" +
                    "Please run CharPreprocessor first to create preprocessed files." );
            }

            if (!fs::exists( tokens_file_ ))
            {
                throw std::runtime_error(
                    "Tokens file not found: " + tokens_file_ + "\n" +
                    "Please run CharPreprocessor first to create preprocessed files." );
            }

            // Read vocabulary size from .vocab file header (don't load full vocabulary)
            loadVocabSize();

            // Load tokenized data and validate tokens against vocab size
            loadTokens();
            createSequences();

            if (is_training_)
            {
                shuffleIndices();
            }

            num_batches_ = num_sequences_ / batch_size;

            // Allocate batch tensors
            input_tensor_ = std::make_shared<TensorType>(
                device_,
                shape_t{ batch_size, seq_length_ } );

            target_tensor_ = std::make_shared<TensorType>(
                device_,
                shape_t{ batch_size, seq_length_ } );

            std::cout << "CharDataLoader initialized:" << std::endl;
            std::cout << "  Base path: " << text_file_base << std::endl;
            std::cout << "  Total tokens: " << tokens_.size() << std::endl;
            std::cout << "  Vocabulary size: " << vocab_size_ << std::endl;
            std::cout << "  Sequence length: " << seq_length_ << std::endl;
            std::cout << "  Stride: " << stride_ << std::endl;
            std::cout << "  Total sequences: " << num_sequences_ << std::endl;
            std::cout << "  Batches: " << num_batches_ << std::endl;
            std::cout << "  Device: " << device_.toString() << std::endl;
        }

        int64_t numBatches() const override
        {
            return num_batches_;
        }

        void reset() override
        {
            BaseLoader::reset();

            if (is_training_)
            {
                shuffleIndices();
            }
        }

        void nextBatch() override
        {
            if (this->currentBatch() >= num_batches_)
            {
                return;
            }

            size_t batch_start = this->currentBatch() * this->batchSize();

            for (size_t i = 0; i < this->batchSize(); ++i)
            {
                size_t seq_start_idx = indices_[batch_start + i];

                // Copy input sequence (tokens[start : start+seq_length])
                for (int64_t t = 0; t < seq_length_; ++t)
                {
                    const int32_t token = tokens_[seq_start_idx + t];

                    // Safety check: ensure token still within vocabulary bounds
                    if (token < 0 || static_cast<uint32_t>(token) >= vocab_size_)
                    {
                        throw std::runtime_error(
                            "CharDataLoader::nextBatch - token index out of range at global token position " +
                            std::to_string(seq_start_idx + t) + ": value=" + std::to_string(token) +
                            ", vocab_size=" + std::to_string(vocab_size_) );
                    }

                    input_tensor_->data()[i * seq_length_ + t] =
                        static_cast<HostType>( token );
                }

                // Copy target sequence (tokens[start+1 : start+seq_length+1])
                for (int64_t t = 0; t < seq_length_; ++t)
                {
                    const int32_t token = tokens_[seq_start_idx + t + 1];

                    // Safety check: ensure token still within vocabulary bounds
                    if (token < 0 || static_cast<uint32_t>(token) >= vocab_size_)
                    {
                        throw std::runtime_error(
                            "CharDataLoader::nextBatch - target token index out of range at global token position " +
                            std::to_string(seq_start_idx + t + 1) + ": value=" + std::to_string(token) +
                            ", vocab_size=" + std::to_string(vocab_size_) );
                    }

                    target_tensor_->data()[i * seq_length_ + t] =
                        static_cast<HostType>( token );
                }
            }

            this->incrementBatch();
        }

        TensorType& inputs() override
        {
            return *input_tensor_;
        }

        const TensorType& inputs() const override
        {
            return *input_tensor_;
        }

        TensorType& targets() override
        {
            return *target_tensor_;
        }

        const TensorType& targets() const override
        {
            return *target_tensor_;
        }

        /**
         * @brief Gets the vocabulary size.
         */
        size_t vocabSize() const
        {
            return vocab_size_;
        }

        /**
         * @brief Gets the sequence length.
         */
        int64_t sequenceLength() const
        {
            return seq_length_;
        }

    private:
        DeviceId device_;
        bool is_training_;
        int64_t seq_length_;
        int64_t stride_;
        size_t num_sequences_;
        int64_t num_batches_;
        size_t vocab_size_;

        std::string vocab_file_;
        std::string tokens_file_;
        std::vector<int32_t> tokens_;               // Loaded from preprocessed file
        std::vector<size_t> indices_;               // Start indices for sequences

        std::shared_ptr<TensorType> input_tensor_;
        std::shared_ptr<TensorType> target_tensor_;

        /**
         * @brief Validates device and ensures it matches memory resource requirements.
         */
        static DeviceId validateDeviceId(
            DeviceId device )
        {
            if constexpr (std::is_same_v<TMemoryResource, CpuMemoryResource>)
            {
                if (device.type != DeviceType::Cpu)
                {
                    throw std::runtime_error(
                        "CpuMemoryResource requires CPU device, got " +
                        device.toString() );
                }
            }

            if constexpr (std::is_same_v<TMemoryResource, CudaPinnedMemoryResource>)
            {
                if (device.type != DeviceType::Cuda)
                {
                    throw std::runtime_error(
                        "CudaPinnedMemoryResource requires CUDA device, got " +
                        device.toString() );
                }
            }

            return device;
        }

        /**
         * @brief Loads vocabulary size from .vocab file header.
         */
        void loadVocabSize()
        {
            std::ifstream file( vocab_file_, std::ios::binary );
            if (!file)
            {
                throw std::runtime_error( "Cannot open vocabulary file: " + vocab_file_ );
            }

            // Read only the vocabulary size from header
            file.read( reinterpret_cast<char*>(&vocab_size_), sizeof( vocab_size_ ) );

            if (!file || vocab_size_ == 0)
            {
                throw std::runtime_error( "Invalid vocabulary file: " + vocab_file_ );
            }
        }

        /**
         * @brief Loads tokenized data from preprocessed binary file.
         *
         * Validates that each token index is within [0, vocab_size_).
         */
        void loadTokens()
        {
            std::ifstream file( tokens_file_, std::ios::binary );
            
            if (!file)
            {
                throw std::runtime_error( "Cannot open tokens file: " + tokens_file_ );
            }

            // Read header
            size_t num_tokens;
            file.read( reinterpret_cast<char*>(&num_tokens), sizeof( num_tokens ) );

            if (num_tokens == 0)
            {
                throw std::runtime_error( "Empty tokens file: " + tokens_file_ );
            }

            // Read tokens
            tokens_.resize( num_tokens );
            file.read( reinterpret_cast<char*>(tokens_.data()), num_tokens * sizeof( int32_t ) );

            if (!file)
            {
                throw std::runtime_error( "Error reading tokens file: " + tokens_file_ );
            }

            // Validate tokens are within vocabulary bounds
            for ( size_t i = 0; i < tokens_.size(); ++i )
            {
                int32_t tok = tokens_[i];

                if ( tok < 0 || static_cast<uint32_t>(tok) >= vocab_size_ )
                {
                    throw std::runtime_error(
                        "Invalid token index in tokens file at position " + std::to_string(i) +
                        ": value=" + std::to_string(tok) +
                        ", vocab_size=" + std::to_string(vocab_size_) +
                        ". Ensure tokens were preprocessed with the same vocabulary." );
                }
            }
        }

        /**
         * @brief Creates sequence start indices for sliding window approach.
         */
        void createSequences()
        {
            // Need seq_length+1 tokens for input and shifted target
            if (tokens_.size() < static_cast<size_t>(seq_length_ + 1))
            {
                throw std::runtime_error(
                    "Token sequence too short for sequence length " + std::to_string( seq_length_ ) );
            }

            indices_.clear();
            size_t max_start = tokens_.size() - seq_length_;

            // Store sequence start indices instead of full sequences
            for (size_t start = 0; start < max_start; start += stride_)
            {
                indices_.push_back( start );
            }

            num_sequences_ = indices_.size();
        }

        /**
         * @brief Shuffles sequence indices for training.
         */
        void shuffleIndices()
        {
            std::random_device rd;
            std::mt19937 gen( rd() );
            std::shuffle( indices_.begin(), indices_.end(), gen );
        }
    };
}