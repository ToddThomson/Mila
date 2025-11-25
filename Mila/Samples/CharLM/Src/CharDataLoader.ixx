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
     * Pure data loader that efficiently loads preprocessed token sequences.
     * Does NOT handle vocabulary or text encoding/decoding - only loads
     * raw token indices from preprocessed binary files.
     *
     * Key features:
     * - Fast loading from preprocessed binary files
     * - Sliding window sequence generation with configurable stride
     * - Optional shuffling for training mode
     * - Efficient batch processing with pre-allocated tensors
     * - Support for both CPU and pinned memory resources
     *
     * Prerequisites:
     * - <text_file>.vocab must exist (for reading vocab size only)
     * - <text_file>.tokens must exist (tokenized data)
     *
     * Sequence structure:
     * - Input: [t, t+1, ..., t+seq_len-1]
     * - Target: [t+1, t+2, ..., t+seq_len] (shifted by 1 for next-token prediction)
     *
     * @tparam TInput Tensor data type for token indices (typically FP32 for compatibility)
     * @tparam TMemoryResource Memory resource type (CpuMemoryResource or CudaPinnedMemoryResource)
     */
    export template<TensorDataType TInput, typename TMemoryResource>
        requires PrecisionSupportedOnDevice<TInput, DeviceType::Cpu> &&
    (std::is_same_v<TMemoryResource, CudaPinnedMemoryResource> ||
        std::is_same_v<TMemoryResource, CpuMemoryResource>)
        class CharDataLoader : public DatasetReader<TInput, TInput, TMemoryResource>
    {
    public:
        using BaseLoader = DatasetReader<TInput, TInput, TMemoryResource>;
        using HostType = typename TensorHostTypeMap<TInput>::host_type;
        using TensorType = Tensor<TInput, TMemoryResource>;

        /**
         * @brief Constructs character-level data loader from preprocessed files.
         *
         * Loads tokenized data from preprocessed binary files. Only reads
         * vocabulary size from .vocab file header (doesn't load full vocabulary).
         *
         * @param text_file_base Base path to preprocessed files (without extensions)
         *                       e.g., "data/shakespeare" for shakespeare.vocab and shakespeare.tokens
         * @param batch_size Number of sequences per batch
         * @param seq_length Length of each sequence (context window)
         * @param is_training Whether to shuffle sequences (training mode)
         * @param device Compute device for tensor allocation
         * @param stride Step size for sliding window (default: seq_length for non-overlapping)
         *
         * @throws std::invalid_argument If device is null or seq_length <= 0
         * @throws std::runtime_error If device type doesn't match memory resource requirements
         * @throws std::runtime_error If vocabulary or tokens files are missing or invalid
         */
        CharDataLoader(
            const std::string& text_file_base,
            int64_t batch_size,
            int64_t seq_length,
            bool is_training,
            std::shared_ptr<Compute::ComputeDevice> device,
            int64_t stride = -1 )
            : BaseLoader( batch_size ),
            seq_length_( seq_length ),
            is_training_( is_training ),
            device_( validateDevice( device ) ),
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

            // Load tokenized data
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
            std::cout << "  Device: " << device_->getDeviceName() << std::endl;
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
                    input_tensor_->data()[i * seq_length_ + t] =
                        static_cast<HostType>( tokens_[seq_start_idx + t] );
                }

                // Copy target sequence (tokens[start+1 : start+seq_length+1])
                for (int64_t t = 0; t < seq_length_; ++t)
                {
                    target_tensor_->data()[i * seq_length_ + t] =
                        static_cast<HostType>( tokens_[seq_start_idx + t + 1] );
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
         *
         * Read from vocabulary file header during construction.
         * Does not require loading the full vocabulary.
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
        std::shared_ptr<Compute::ComputeDevice> device_;
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
        static std::shared_ptr<Compute::ComputeDevice> validateDevice(
            std::shared_ptr<Compute::ComputeDevice> device )
        {
            if (!device)
            {
                throw std::invalid_argument( "Device cannot be null" );
            }

            if constexpr (std::is_same_v<TMemoryResource, CpuMemoryResource>)
            {
                if (device->getDeviceType() != DeviceType::Cpu)
                {
                    throw std::runtime_error(
                        "CpuMemoryResource requires CPU device, got " +
                        std::string( deviceTypeToString( device->getDeviceType() ) ) );
                }
            }

            if constexpr (std::is_same_v<TMemoryResource, CudaPinnedMemoryResource>)
            {
                if (device->getDeviceType() != DeviceType::Cuda)
                {
                    throw std::runtime_error(
                        "CudaPinnedMemoryResource requires CUDA device, got " +
                        std::string( deviceTypeToString( device->getDeviceType() ) ) );
                }
            }

            return device;
        }

        /**
         * @brief Loads vocabulary size from .vocab file header.
         *
         * Only reads the header to get vocabulary size - does NOT load
         * the full vocabulary mappings. This is all the data loader needs.
         *
         * Vocab file format:
         *   Bytes 0-7: vocab_size (size_t)
         *   Bytes 8+: vocabulary data (not loaded by data loader)
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

            // Don't read the rest - we only need the size
        }

        /**
         * @brief Loads tokenized data from preprocessed binary file.
         *
         * Format:
         *   Bytes 0-7: num_tokens (size_t)
         *   Bytes 8+: token_indices (int32_t array)
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
        }

        /**
         * @brief Creates sequence start indices for sliding window approach.
         *
         * Stores only start indices (memory efficient) rather than
         * materializing all sequences.
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