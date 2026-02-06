module;
#include <vector>
#include <filesystem>
#include <fstream>
#include <queue>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <memory>
#include <string>
#include <stdexcept>
#include <span>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <random>
#include <algorithm>
#include <type_traits>
#include <exception>

export module Data.TokenSequenceLoader;
export import :Config;

import Data.DataLoader;
import Data.Tokenizer;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;
import Dnn.TensorHostTypeMap;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.CpuMemoryResource;
import Compute.CudaPinnedMemoryResource;

namespace Mila::Dnn::Data
{
    using namespace Mila::Dnn::Compute;

    // REVIEW: Semantically, Token Ids are unsigned. However, I want to use int* in my CUDA Encoder kenels
    // For now we 'll use int32_t as the TokenId type

    /**
     * @brief Token sequence loader for autoregressive language models.
     *
     * Loads tokenized text data for causal language modeling tasks such as GPT,
     * LLaMA, and other transformer-based models. Reads from pre-tokenized binary
     * .tokens files and produces batches of (input, target) sequence pairs where
     * target[i] = input[i+1] (next-token prediction).
     *
     * Implementation uses efficient disk streaming with double-buffered producer-consumer
     * pattern for high-throughput training on large corpora.
     *
     * @tparam TMemoryResource CpuMemoryResource or CudaPinnedMemoryResource
     */
    export template<typename TMemoryResource>
        requires (std::is_same_v<TMemoryResource, CudaPinnedMemoryResource> ||
            std::is_same_v<TMemoryResource, CpuMemoryResource>)
    class TokenSequenceLoader : public DataLoader<TensorDataType::INT32, TensorDataType::INT32, TMemoryResource>
    {
    public:
        using BaseLoader = DataLoader<TensorDataType::INT32, TensorDataType::INT32, TMemoryResource>;
        using HostType = typename TensorHostTypeMap<TensorDataType::INT32>::host_type;
        using TensorType = Tensor<TensorDataType::INT32, TMemoryResource>;

        /**
         * @brief Constructs streaming autoregressive sequence loader.
         *
         * @param tokens_file Path to binary .tokens file (uint32_t format)
         * @param batch_size Number of sequences per batch
         * @param seq_length Context window length (tokens per sequence)
         * @param is_training Enable shuffling and continuous epochs
         * @param device Compute device for tensor allocation
         * @param config Performance and streaming configuration
         *
         * @throws std::invalid_argument If batch_size or seq_length is zero
         * @throws std::runtime_error If file operations or initialization fails
         */
        TokenSequenceLoader(
            const std::filesystem::path& tokens_file,
            int64_t batch_size,
            int64_t seq_length,
            bool is_training,
            DeviceId device,
            const TokenSequenceLoaderConfig& config = TokenSequenceLoaderConfig() )
            : BaseLoader( batch_size ),
            seq_length_( seq_length ),
            is_training_( is_training ),
            device_( validateDeviceId( device ) ),
            config_( config ),
            stop_( false ),
            tokens_file_path_( tokens_file ),
            producer_exception_( nullptr ),
            front_buffer_ready_( false ),
            back_buffer_ready_( false ),
            current_window_idx_( 0 ),
            current_batch_in_window_( 0 )
        {
            if ( seq_length_ <= 0 )
            {
                throw std::invalid_argument( "Sequence length must be positive" );
            }

            if ( !std::filesystem::exists( tokens_file ) )
            {
                throw std::runtime_error( "Tokens file not found: " + tokens_file.string() );
            }

            file_size_ = std::filesystem::file_size( tokens_file );

            if ( file_size_ % sizeof( TokenId ) != 0 )
            {
                throw std::runtime_error( "File size is not a multiple of TokenId size" );
            }

            if ( file_size_ == 0 )
            {
                throw std::runtime_error( "Tokens file is empty" );
            }

            initializeDataset();
            allocateBuffers();

            try
            {
                // Start producer thread
                producer_thread_ = std::thread( [this] { producerThreadFunc(); } );

                // Wait for first batch with timeout
                std::unique_lock<std::mutex> lock( mutex_ );
                bool ready = cv_consumer_.wait_for(
                    lock,
                    std::chrono::milliseconds( config_.initialization_timeout_ms ),
                    [this] { return front_buffer_ready_.load() || producer_exception_ != nullptr; } );

                if ( producer_exception_ )
                {
                    std::rethrow_exception( producer_exception_ );
                }

                if ( !ready )
                {
                    stop_ = true;
                    cv_producer_.notify_all();
                    lock.unlock();

                    if ( producer_thread_.joinable() )
                        producer_thread_.join();

                    cleanupBuffers();
                    throw std::runtime_error( "Failed to initialize first batch within timeout" );
                }
            }
            catch ( ... )
            {
                stop_ = true;
                cv_producer_.notify_all();

                if ( producer_thread_.joinable() )
                    producer_thread_.join();

                cleanupBuffers();
                throw;
            }

            if ( config_.verbose_logging )
            {
                std::cout << "TokenSequenceLoader initialized:\n"
                    << "  File: " << tokens_file << "\n"
                    << "  Total tokens: " << num_tokens_ << "\n"
                    << "  Window size: " << window_size_tokens_ << " tokens\n"
                    << "  Sequences per window: " << sequences_per_window_ << "\n"
                    << "  Batches per window: " << batches_per_window_ << "\n"
                    << "  Total windows: " << num_windows_ << "\n"
                    << "  Total batches per epoch: " << num_batches_ << "\n"
                    << "  Batch size: " << batch_size << "\n"
                    << "  Sequence length: " << seq_length_ << "\n"
                    << "  Training mode: " << (is_training_ ? "yes" : "no") << "\n"
                    << "  Device: " << device_.toString() << std::endl;
            }
        }

        ~TokenSequenceLoader() noexcept
        {
            stop_ = true;
            cv_producer_.notify_all();

            if ( producer_thread_.joinable() )
            {
                producer_thread_.join();
            }

            cleanupBuffers();
        }

        TokenSequenceLoader( const TokenSequenceLoader& ) = delete;
        TokenSequenceLoader& operator=( const TokenSequenceLoader& ) = delete;
        TokenSequenceLoader( TokenSequenceLoader&& ) = delete;
        TokenSequenceLoader& operator=( TokenSequenceLoader&& ) = delete;

        int64_t numBatches() const override
        {
            return num_batches_;
        }

        void reset() override
        {
            BaseLoader::reset();

            std::unique_lock<std::mutex> lock( mutex_ );

            // Signal producer to reset
            current_window_idx_ = 0;
            current_batch_in_window_ = 0;
            front_buffer_ready_ = false;
            back_buffer_ready_ = false;

            if ( is_training_ )
            {
                shuffleSequenceIndices();
            }

            // Wake up producer to start fresh
            cv_producer_.notify_all();

            // Wait for new first batch
            bool ready = cv_consumer_.wait_for(
                lock,
                std::chrono::milliseconds( config_.initialization_timeout_ms ),
                [this] { return front_buffer_ready_.load() || producer_exception_ != nullptr; } );

            if ( producer_exception_ )
            {
                std::rethrow_exception( producer_exception_ );
            }

            if ( !ready )
            {
                throw std::runtime_error( "Failed to reset: timeout waiting for first batch" );
            }
        }

        void nextBatch() override
        {
            // Don't advance past the end in non-training mode
            if ( this->currentBatch() >= num_batches_ )
            {
                return;
            }

            std::unique_lock<std::mutex> lock( mutex_ );

            // Mark front buffer as consumed
            front_buffer_ready_ = false;

            // Swap buffers if back buffer is ready
            if ( back_buffer_ready_.load() )
            {
                swapBuffers();
                back_buffer_ready_ = false;
                front_buffer_ready_ = true;
            }

            // Wake producer to fill next buffer
            cv_producer_.notify_one();

            // Increment batch counter before waiting (so currentBatch() reflects next batch)
            this->incrementBatch();

            // If we've reached the end in non-training mode, don't wait
            if ( !is_training_ && this->currentBatch() >= num_batches_ )
            {
                return;
            }

            // Wait for front buffer to be ready
            bool ready = cv_consumer_.wait_for(
                lock,
                std::chrono::milliseconds( config_.batch_timeout_ms ),
                [this] { return front_buffer_ready_.load() || producer_exception_ != nullptr || stop_; } );

            if ( producer_exception_ )
            {
                std::rethrow_exception( producer_exception_ );
            }

            if ( !ready )
            {
                throw std::runtime_error( "Timeout waiting for next batch" );
            }

            if ( stop_ )
            {
                throw std::runtime_error( "Loader stopped while waiting for batch" );
            }
        }

        TensorType& inputs() override
        {
            return *front_input_tensor_;
        }

        const TensorType& inputs() const override
        {
            return *front_input_tensor_;
        }

        TensorType& targets() override
        {
            return *front_target_tensor_;
        }

        const TensorType& targets() const override
        {
            return *front_target_tensor_;
        }

        size_t numTokens() const
        {
            return num_tokens_;
        }

        int64_t sequenceLength() const
        {
            return seq_length_;
        }

        size_t windowSizeTokens() const
        {
            return window_size_tokens_;
        }

        size_t numWindows() const
        {
            return num_windows_;
        }

    private:
        // Configuration
        int64_t seq_length_;
        bool is_training_;
        DeviceId device_;
        TokenSequenceLoaderConfig config_;

        // File management
        std::filesystem::path tokens_file_path_;
        size_t file_size_;
        size_t num_tokens_;

        // Window management
        size_t window_size_tokens_;
        size_t sequences_per_window_;
        size_t batches_per_window_;
        size_t num_windows_;
        int64_t num_batches_;

        // Current position
        std::atomic<size_t> current_window_idx_;
        std::atomic<size_t> current_batch_in_window_;

        // Shuffling for training
        std::vector<size_t> sequence_indices_;

        // Double buffering
        std::shared_ptr<TensorType> front_input_tensor_;
        std::shared_ptr<TensorType> front_target_tensor_;
        std::shared_ptr<TensorType> back_input_tensor_;
        std::shared_ptr<TensorType> back_target_tensor_;

        // Thread synchronization
        std::atomic<bool> stop_;
        std::atomic<bool> front_buffer_ready_;
        std::atomic<bool> back_buffer_ready_;
        std::exception_ptr producer_exception_;
        std::thread producer_thread_;
        std::mutex mutex_;
        std::condition_variable cv_producer_;
        std::condition_variable cv_consumer_;

        static DeviceId validateDeviceId( DeviceId device )
        {
            if constexpr ( std::is_same_v<TMemoryResource, CpuMemoryResource> )
            {
                if ( device.type != DeviceType::Cpu )
                {
                    throw std::runtime_error(
                        "CpuMemoryResource requires CPU device, got " + device.toString() );
                }
            }

            if constexpr ( std::is_same_v<TMemoryResource, CudaPinnedMemoryResource> )
            {
                if ( device.type != DeviceType::Cuda )
                {
                    throw std::runtime_error(
                        "CudaPinnedMemoryResource requires CUDA device, got " + device.toString() );
                }
            }

            return device;
        }

        void initializeDataset()
        {
            num_tokens_ = file_size_ / sizeof( TokenId );

            const size_t tokens_needed = this->batchSize() * (seq_length_ + 1); // +1 for target shift
            if ( num_tokens_ < tokens_needed )
            {
                throw std::runtime_error(
                    "Not enough tokens for requested batch and sequence length. "
                    "Need at least " + std::to_string( tokens_needed ) +
                    " tokens, have " + std::to_string( num_tokens_ ) );
            }

            // Calculate window size
            if ( config_.token_window_size > 0 )
            {
                window_size_tokens_ = std::min( config_.token_window_size, num_tokens_ );
            }
            else
            {
                // Default: use reasonable window size based on memory constraints
                const size_t default_window = 25'000'000; // ~100MB for uint32
                const size_t min_window = tokens_needed * 10;
                window_size_tokens_ = std::min(
                    default_window,
                    std::max( min_window, num_tokens_ / 4 )
                );
                window_size_tokens_ = std::min( window_size_tokens_, num_tokens_ );
            }

            // Ensure window is large enough for at least one batch
            if ( window_size_tokens_ < tokens_needed )
            {
                window_size_tokens_ = tokens_needed;
            }

            // Calculate sequences per window (non-overlapping)
            // Each sequence needs seq_length + 1 tokens (for target shift)
            sequences_per_window_ = window_size_tokens_ / (seq_length_ + 1);

            if ( sequences_per_window_ < static_cast<size_t>( this->batchSize() ) )
            {
                throw std::runtime_error(
                    "Window too small to produce even one batch. "
                    "Increase window size or decrease batch size/sequence length." );
            }

            batches_per_window_ = sequences_per_window_ / this->batchSize();

            // Calculate number of windows needed to cover dataset
            const size_t total_sequences = num_tokens_ / (seq_length_ + 1);
            num_windows_ = (total_sequences + sequences_per_window_ - 1) / sequences_per_window_;

            if ( num_windows_ == 0 )
            {
                num_windows_ = 1;
            }

            num_batches_ = num_windows_ * batches_per_window_;

            if ( num_batches_ == 0 )
            {
                throw std::runtime_error( "Configuration produces zero batches" );
            }

            // Initialize sequence indices for shuffling
            if ( is_training_ )
            {
                prepareSequenceIndices();
            }
        }

        void prepareSequenceIndices()
        {
            sequence_indices_.clear();
            sequence_indices_.reserve( sequences_per_window_ );

            for ( size_t i = 0; i < sequences_per_window_; ++i )
            {
                sequence_indices_.push_back( i );
            }

            shuffleSequenceIndices();
        }

        void shuffleSequenceIndices()
        {
            std::random_device rd;
            std::mt19937 gen( rd() );
            std::shuffle( sequence_indices_.begin(), sequence_indices_.end(), gen );
        }

        void allocateBuffers()
        {
            const shape_t tensor_shape{ this->batchSize(), seq_length_ };

            // Front buffer (consumed by main thread)
            front_input_tensor_ = std::make_shared<TensorType>( device_, tensor_shape );
            front_target_tensor_ = std::make_shared<TensorType>( device_, tensor_shape );

            // Back buffer (filled by producer thread)
            back_input_tensor_ = std::make_shared<TensorType>( device_, tensor_shape );
            back_target_tensor_ = std::make_shared<TensorType>( device_, tensor_shape );
        }

        void cleanupBuffers() noexcept
        {
            front_input_tensor_.reset();
            front_target_tensor_.reset();
            back_input_tensor_.reset();
            back_target_tensor_.reset();
        }

        void swapBuffers() noexcept
        {
            std::swap( front_input_tensor_, back_input_tensor_ );
            std::swap( front_target_tensor_, back_target_tensor_ );
        }

        /**
         * @brief Producer thread: streams windows from disk and fills batches.
         *
         * Workflow:
         * 1. Load window from disk
         * 2. Fill back buffer with batch
         * 3. Mark back buffer ready
         * 4. Wait for consumer to swap buffers
         * 5. Repeat
         *
         * Exception safety: Catches all exceptions and stores them for consumer.
         */
        void producerThreadFunc() noexcept
        {
            try
            {
                std::ifstream file( tokens_file_path_, std::ios::binary );

                if ( !file.is_open() )
                {
                    throw std::runtime_error( "Failed to open tokens file in producer thread" );
                }

                auto window_buffer = std::make_unique<TokenId[]>( window_size_tokens_ );

                while ( !stop_ )
                {
                    const size_t window_idx = current_window_idx_.load();
                    const size_t batch_idx = current_batch_in_window_.load();

                    // Calculate global batch number
                    const size_t global_batch = window_idx * batches_per_window_ + batch_idx;

                    // Check if we've completed all batches in non-training mode
                    if ( !is_training_ && global_batch >= static_cast<size_t>(num_batches_) )
                    {
                        break; // Evaluation mode: stop at epoch end
                    }

                    // Load window if starting new window
                    if ( batch_idx == 0 )
                    {
                        loadWindowFromFile( file, window_buffer.get(), window_idx );

                        // Shuffle indices for new window in training mode
                        if ( is_training_ && window_idx > 0 )
                        {
                            shuffleSequenceIndices();
                        }
                    }

                    // Wait for back buffer to be free
                    {
                        std::unique_lock<std::mutex> lock( mutex_ );
                        cv_producer_.wait( lock, [this] {
                            return !back_buffer_ready_.load() || stop_;
                            } );

                        if ( stop_ )
                            break;
                    }

                    // Fill back buffer
                    fillBatch(
                        window_buffer.get(),
                        batch_idx,
                        back_input_tensor_->data(),
                        back_target_tensor_->data()
                    );

                    // Mark back buffer ready and notify consumer
                    {
                        std::lock_guard<std::mutex> lock( mutex_ );
                        back_buffer_ready_ = true;

                        // If front buffer not ready, swap immediately (for first batch)
                        if ( !front_buffer_ready_.load() )
                        {
                            swapBuffers();
                            back_buffer_ready_ = false;
                            front_buffer_ready_ = true;
                        }

                        cv_consumer_.notify_one();
                    }

                    // Advance to next batch
                    size_t next_batch = batch_idx + 1;
                    if ( next_batch >= batches_per_window_ )
                    {
                        next_batch = 0;
                        size_t next_window = window_idx + 1;

                        if ( is_training_ )
                        {
                            // Wrap around in training mode
                            next_window %= num_windows_;
                        }

                        current_window_idx_ = next_window;
                    }
                    current_batch_in_window_ = next_batch;
                }

                // Producer finished naturally (non-training mode reached end)
                // Make sure consumer doesn't hang waiting
                {
                    std::lock_guard<std::mutex> lock( mutex_ );
                    cv_consumer_.notify_all();
                }
            }
            catch ( ... )
            {
                std::lock_guard<std::mutex> lock( mutex_ );
                producer_exception_ = std::current_exception();
                cv_consumer_.notify_all();
            }
        }

        /**
         * @brief Loads a window from the token file.
         *
         * @param file Input file stream
         * @param buffer Destination buffer (must have space for window_size_tokens_)
         * @param window_idx Which window to load
         */
        void loadWindowFromFile( std::ifstream& file, TokenId* buffer, size_t window_idx )
        {
            const size_t start_token = (window_idx * sequences_per_window_ * (seq_length_ + 1)) % num_tokens_;
            size_t tokens_to_read = std::min( window_size_tokens_, num_tokens_ - start_token );

            // Seek and read first part
            file.clear();
            file.seekg( start_token * sizeof( TokenId ) );

            if ( !file.good() )
            {
                throw std::runtime_error( "Failed to seek to window position" );
            }

            file.read( reinterpret_cast<char*>(buffer), tokens_to_read * sizeof( TokenId ) );

            if ( file.gcount() != static_cast<std::streamsize>(tokens_to_read * sizeof( TokenId )) )
            {
                throw std::runtime_error( "Failed to read window from file" );
            }

            // Wrap around if needed
            if ( tokens_to_read < window_size_tokens_ )
            {
                const size_t remaining = window_size_tokens_ - tokens_to_read;

                file.clear();
                file.seekg( 0 );

                file.read(
                    reinterpret_cast<char*>( buffer + tokens_to_read ),
                    remaining * sizeof( TokenId )
                );

                if ( file.gcount() != static_cast<std::streamsize>(remaining * sizeof( TokenId )) )
                {
                    throw std::runtime_error( "Failed to read wrapped portion of window" );
                }
            }
        }

        /**
         * @brief Fills a batch from the current window buffer.
         *
         * Creates non-overlapping sequences where target[i] = input[i+1].
         *
         * @param window_buffer Source tokens for current window
         * @param batch_idx Batch index within current window
         * @param input_dest Destination for input sequences
         * @param target_dest Destination for target sequences
         */
        void fillBatch(
            const TokenId* window_buffer,
            size_t batch_idx,
            HostType* input_dest,
            HostType* target_dest )
        {
            const size_t batch_start_seq = batch_idx * this->batchSize();

            for ( size_t b = 0; b < static_cast<size_t>( this->batchSize() ); ++b )
            {
                const size_t seq_idx = is_training_
                    ? sequence_indices_[ batch_start_seq + b ]
                    : batch_start_seq + b;

                // Each sequence starts at seq_idx * (seq_length + 1) to avoid overlap
                // and to have room for target shift
                const size_t token_offset = seq_idx * (seq_length_ + 1);

                for ( size_t t = 0; t < static_cast<size_t>( seq_length_ ); ++t )
                {
                    const size_t input_idx = (token_offset + t) % window_size_tokens_;
                    const size_t target_idx = (token_offset + t + 1) % window_size_tokens_;

                    input_dest[ b * seq_length_ + t ] = static_cast<HostType>( window_buffer[ input_idx ] );
                    target_dest[ b * seq_length_ + t ] = static_cast<HostType>( window_buffer[ target_idx ] );
                }
            }
        }
    };
}