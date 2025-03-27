module;
#include <cuda_runtime.h>
#include <vector>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <queue>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <optional>
#include <functional>
#include <memory>
#include <string>
#include <limits>
#include <stdexcept>
#include <exception>
#include <span>

export module Dnn.Gpt2.DatasetReader;

import Dnn.Tensor;

import Utils.Logger;
import Utils.DefaultLogger;

namespace Mila::Dnn::Gpt2
{
    /**
     * @class DatasetReader
     * @brief High-performance data loading class for GPT-2 tokenized datasets with multi-threaded processing.
     *
     * The DatasetReader class provides an efficient way to read, preprocess, and batch tokenized
     * datasets for language model training. It implements a multi-threaded pipeline with background
     * I/O operations and preprocessing to minimize training latency:
     *  - Streams data from files using a sliding window approach
     *  - Supports pinned memory allocation for efficient GPU transfers
     *  - Implements a producer-consumer pattern with configurable queue sizes
     *  - Can be paused and resumed for controlled resource utilization
     *  - Automatically handles dataset wrap-around for continuous training
     */
    export class DatasetReader {
    public:
        /**
         * @struct Config
         * @brief Configuration parameters for the DatasetReader.
         */
        struct Config {
            size_t max_queue_size = 10;        /**< Maximum number of batches to queue in memory */
            size_t read_chunk_size = 1024 * 1024; /**< Size of chunks when reading large files (1MB default) */
            bool verbose_logging = false;      /**< Whether to output detailed performance logs */
            std::shared_ptr<Mila::Utils::Logger> logger = std::make_shared<Mila::Utils::DefaultLogger>(); /**< Logger implementation */
            size_t token_window_size = 0;      /**< Size of token window in number of tokens (0 for auto-size) */
        };

        /**
        * @brief Constructs a new DatasetReader object.
        *
        * @param file_path Path to the tokenized dataset file containing integers.
        * @param batch_size Number of sequences in each batch.
        * @param seq_len Length of each sequence in tokens.
        * @param config Additional configuration parameters.
        *
        * @throws std::invalid_argument If batch_size or seq_len is zero.
        * @throws std::runtime_error If file operations fail or memory allocation fails.
        *
        * @details The constructor performs several initialization steps:
        *   1. Opens the dataset file and initializes window-based streaming access
        *   2. Allocates memory for input/target tensors (using pinned memory if CUDA is available)
        *   3. Starts background threads for parallel I/O and preprocessing
        */
        DatasetReader( const std::string& file_path, size_t batch_size, size_t seq_len,
            const Config& config = Config() )
            : batch_size_( batch_size ), seq_len_( seq_len ), stop_( false ), config_( config ),
            pinned_inputs_( nullptr ), pinned_targets_( nullptr ), file_path_( file_path ) {

            if ( batch_size_ == 0 || seq_len_ == 0 ) {
                throw std::invalid_argument( "Batch size and sequence length must be positive" );
            }

            // Get file size and validate
            std::error_code ec;
            file_size_ = std::filesystem::file_size( file_path, ec );
            if ( ec ) throw std::runtime_error( "Failed to get file size: " + ec.message() );

            if ( file_size_ % sizeof( int ) != 0 ) {
                throw std::runtime_error( "File size is not a multiple of int size" );
            }

            // Open the file
            file_.open( file_path, std::ios::binary | std::ios::in );
            if ( !file_.is_open() ) throw std::runtime_error( "Failed to open file: " + file_path );

            // Initialize dataset access with streaming approach
            initialize_dataset();

            // Allocate memory for input-target tensors
            try {
                allocate_memory( pinned_inputs_ );
                allocate_memory( pinned_targets_ );
            }
            catch ( const std::exception& e ) {
                if ( pinned_inputs_ ) {
                    free_memory( pinned_inputs_ );
                }
                throw std::runtime_error( std::string( "Memory allocation failed: " ) + e.what() );
            }

            // Start background threads
            try {
                io_thread_ = std::thread( [this] { read_from_disk(); } );
                processing_thread_ = std::thread( [this] { preprocess_batches(); } );
            }
            catch ( const std::exception& e ) {
                stop_ = true;
                if ( io_thread_.joinable() ) io_thread_.join();
                if ( processing_thread_.joinable() ) processing_thread_.join();
                free_memory( pinned_inputs_ );
                free_memory( pinned_targets_ );
                throw std::runtime_error( std::string( "Failed to create threads: " ) + e.what() );
            }

            log( "DatasetReader initialized successfully", 0 );
        }

        /**
         * @brief Destroys the DatasetReader object.
         *
         * @details Safely stops all background threads and releases allocated memory.
         */
        ~DatasetReader() {
            stop_ = true;
            cv_io_.notify_all();
            cv_processing_.notify_all();

            if ( io_thread_.joinable() ) io_thread_.join();
            if ( processing_thread_.joinable() ) processing_thread_.join();

            // Clean up memory
            if ( pinned_inputs_ ) free_memory( pinned_inputs_ );
            if ( pinned_targets_ ) free_memory( pinned_targets_ );

            // Clean up any remaining raw buffers in the queue
            while ( !raw_data_queue_.empty() ) {
                int* buffer = raw_data_queue_.front();
                raw_data_queue_.pop();
                delete[] buffer;
            }

            file_.close();
        }

        /**
         * @brief Fetches the next batch of training data with inputs and targets.
         *
         * @tparam TensorType The tensor type to return (defaults to DeviceTensor<int>).
         * @return std::pair<TensorType, TensorType> A pair containing (input tensor, target tensor).
         *
         * @throws std::runtime_error If timeout occurs waiting for batch or if the reader was stopped.
         *
         * @details This method blocks until a preprocessed batch is available (up to 5 seconds timeout).
         * The returned tensors contain shifted versions of the same data - targets are inputs shifted
         * by one position (implementing language modeling next-token prediction).
         */
        template <typename TensorType = DeviceTensor<int>>
            requires (std::same_as<TensorType, HostTensor<int>> ||
        std::same_as<TensorType, PinnedTensor<int>> ||
            std::same_as<TensorType, DeviceTensor<int>>)
            std::pair<TensorType, TensorType> next_batch() {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Get data from queue
            std::unique_lock<std::mutex> lock( mutex_ );

            // Always notify the I/O thread before waiting for a batch
            // This ensures that it has a chance to refill the queue even if batch_queue_ is empty
            cv_io_.notify_one();

            bool result = cv_processing_.wait_for( lock,
                std::chrono::seconds( 20 ),
                [this] { return !batch_queue_.empty() || stop_; } );

            if ( !result ) {
                log( "Timeout waiting for the next batch", 1 );
                throw std::runtime_error( "Timeout waiting for the next batch" );
            }

            if ( stop_ || batch_queue_.empty() ) {
                throw std::runtime_error( "Dataset reader stopped or queue empty" );
            }

            auto batch = std::move( batch_queue_.front() );
            batch_queue_.pop();

            // Notify again after popping to ensure continuous production
            cv_io_.notify_one();

            lock.unlock();

            int* raw_inputs = batch.first;
            int* raw_targets = batch.second;

            // Create the tensors with the appropriate shape
            std::vector<size_t> input_shape = { batch_size_, seq_len_ };
            std::vector<size_t> target_shape = { batch_size_, seq_len_ };
            TensorType inputs, targets;

            // Create tensors based on the tensor type
            if constexpr ( std::is_same_v<TensorType, DeviceTensor<int>> ) {
                // For device tensors, we need to create new tensors and copy data to the device
                inputs = TensorType( input_shape );
                targets = TensorType( target_shape );

                // Copy data from pinned memory to device memory
                cudaError_t status_inputs = cudaMemcpy(
                    inputs.data(), raw_inputs,
                    batch_size_ * seq_len_ * sizeof( int ),
                    cudaMemcpyHostToDevice
                );

                cudaError_t status_targets = cudaMemcpy(
                    targets.data(), raw_targets,
                    batch_size_ * seq_len_ * sizeof( int ),
                    cudaMemcpyHostToDevice
                );

                if ( status_inputs != cudaSuccess || status_targets != cudaSuccess ) {
                    throw std::runtime_error( "Failed to copy data to device: " +
                        std::string( cudaGetErrorString( status_inputs != cudaSuccess ? status_inputs : status_targets ) ) );
                }
            }
            else if constexpr ( std::is_same_v<TensorType, PinnedTensor<int>> ) {
                // For pinned tensors, create shared memory views of the existing pinned memory
                // Create shared pointers that won't delete the memory since we manage it separately
                auto inputs_ptr = std::shared_ptr<int>( raw_inputs, []( int* ) {} );
                auto targets_ptr = std::shared_ptr<int>( raw_targets, []( int* ) {} );

                // Create tensors that wrap the existing pinned memory
                inputs = TensorType( input_shape, inputs_ptr );
                targets = TensorType( target_shape, targets_ptr );
            }
            else if constexpr ( std::is_same_v<TensorType, HostTensor<int>> ) {
                // For host tensors, create new tensors and copy data from pinned memory
                inputs = TensorType( input_shape );
                targets = TensorType( target_shape );

                // Manually copy data
                std::memcpy( inputs.data(), raw_inputs, batch_size_ * seq_len_ * sizeof( int ) );
                std::memcpy( targets.data(), raw_targets, batch_size_ * seq_len_ * sizeof( int ) );
            }

            // Set names for the tensors
            inputs.setName( "batch_inputs" );
            targets.setName( "batch_targets" );

            // Timing and logging
            auto end_time = std::chrono::high_resolution_clock::now();
            auto batch_load_time = std::chrono::duration<double, std::milli>( end_time - start_time ).count();

            // Log every 10 batches if verbose
            static int batch_count = 0;
            if ( config_.verbose_logging && ++batch_count % 10 == 0 ) {
                log( "Tensor batch " + std::to_string( batch_count ) + " loaded in " +
                    std::to_string( batch_load_time ) + " ms", 0 );
            }

            return { std::move( inputs ), std::move( targets ) };
        }

        /**
         * @brief Pauses the background loading threads.
         *
         * @details This can be used to temporarily reduce system load without destroying the reader.
         * Background threads will wait until resume() is called.
         */
        void pause() {
            std::lock_guard<std::mutex> lock( control_mutex_ );
            paused_ = true;
            log( "Pausing data loading", 0 );
        }

        /**
         * @brief Resumes the background loading threads after being paused.
         *
         * @details Wakes up the background threads and continues data loading operations.
         */
        void resume() {
            std::lock_guard<std::mutex> lock( control_mutex_ );
            paused_ = false;
            log( "Resuming data loading", 0 );
            cv_io_.notify_one();
            cv_processing_.notify_one();
        }

    private:
        size_t batch_size_;              /**< Number of sequences in each batch */
        size_t seq_len_;                 /**< Length of each sequence in tokens */
        size_t num_tokens_;              /**< Total number of tokens in the dataset */
        size_t file_size_;               /**< Size of the dataset file in bytes */
        std::string file_path_;          /**< Path to the dataset file */

        std::ifstream file_;             /**< File stream for the dataset */
        int* pinned_inputs_;             /**< Buffer for input tensors */
        int* pinned_targets_;            /**< Buffer for target tensors */

        std::unique_ptr<int[]> token_window_;  /**< Window buffer for tokens */
        size_t token_window_size_;       /**< Size of token window in tokens */
        size_t current_window_start_;    /**< First token index in current window */
        size_t current_window_end_;      /**< Last token index in current window */

        std::queue<int*> raw_data_queue_; /**< Queue for raw data batches */
        std::queue<std::pair<int*, int*>> batch_queue_; /**< Queue for preprocessed input/target pairs */

        std::thread io_thread_;          /**< Thread for disk I/O operations */
        std::thread processing_thread_;  /**< Thread for data preprocessing */

        std::mutex mutex_;               /**< Mutex for protecting shared queues */
        std::mutex control_mutex_;       /**< Mutex for control operations */

        std::condition_variable cv_io_;  /**< Condition variable for I/O thread synchronization */
        std::condition_variable cv_processing_; /**< Condition variable for processing thread synchronization */

        std::atomic<bool> stop_{ false }; /**< Flag to signal threads to stop */
        std::atomic<bool> paused_{ false }; /**< Flag to signal threads to pause */

        Config config_;                  /**< Configuration settings */

        /**
         * @brief Logs messages through the configured logger.
         *
         * @param message The message text to log.
         * @param level The log level (higher values mean less important).
         */
        void log( const std::string& message, int level ) {
            if ( config_.logger ) {
                config_.logger->log( message, level );
            }
        }

        /**
         * @brief Initializes the dataset access without loading the entire file into memory.
         *
         * @throws std::runtime_error If reading fails or initialization fails.
         *
         * @details Uses a streaming approach to access the dataset file with a fixed-size
         * token window buffer, using std::span for safe, non-owning memory views.
         */
        void initialize_dataset() {
            // Calculate the number of tokens in the dataset
            num_tokens_ = file_size_ / sizeof( int );

            // Validate that we have enough data
            if ( num_tokens_ < batch_size_ * seq_len_ ) {
                throw std::runtime_error( "Not enough tokens in file for requested batch size and sequence length" );
            }

            // Determine an optimal window size based on available memory and access patterns
            if ( config_.token_window_size > 0 ) {
                // Use user-specified window size
                token_window_size_ = config_.token_window_size;
            }
            else {
                // Default: 25 million tokens (approx. 100MB) or 10% of dataset, whichever is smaller
                const size_t max_window_tokens = 25'000'000;
                const size_t min_window_size = batch_size_ * seq_len_ * 10; // At least 10 batches worth
                token_window_size_ = std::min( max_window_tokens, std::max( min_window_size, num_tokens_ / 10 ) );
            }

            // Initialize the token window buffer
            token_window_ = std::make_unique<int[]>( token_window_size_ );
            current_window_start_ = 0;

            // Load the initial window
            load_window( 0 );

            log( "Dataset initialized with streaming access. Total tokens: " +
                std::to_string( num_tokens_ ) + ", Window size: " +
                std::to_string( token_window_size_ ) + " tokens", 0 );
        }

        /**
         * @brief Loads a specific window of tokens from the dataset file.
         *
         * @param start_token Index of the first token in the window to load.
         * @throws std::runtime_error If reading fails.
         */
        void load_window( size_t start_token ) {
            // Ensure we don't try to read past the end of the file
            if ( start_token >= num_tokens_ ) {
                start_token = 0; // Reset to the beginning
            }

            // Calculate how many tokens we can read before the end of the file
            size_t tokens_to_read = std::min( token_window_size_, num_tokens_ - start_token );

            // Seek to the right position in the file
            file_.clear(); // Clear any error flags
            file_.seekg( start_token * sizeof( int ) );

            if ( !file_.good() ) {
                throw std::runtime_error( "Failed to seek to position " +
                    std::to_string( start_token ) + " in dataset file" );
            }

            // Read the tokens from the file
            file_.read( reinterpret_cast<char*>(token_window_.get()),
                tokens_to_read * sizeof( int ) );

            if ( file_.gcount() != static_cast<std::streamsize>(tokens_to_read * sizeof( int )) ) {
                throw std::runtime_error( "Failed to read complete window from dataset file" );
            }

            // If we need to wrap around the file to fill the window
            if ( tokens_to_read < token_window_size_ ) {
                file_.clear(); // Clear EOF flag
                file_.seekg( 0 ); // Back to beginning

                size_t remaining_tokens = token_window_size_ - tokens_to_read;
                file_.read( reinterpret_cast<char*>( token_window_.get() + tokens_to_read ),
                    remaining_tokens * sizeof( int ) );

                if ( file_.gcount() != static_cast<std::streamsize>( remaining_tokens * sizeof( int ) ) ) {
                    throw std::runtime_error( "Failed to read remaining tokens after wrap-around" );
                }
            }

            current_window_start_ = start_token;
            current_window_end_ = start_token + token_window_size_;

            if ( current_window_end_ > num_tokens_ ) {
                current_window_end_ = (current_window_end_ % num_tokens_);
            }

            if ( config_.verbose_logging ) {
                log( "Loaded token window from index " + std::to_string( current_window_start_ ) +
                    " with " + std::to_string( token_window_size_ ) + " tokens", 1 );
            }
        }

        /**
         * @brief Gets a non-owning view of tokens at the specified position.
         *
         * @param position Starting token position
         * @param count Number of tokens to access
         * @return std::span<const int> A view of the requested tokens
         * @throws std::runtime_error If the position is invalid or reading fails.
         */
        std::span<const int> get_tokens( size_t position, size_t count ) {
            // Handle position wrapping
            if ( position >= num_tokens_ ) {
                position = position % num_tokens_;
            }

            // Check if the requested tokens are within the current window
            bool needs_reload = false;

            // Case 1: Position is before the current window
            if ( position < current_window_start_ ) {
                needs_reload = true;
            }

            // Case 2: Position+count extends beyond the current window
            if ( position + count > current_window_start_ + token_window_size_ ) {
                needs_reload = true;
            }

            // Case 3: Window crosses the file boundary but position doesn't need to
            if ( current_window_end_ < current_window_start_ &&
                (position >= current_window_end_ && position < current_window_start_) ) {
                needs_reload = true;
            }

            // Load a new window if needed
            if ( needs_reload ) {
                load_window( position );
            }

            // Calculate the local index within our window
            size_t local_index = 0;
            if ( position >= current_window_start_ ) {
                local_index = position - current_window_start_;
            }
            else {
                // Handle the case where the position wraps around
                local_index = (num_tokens_ - current_window_start_) + position;
            }

            // Return a span (non-owning view) of the token data
            return std::span<const int>( token_window_.get() + local_index, count );
        }

        /**
         * @brief Background thread function that reads data from the dataset file.
         *
         * @details Continuously reads batches from the dataset file using the window buffer
         * and pushes them to the raw_data_queue_ for preprocessing.
         */
        void read_from_disk() {
            try {
                auto thread_start_time = std::chrono::high_resolution_clock::now();
                size_t position = 0;

                while ( !stop_ ) {
                    // Check if we should pause
                    {
                        std::unique_lock<std::mutex> lock( control_mutex_ );
                        if ( paused_ ) {
                            cv_io_.wait( lock, [this] { return !paused_ || stop_; } );
                            if ( stop_ ) break;
                        }
                    }

                    // Check queue size before adding more data
                    {
                        std::unique_lock<std::mutex> lock( mutex_ );
                        if ( raw_data_queue_.size() >= config_.max_queue_size ) {
                            cv_io_.wait( lock, [this] {
                                return raw_data_queue_.size() < config_.max_queue_size || stop_;
                            } );
                            if ( stop_ ) break;
                        }
                    }

                    auto read_start_time = std::chrono::high_resolution_clock::now();

                    // Allocate buffer and handle potential exceptions
                    int* raw_buffer = nullptr;
                    try {
                        raw_buffer = new int[ batch_size_ * seq_len_ ];
                    }
                    catch ( const std::bad_alloc& ) {
                        log( "Memory allocation failed for raw buffer", 2 );
                        std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
                        continue;
                    }

                    // Handle wrap-around for continuous training
                    if ( position + batch_size_ * seq_len_ > num_tokens_ ) {
                        position = 0;
                        log( "Reached end of dataset, wrapping around", 0 );
                    }

                    // Get tokens from our window buffer
                    auto tokens_view = get_tokens( position, batch_size_ * seq_len_ );

                    // Copy the tokens to our raw buffer
                    std::memcpy( raw_buffer, tokens_view.data(), batch_size_ * seq_len_ * sizeof( int ) );
                    position += batch_size_ * seq_len_;

                    auto read_end_time = std::chrono::high_resolution_clock::now();
                    auto read_time = std::chrono::duration<double, std::milli>(
                        read_end_time - read_start_time ).count();

                    {
                        std::lock_guard<std::mutex> lock( mutex_ );
                        raw_data_queue_.push( raw_buffer );
                    }
                    cv_processing_.notify_one();

                    if ( config_.verbose_logging ) {
                        auto thread_end_time = std::chrono::high_resolution_clock::now();
                        auto thread_elapsed_time = std::chrono::duration<double, std::milli>(
                            thread_end_time - thread_start_time ).count();

                        log( "Disk read time: " + std::to_string( read_time ) + " ms | Thread elapsed: " +
                            std::to_string( thread_elapsed_time ) + " ms", 1 );
                    }
                }
            }
            catch ( const std::exception& e ) {
                log( "Exception in read_from_disk: " + std::string( e.what() ), 2 );
                stop_ = true;
                cv_processing_.notify_all();
            }
        }

        /**
         * @brief Background thread function that preprocesses raw data into input/target pairs.
         *
         * @details Takes raw data from raw_data_queue_, creates input/target pairs by
         * shifting sequences, and places results in batch_queue_ for consumption.
         */
        void preprocess_batches() {
            try {
                while ( !stop_ ) {
                    // Check for pause condition first
                    {
                        std::unique_lock<std::mutex> lock( control_mutex_ );
                        if ( paused_ ) {
                            cv_processing_.wait( lock, [this] { return !paused_ || stop_; } );
                            if ( stop_ ) break;
                        }
                    }

                    // CRITICAL FIX: Check batch_queue_ status and notify I/O thread if needed
                    bool need_more_batches = false;
                    {
                        std::lock_guard<std::mutex> lock( mutex_ );
                        // We need more batches if the queue is less than half full
                        need_more_batches = (batch_queue_.size() < config_.max_queue_size / 2);

                        // If queue is almost empty, this is urgent
                        if ( batch_queue_.size() < 2 ) {
                            cv_io_.notify_all(); // Urgent notification
                        }
                    }

                    // Keep the I/O pipeline active if we need more batches
                    if ( need_more_batches ) {
                        cv_io_.notify_one();
                    }

                    // Try to get raw data to process
                    int* raw_buffer = nullptr;
                    {
                        std::unique_lock<std::mutex> lock( mutex_ );

                        // Case 1: If batch queue is full, wait until there's space
                        if ( batch_queue_.size() >= config_.max_queue_size ) {
                            cv_processing_.wait_for( lock, std::chrono::milliseconds( 50 ),
                                [this] { return batch_queue_.size() < config_.max_queue_size || stop_; } );
                            if ( stop_ ) break;
                            continue; // Re-check conditions after waking up
                        }

                        // Case 2: If there's raw data available, process it
                        if ( !raw_data_queue_.empty() ) {
                            raw_buffer = raw_data_queue_.front();
                            raw_data_queue_.pop();
                        }
                        // Case 3: No raw data, so wait briefly for some to arrive
                        else {
                            // Signal the I/O thread that we need data
                            cv_io_.notify_one();

                            // Short wait with timeout to prevent deadlock
                            cv_processing_.wait_for( lock, std::chrono::milliseconds( 20 ),
                                [this] { return !raw_data_queue_.empty() || stop_; } );

                            if ( stop_ ) break;

                            // Check again after wait
                            if ( !raw_data_queue_.empty() ) {
                                raw_buffer = raw_data_queue_.front();
                                raw_data_queue_.pop();
                            }
                            else {
                                // No data arrived, restart loop
                                continue;
                            }
                        }
                    }

                    if ( !raw_buffer ) continue;

                    // Process the raw buffer into inputs/targets
                    for ( size_t i = 0; i < batch_size_; ++i ) {
                        std::memcpy( pinned_inputs_ + i * seq_len_,
                            raw_buffer + i * seq_len_,
                            seq_len_ * sizeof( int ) );

                        std::memcpy( pinned_targets_ + i * seq_len_,
                            raw_buffer + i * seq_len_ + 1,
                            (seq_len_ - 1) * sizeof( int ) );

                        pinned_targets_[ i * seq_len_ + (seq_len_ - 1) ] = -1;  // Pad last target
                    }

                    delete[] raw_buffer;

                    // Add the processed batch to the queue
                    {
                        std::lock_guard<std::mutex> lock( mutex_ );
                        batch_queue_.push( { pinned_inputs_, pinned_targets_ } );

                        // Always notify waiting consumers
                        cv_processing_.notify_all();

                        // If queue was nearly empty before this push, notify again to be safe
                        if ( batch_queue_.size() <= 2 ) {
                            cv_processing_.notify_all();
                        }
                    }

                    // Keep the I/O pipeline moving
                    cv_io_.notify_one();
                }
            }
            catch ( const std::exception& e ) {
                log( "Exception in preprocess_batches: " + std::string( e.what() ), 2 );
                stop_ = true;
                cv_io_.notify_all();
                cv_processing_.notify_all();
            }
        }
         /**
        * @brief Allocates memory for tensor buffers.
        *
        * @param buffer Reference to pointer that will hold the allocated memory.
        *
        * @throws std::runtime_error If CUDA memory allocation fails.
        * @throws std::bad_alloc If standard memory allocation fails.
        *
        * @details Attempts to allocate CUDA pinned memory for optimal performance.
        * Falls back to standard allocation if CUDA is not available.
        */
        void allocate_memory( int*& buffer ) {
            cudaError_t cuda_status = cudaMallocHost( &buffer, batch_size_ * seq_len_ * sizeof( int ) );
            if ( cuda_status != cudaSuccess ) {
                // CUDA pinned memory allocation failed, fall back to standard allocation
                log( "CUDA pinned memory allocation failed, falling back to standard allocation", 1 );
                buffer = new int[ batch_size_ * seq_len_ ];
                if ( !buffer ) {
                    throw std::bad_alloc();
                }
            }
        }

        /**
        * @brief Frees previously allocated memory.
        *
        * @param buffer Reference to pointer to free and nullify.
        */
        void free_memory( int*& buffer ) {
            if ( buffer ) {
                // Try to free as pinned memory first
                cudaError_t cuda_status = cudaFreeHost( buffer );
                if ( cuda_status != cudaSuccess ) {
                    // If not pinned memory, free as standard memory
                    delete[] buffer;
                }
                buffer = nullptr;
            }
        }
    };
}