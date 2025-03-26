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

export module Gpt2DataLoader;

import Dnn.Tensor;

namespace Mila::Dnn
{
	// Forward declaration for logging interface
	export class Logger {
	public:
		virtual void log( const std::string& message, int level = 0 ) = 0;
		virtual ~Logger() = default;
	};

	export class DefaultLogger : public Logger {
	public:
		void log( const std::string& message, int level = 0 ) override {
			if ( level >= log_level_ ) {
				std::cout << "[LOG] " << message << std::endl;
			}
		}

		void set_log_level( int level ) { log_level_ = level; }
	private:
		int log_level_ = 0;
	};

	export class DatasetReader {
	public:
		// Configuration struct
		struct Config {
			size_t max_queue_size = 10;
			size_t read_chunk_size = 1024 * 1024; // 1MB chunks for large files
			bool verbose_logging = false;
			std::shared_ptr<Logger> logger = std::make_shared<DefaultLogger>();
		};

		DatasetReader( const std::string& file_path, size_t batch_size, size_t seq_len,
			bool use_pinned_memory, const Config& config = Config() )
			: batch_size_( batch_size ), seq_len_( seq_len ), stop_( false ),
			use_pinned_memory_( use_pinned_memory ), config_( config ),
			pinned_inputs_( nullptr ), pinned_targets_( nullptr ), data_( nullptr ) {

			if ( batch_size_ == 0 || seq_len_ == 0 ) {
				throw std::invalid_argument( "Batch size and sequence length must be positive" );
			}

			// Open and memory-map the file using standard C++ features
			std::error_code ec;
			file_size_ = std::filesystem::file_size( file_path, ec );
			if ( ec ) throw std::runtime_error( "Failed to get file size: " + ec.message() );

			if ( file_size_ % sizeof( int ) != 0 ) {
				throw std::runtime_error( "File size is not a multiple of int size" );
			}

			file_.open( file_path, std::ios::binary | std::ios::in );
			if ( !file_.is_open() ) throw std::runtime_error( "Failed to open file: " + file_path );

			// Create custom buffer with good performance characteristics
			try {
				buffer_ = std::make_unique<char[]>( file_size_ );
			}
			catch ( const std::bad_alloc& ) {
				throw std::runtime_error( "Failed to allocate memory for file buffer" );
			}

			// Read the file in chunks to handle large files
			read_file_in_chunks();

			// Map the buffer to int* for processing
			data_ = reinterpret_cast<int*>(buffer_.get());
			num_tokens_ = file_size_ / sizeof( int );

			// Validate that we have enough data
			if ( num_tokens_ < batch_size_ * seq_len_ ) {
				throw std::runtime_error( "Not enough tokens in file for requested batch size and sequence length" );
			}

			// Allocate memory for input-target tensors
			try {
				allocate_memory( pinned_inputs_ );
				allocate_memory( pinned_targets_ );
			}
			catch ( const std::exception& e ) {
				// Clean up any already allocated memory
				if ( pinned_inputs_ ) free_memory( pinned_inputs_ );
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

		template <typename TensorType = PinnedTensor<int>>
		std::pair<TensorType, TensorType> next_batch() {
			auto start_time = std::chrono::high_resolution_clock::now();

			// Get data from queue
			std::unique_lock<std::mutex> lock( mutex_ );
			bool result = cv_processing_.wait_for( lock,
				std::chrono::seconds( 5 ),
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
			lock.unlock();
			cv_io_.notify_one();

			int* raw_inputs = batch.first;
			int* raw_targets = batch.second;

			// Create the tensors with the appropriate shape
			std::vector<size_t> input_shape = { batch_size_, seq_len_ };
			std::vector<size_t> target_shape = { batch_size_, seq_len_ };
			TensorType inputs, targets;

			// Create tensors that wrap the existing data
			// ... (rest of the tensor creation code) ...

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

		// Add ability to pause/resume loading
		void pause() {
			std::lock_guard<std::mutex> lock( control_mutex_ );
			paused_ = true;
			log( "Pausing data loading", 0 );
		}

		void resume() {
			std::lock_guard<std::mutex> lock( control_mutex_ );
			paused_ = false;
			log( "Resuming data loading", 0 );
			cv_io_.notify_one();
			cv_processing_.notify_one();
		}

	private:
		size_t batch_size_, seq_len_, num_tokens_, file_size_;
		std::ifstream file_;
		std::unique_ptr<char[]> buffer_; // In-memory buffer for the whole file
		int* data_; // Pointer to the data as integers
		int* pinned_inputs_;
		int* pinned_targets_;
		std::queue<int*> raw_data_queue_;
		std::queue<std::pair<int*, int*>> batch_queue_;
		std::thread io_thread_, processing_thread_;
		std::mutex mutex_;
		std::mutex control_mutex_; // Separate mutex for control operations
		std::condition_variable cv_io_, cv_processing_;
		std::atomic<bool> stop_{ false };
		std::atomic<bool> paused_{ false };
		bool use_pinned_memory_;
		Config config_;

		void log( const std::string& message, int level ) {
			if ( config_.logger ) {
				config_.logger->log( message, level );
			}
		}

		void read_file_in_chunks() {
			size_t bytes_read = 0;
			while ( bytes_read < file_size_ ) {
				size_t chunk_size = std::min( config_.read_chunk_size, file_size_ - bytes_read );
				file_.read( buffer_.get() + bytes_read, chunk_size );

				if ( !file_ ) {
					throw std::runtime_error( "Failed to read file chunk at position " +
						std::to_string( bytes_read ) );
				}

				bytes_read += chunk_size;
			}
		}

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

					// Copy the next batch of data
					std::memcpy( raw_buffer, data_ + position, batch_size_ * seq_len_ * sizeof( int ) );
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

		void preprocess_batches() {
			try {
				auto thread_start_time = std::chrono::high_resolution_clock::now();
				while ( !stop_ ) {
					// Check if we should pause
					{
						std::unique_lock<std::mutex> lock( control_mutex_ );
						if ( paused_ ) {
							cv_processing_.wait( lock, [this] { return !paused_ || stop_; } );
							if ( stop_ ) break;
						}
					}

					// Check batch queue size
					{
						std::unique_lock<std::mutex> lock( mutex_ );
						if ( batch_queue_.size() >= config_.max_queue_size ) {
							cv_processing_.wait( lock, [this] {
								return batch_queue_.size() < config_.max_queue_size || stop_;
							} );
							if ( stop_ ) break;
						}
					}

					int* raw_buffer = nullptr;

					{
						std::unique_lock<std::mutex> lock( mutex_ );
						cv_processing_.wait( lock, [this] {
							return !raw_data_queue_.empty() || stop_;
						} );

						if ( stop_ || raw_data_queue_.empty() ) break;

						raw_buffer = raw_data_queue_.front();
						raw_data_queue_.pop();
					}

					if ( !raw_buffer ) continue;

					auto preprocess_start_time = std::chrono::high_resolution_clock::now();

					// Process raw data into input/target pairs
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

					auto preprocess_end_time = std::chrono::high_resolution_clock::now();
					auto preprocess_time = std::chrono::duration<double, std::milli>(
						preprocess_end_time - preprocess_start_time ).count();

					{
						std::lock_guard<std::mutex> lock( mutex_ );
						batch_queue_.push( { pinned_inputs_, pinned_targets_ } );
					}
					cv_io_.notify_one();

					if ( config_.verbose_logging ) {
						auto thread_end_time = std::chrono::high_resolution_clock::now();
						auto thread_elapsed_time = std::chrono::duration<double, std::milli>(
							thread_end_time - thread_start_time ).count();

						log( "Preprocessing time: " + std::to_string( preprocess_time ) +
							" ms | Thread elapsed: " + std::to_string( thread_elapsed_time ) + " ms", 1 );
					}
				}
			}
			catch ( const std::exception& e ) {
				log( "Exception in preprocess_batches: " + std::string( e.what() ), 2 );
				stop_ = true;
				cv_io_.notify_all();
			}
		}

		void allocate_memory( int*& buffer ) {
			if ( use_pinned_memory_ ) {
				cudaError_t cuda_status = cudaMallocHost( &buffer, batch_size_ * seq_len_ * sizeof( int ) );
				if ( cuda_status != cudaSuccess ) {
					throw std::runtime_error( "CUDA pinned memory allocation failed: " +
						std::string( cudaGetErrorString( cuda_status ) ) );
				}
			}
			else {
				buffer = new int[ batch_size_ * seq_len_ ];
				if ( !buffer ) {
					throw std::bad_alloc();
				}
			}
		}

		void free_memory( int*& buffer ) {
			if ( buffer ) {
				if ( use_pinned_memory_ ) {
					cudaFreeHost( buffer );
				}
				else {
					delete[] buffer;
				}
				buffer = nullptr;
			}
		}
	};
}