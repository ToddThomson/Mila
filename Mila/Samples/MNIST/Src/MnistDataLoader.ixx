module;
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <filesystem>
#include <fstream>

export module Mnist.DataLoader;

import Mila;

namespace Mila::Mnist
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Data;

    namespace fs = std::filesystem;

    export constexpr int MNIST_IMAGE_SIZE = 28 * 28;     // 28x28 pixels
    export constexpr int MNIST_NUM_CLASSES = 10;         // 10 digits (0-9)
    export constexpr int MNIST_TRAIN_SIZE = 60000;       // Training set size
    export constexpr int MNIST_TEST_SIZE = 10000;        // Test set size

    export template<typename TInput, typename TMemoryResource>
        requires ValidFloatTensorType<TInput> && 
            ( std::is_same_v<TMemoryResource, CudaPinnedMemoryResource> || std::is_same_v<TMemoryResource, CpuMemoryResource> )
    class MnistDataLoader : public DataLoader<TInput, TInput, TMemoryResource> {
    public:
        using BaseLoader = DataLoader<TInput, TInput, TMemoryResource>;

        MnistDataLoader( const std::string& data_dir, size_t batch_size, bool is_training )
            : BaseLoader( batch_size ), is_training_( is_training ) {

            std::string images_file = is_training_ ?
                data_dir + "/train-images.idx3-ubyte" :
                data_dir + "/t10k-images.idx3-ubyte";

            std::string labels_file = is_training_ ?
                data_dir + "/train-labels.idx1-ubyte" :
                data_dir + "/t10k-labels.idx1-ubyte";

            // Load MNIST dataset
            loadImages( images_file );
            loadLabels( labels_file );

            // Initialize random indices for shuffling
            for ( size_t i = 0; i < num_samples_; ++i ) {
                indices_.push_back( i );
            }

            // Shuffle indices if in training mode
            if ( is_training_ ) {
                shuffleIndices();
            }

                
            num_batches_ = num_samples_ / this->batch_size_;  // Integer division to skip partial batch

            input_tensor_ = Tensor<TInput, TMemoryResource>( { this->batch_size_, static_cast<size_t>(MNIST_IMAGE_SIZE) } );
            target_tensor_ = Tensor<TInput, TMemoryResource>( { this->batch_size_, static_cast<size_t>(MNIST_NUM_CLASSES) } );

            std::cout << "MNIST DataLoader initialized with "
                << num_samples_ << " samples and " << num_batches_ << " batches." << std::endl;
        }

        size_t numBatches() const override {
            return num_batches_;
        }

        void reset() override {
            BaseLoader::reset();  // Reset current_batch_ to 0

            if ( is_training_ ) {
                shuffleIndices();
            }
        }

        void nextBatch() override {
            if ( this->current_batch_ >= num_batches_ ) {
                return;
            }

            // Calculate batch start
            size_t batch_start = this->current_batch_ * this->batch_size_;

            // Fill the batch with data
            for ( size_t i = 0; i < this->batch_size_; ++i ) {
                size_t idx = indices_[ batch_start + i ];

                // Copy image data and normalize to [0, 1]
                for ( int j = 0; j < MNIST_IMAGE_SIZE; ++j ) {
                    input_tensor_.data()[ i * MNIST_IMAGE_SIZE + j ] =
                        static_cast<TInput>( images_[ idx * MNIST_IMAGE_SIZE + j ] ) / 255.0f;
                }

                // One-hot encode the labels
                for ( int j = 0; j < MNIST_NUM_CLASSES; ++j ) {
                    target_tensor_.data()[ i * MNIST_NUM_CLASSES + j ] =
                        static_cast<TInput>( j == labels_[ idx ] ? 1.0f : 0.0f );
                }
            }

            this->current_batch_++;
        }

        /**
        * @brief Get the input tensor.
        * @return Reference to the input tensor.
        */
        Tensor<TInput, TMemoryResource>& inputs() override {
            return input_tensor_;
        }

        /**
        * @brief Get the input tensor (const version).
        * @return Const reference to the input tensor.
        */
        const Tensor<TInput, TMemoryResource>& inputs() const override {
            return input_tensor_;
        }

        /**
        * @brief Get the target tensor.
        * @return Reference to the target tensor.
        */
        Tensor<TInput, TMemoryResource>& targets() override {
            return target_tensor_;
        }

        /**
        * @brief Get the target tensor (const version).
        * @return Const reference to the target tensor.
        */
        const Tensor<TInput, TMemoryResource>& targets() const override {
            return target_tensor_;
        }

    private:
        bool is_training_;
        size_t num_samples_;
        size_t num_batches_;

        std::vector<unsigned char> images_;
        std::vector<unsigned char> labels_;
        std::vector<size_t> indices_;

        // Pre-allocated tensors for batch processing
        Tensor<TInput, TMemoryResource> input_tensor_;
        Tensor<TInput, TMemoryResource> target_tensor_;

        // Read 4 bytes as big-endian integer
        uint32_t readInt32( std::ifstream& stream ) {
            unsigned char buffer[ 4 ];
            stream.read( reinterpret_cast<char*>(buffer), 4 );

            // Convert from big-endian to native
            return (static_cast<uint32_t>(buffer[ 0 ]) << 24) |
                (static_cast<uint32_t>(buffer[ 1 ]) << 16) |
                (static_cast<uint32_t>(buffer[ 2 ]) << 8) |
                static_cast<uint32_t>(buffer[ 3 ]);
        }

        // Load MNIST images
        void loadImages( const std::string& filename ) {
            std::ifstream file( filename, std::ios::binary );
            if ( !file ) {
                throw std::runtime_error( "Cannot open file: " + filename );
            }

            // Read header
            uint32_t magic = readInt32( file );
            if ( magic != 0x803 ) {
                throw std::runtime_error( "Invalid MNIST image file format" );
            }

            // Read number of images, rows, and columns
            uint32_t num_images = readInt32( file );
            uint32_t rows = readInt32( file );
            uint32_t cols = readInt32( file );

            // Verify dimensions
            if ( rows != 28 || cols != 28 ) {
                throw std::runtime_error( "Expected 28x28 images, got " +
                    std::to_string( rows ) + "x" + std::to_string( cols ) );
            }

            // Allocate buffer and read all images at once
            num_samples_ = num_images;
            images_.resize( num_images * rows * cols );
            file.read( reinterpret_cast<char*>(images_.data()), images_.size() );

            // Check if read was successful
            if ( !file ) {
                throw std::runtime_error( "Error reading MNIST image file" );
            }
        }

        // Load MNIST labels
        void loadLabels( const std::string& filename ) {
            std::ifstream file( filename, std::ios::binary );
            if ( !file ) {
                throw std::runtime_error( "Cannot open file: " + filename );
            }

            // Read header
            uint32_t magic = readInt32( file );
            if ( magic != 0x801 ) {
                throw std::runtime_error( "Invalid MNIST label file format" );
            }

            // Read number of labels
            uint32_t num_labels = readInt32( file );

            // Verify number matches number of images
            if ( num_labels != num_samples_ ) {
                throw std::runtime_error( "Number of labels doesn't match number of images" );
            }

            // Allocate buffer and read all labels at once
            labels_.resize( num_labels );
            file.read( reinterpret_cast<char*>(labels_.data()), labels_.size() );

            // Check if read was successful
            if ( !file ) {
                throw std::runtime_error( "Error reading MNIST label file" );
            }
        }

        // Shuffle indices for random batch sampling
        void shuffleIndices() {
            std::random_device rd;
            std::mt19937 gen( rd() );
            std::shuffle( indices_.begin(), indices_.end(), gen );
        }
    };
}
