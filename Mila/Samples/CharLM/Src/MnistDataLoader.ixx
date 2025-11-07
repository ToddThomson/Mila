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

export module Mnist.DataLoader;

import Mila;

namespace Mila::Mnist
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Data;

    namespace fs = std::filesystem;

    export constexpr int64_t MNIST_IMAGE_SIZE = 28 * 28;     // 28x28 pixels
    export constexpr int64_t MNIST_NUM_CLASSES = 10;         // 10 digits (0-9)
    export constexpr int64_t MNIST_TRAIN_SIZE = 60000;       // Training set size
    export constexpr int64_t MNIST_TEST_SIZE = 10000;        // Test set size

    /**
     * @brief MNIST dataset loader with host-accessible memory support.
     *
     * Loads MNIST handwritten digit images and labels from IDX format files,
     * providing batched access for training and evaluation. Supports both
     * CPU memory and CUDA pinned memory for efficient GPU training pipelines.
     *
     * Key features:
     * - Automatic normalization of pixel values to [0, 1] range
     * - One-hot encoding of labels
     * - Optional shuffling for training mode
     * - Efficient batch processing with pre-allocated tensors
     * - Support for both CPU and pinned memory resources
     *
     * @tparam TInput Tensor data type for both inputs and targets
     * @tparam TMemoryResource Memory resource type (CpuMemoryResource or CudaPinnedMemoryResource)
     */
    export template<TensorDataType TInput, typename TMemoryResource>
        requires PrecisionSupportedOnDevice<TInput, DeviceType::Cpu> &&
            (std::is_same_v<TMemoryResource, CudaPinnedMemoryResource> || std::is_same_v<TMemoryResource, CpuMemoryResource>)
    class MnistDataLoader : public DataLoader<TInput, TInput, TMemoryResource>
    {
    public:
        using BaseLoader = DataLoader<TInput, TInput, TMemoryResource>;
        using HostType = typename TensorHostTypeMap<TInput>::host_type;
        using TensorType = Tensor<TInput, TMemoryResource>;

        /**
         * @brief Constructs MNIST data loader with specified configuration.
         *
         * @param data_directory Path to directory containing MNIST IDX files
         * @param batch_size Number of samples per batch
         * @param is_training Whether to load training set (true) or test set (false)
         * @param device Compute device for tensor allocation
         *
         * @throws std::invalid_argument If device is null
         * @throws std::runtime_error If device type doesn't match memory resource requirements
         * @throws std::runtime_error If MNIST files cannot be opened or have invalid format
         */
        MnistDataLoader(
            const std::string& data_directory,
            int64_t batch_size,
            bool is_training,
            std::shared_ptr<Compute::ComputeDevice> device )
            : BaseLoader( batch_size ),
            is_training_( is_training ),
            device_( validateDevice( device ) )
        {
            std::string images_file = is_training_ ?
                data_directory + "/train-images.idx3-ubyte" :
                data_directory + "/t10k-images.idx3-ubyte";

            std::string labels_file = is_training_ ?
                data_directory + "/train-labels.idx1-ubyte" :
                data_directory + "/t10k-labels.idx1-ubyte";

            loadImages( images_file );
            loadLabels( labels_file );

            for (size_t i = 0; i < num_samples_; ++i)
            {
                indices_.push_back( i );
            }

            if (is_training_)
            {
                shuffleIndices();
            }

            num_batches_ = num_samples_ / this->batchSize();

            input_tensor_ = std::make_shared<TensorType>(
                device_,
                shape_t{ this->batchSize(), MNIST_IMAGE_SIZE } );

            target_tensor_ = std::make_shared<TensorType>(
                device_,
                shape_t{ this->batchSize(), MNIST_NUM_CLASSES } );

            std::cout << "MNIST DataLoader initialized with "
                << num_samples_ << " samples and " << num_batches_ << " batches "
                << "on device: " << device_->getDeviceName() << std::endl;
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
                size_t idx = indices_[batch_start + i];

                for (int j = 0; j < MNIST_IMAGE_SIZE; ++j)
                {
                    input_tensor_->data()[i * MNIST_IMAGE_SIZE + j] =
                        static_cast<HostType>( images_[idx * MNIST_IMAGE_SIZE + j] ) / 255.0f;
                }

                for (int j = 0; j < MNIST_NUM_CLASSES; ++j)
                {
                    target_tensor_->data()[i * MNIST_NUM_CLASSES + j] =
                        static_cast<HostType>( j == labels_[idx] ? 1.0f : 0.0f );
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

    private:
        std::shared_ptr<Compute::ComputeDevice> device_;
        bool is_training_;
        size_t num_samples_;
        int64_t num_batches_;

        std::vector<unsigned char> images_;
        std::vector<unsigned char> labels_;
        std::vector<size_t> indices_;

        std::shared_ptr<TensorType> input_tensor_;
        std::shared_ptr<TensorType> target_tensor_;

        /**
         * @brief Validates device and ensures it matches memory resource requirements.
         *
         * @param device Device to validate
         * @return The validated device
         * @throws std::invalid_argument If device is null
         * @throws std::runtime_error If device type doesn't match memory resource
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

        uint32_t readInt32( std::ifstream& stream )
        {
            unsigned char buffer[4];
            stream.read( reinterpret_cast<char*>(buffer), 4 );

            return (static_cast<uint32_t>(buffer[0]) << 24) |
                (static_cast<uint32_t>(buffer[1]) << 16) |
                (static_cast<uint32_t>(buffer[2]) << 8) |
                static_cast<uint32_t>(buffer[3]);
        }

        void loadImages( const std::string& filename )
        {
            std::ifstream file( filename, std::ios::binary );
            if (!file)
            {
                throw std::runtime_error( "Cannot open file: " + filename );
            }

            uint32_t magic = readInt32( file );
            if (magic != 0x803)
            {
                throw std::runtime_error( "Invalid MNIST image file format" );
            }

            uint32_t num_images = readInt32( file );
            uint32_t rows = readInt32( file );
            uint32_t cols = readInt32( file );

            if (rows != 28 || cols != 28)
            {
                throw std::runtime_error( "Expected 28x28 images, got " +
                    std::to_string( rows ) + "x" + std::to_string( cols ) );
            }

            num_samples_ = num_images;
            images_.resize( num_images * rows * cols );
            file.read( reinterpret_cast<char*>(images_.data()), images_.size() );

            if (!file)
            {
                throw std::runtime_error( "Error reading MNIST image file" );
            }
        }

        void loadLabels( const std::string& filename )
        {
            std::ifstream file( filename, std::ios::binary );
            if (!file)
            {
                throw std::runtime_error( "Cannot open file: " + filename );
            }

            uint32_t magic = readInt32( file );
            if (magic != 0x801)
            {
                throw std::runtime_error( "Invalid MNIST label file format" );
            }

            uint32_t num_labels = readInt32( file );

            if (num_labels != num_samples_)
            {
                throw std::runtime_error( "Number of labels doesn't match number of images" );
            }

            labels_.resize( num_labels );
            file.read( reinterpret_cast<char*>(labels_.data()), labels_.size() );

            if (!file)
            {
                throw std::runtime_error( "Error reading MNIST label file" );
            }
        }

        void shuffleIndices()
        {
            std::random_device rd;
            std::mt19937 gen( rd() );
            std::shuffle( indices_.begin(), indices_.end(), gen );
        }
    };
}