module;
#include <string>
#include <vector>
#include <memory>
#include <sstream>

export module Mnist.Classifier;

import Mila;
import Mnist.DataLoader;

namespace Mila::Mnist
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief MNIST image classifier model.
     *
     * This neural network classifier is designed for the MNIST handwritten digit
     * recognition task. It consists of a multi-layer architecture with MLPs.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TDataType The data type used for tensor elements throughout the network.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TDataType = float>
        class MnistClassifier : public CompositeModule<TDeviceType, TDataType> {
        public:
            using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, HostMemoryResource>;
            using CompositeModuleBase = CompositeModule<TDeviceType, TDataType>;

            /**
             * @brief Constructs a new MnistClassifier with the specified device.
             *
             * @param name The name of the classifier for identification purposes.
             * @param device_name The name of the device to use for this module.
             * @param batch_size The batch size for training/inference.
             * @param precision The compute precision policy to use (defaults to Auto).
             */
            MnistClassifier( const std::string& name, const std::string& device_name, size_t batch_size,
                ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
                : CompositeModuleBase( device_name, precision ) {

                this->setName( name );
                input_shape_ = { batch_size, static_cast<size_t>(MNIST_IMAGE_SIZE) };

                initializeModules();
            }

            /**
             * @brief Constructs a new MnistClassifier with a specific device context.
             *
             * @param name The name of the classifier for identification purposes.
             * @param context The device context to use for this module.
             * @param batch_size The batch size for training/inference.
             * @param precision The compute precision policy to use (defaults to Auto).
             */
            MnistClassifier( const std::string& name, std::shared_ptr<DeviceContext> context, size_t batch_size,
                ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
                : CompositeModuleBase( context, precision ) {

                this->setName( name );
                input_shape_ = { batch_size, static_cast<size_t>(MNIST_IMAGE_SIZE) };

                initializeModules();
            }

            /**
             * @brief Performs the forward pass of the classifier.
             *
             * @param input The input tensor containing MNIST images.
             * @param output The output tensor to store class logits.
             */
            void forward( const Tensor<TDataType, MR>& input, Tensor<TDataType, MR>& output ) {
                mlp1_->forward( input, hidden1_ );

                // Second MLP block: hidden1 -> hidden2
                mlp2_->forward( hidden1_, hidden2_ );

                // Output layer: hidden2 -> output (logits)
                output_fc_layer_->forward( hidden2_, output );
            }

            //void backward();

            /**
             * @brief Gets the number of trainable parameters in this module.
             *
             * @return size_t The total number of parameters.
             */
            size_t parameterCount() const override {
                size_t total = 0;
                for ( const auto& module : this->getModules() ) {
                    total += module->parameterCount();
                }
                return total;
            }

            /**
             * @brief Saves the module state to a ZIP archive.
             *
             * @param zip The ZIP archive to save the module state to.
             */
            void save( mz_zip_archive& zip ) const override {
                for ( const auto& module : this->getModules() ) {
                    module->save( zip );
                }
            }

            /**
             * @brief Loads the module state from a ZIP archive.
             *
             * @param zip The ZIP archive to load the module state from.
             */
            void load( ModelArchive& archive ) override {
                for ( const auto& module : this->getModules() ) {
                    module->load( archive );
                }
            }

            /**
             * @brief Converts the module information to a human-readable string.
             *
             * @return std::string A string representation of the module information.
             */
            std::string toString() const override {
                std::ostringstream oss;
                oss << "====================" << std::endl;
                oss << "MNIST Classifier: " << this->getName() << std::endl;
                oss << "Architecture:" << std::endl;
                oss << "- Input shape: (" << input_shape_[ 0 ] << "," << input_shape_[ 1 ] << ")" << std::endl;
                oss << "- Hidden layer 1: 128 neurons" << std::endl;
                oss << "- Hidden layer 2: 64 neurons" << std::endl;
                oss << "- Output layer: 10 classes" << std::endl;
                oss << "- Total parameters: " << parameterCount() << std::endl;
                oss << "- Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
                oss << this->getComputePrecision().toString() << std::endl;
                oss << "====================" << std::endl;

                return oss.str();
            }

        private:
            std::vector<size_t> input_shape_;

            // Network layers
            std::shared_ptr<MLP<TDeviceType, TDataType>> mlp1_;
            std::shared_ptr<MLP<TDeviceType, TDataType>> mlp2_;
            std::shared_ptr<Linear<TDeviceType, TDataType>> output_fc_layer_;

            // Intermediate tensors for activations
            Tensor<TDataType, MR> hidden1_;
            Tensor<TDataType, MR> hidden2_;

            /**
             * @brief Initializes the sub-modules and intermediate tensors.
             */
            void initializeModules() {
                // Clear any existing modules
                for ( const auto& [name, _] : this->getNamedModules() ) {
                    this->removeModule( name );
                }

                // Get the precision policy
                auto precision = this->getComputePrecision().getPolicy();

                // Layer 1: Input (784) -> Hidden1 (128)
                // Create first MLP block
                mlp1_ = std::make_shared<MLP<TDeviceType, TDataType>>(
                    this->getName() + ".mlp1",
                    this->getDeviceContext(),
                    input_shape_,
                    128,               // Output channels
                    true,              // Use bias
                    this->isTraining(),
                    precision );       // Pass precision policy

                // Layer 2: Hidden1 (128) -> Hidden2 (64)
                // Create second MLP block
                std::vector<size_t> hidden1_shape = { input_shape_[ 0 ], 128 };
                mlp2_ = std::make_shared<MLP<TDeviceType, TDataType>>(
                    this->getName() + ".mlp2",
                    this->getDeviceContext(),
                    hidden1_shape,
                    64,                // Output channels
                    true,              // Use bias
                    this->isTraining(),
                    precision );       // Pass precision policy

                // Layer 3: Hidden2 (64) -> Output (10)
                // Output layer for classification
                output_fc_layer_ = std::make_shared<Linear<TDeviceType, TDataType>>(
                    this->getName() + ".output",
                    this->getDeviceContext(),
                    64,                // Input features
                    MNIST_NUM_CLASSES, // Output classes
                    true,              // Use bias
                    this->isTraining(),
                    precision );       // Pass precision policy

                // Register modules
                this->addModule( "mlp1", mlp1_ );
                this->addModule( "mlp2", mlp2_ );
                this->addModule( "output", output_fc_layer_ );

                // Initialize intermediate tensors
                hidden1_ = Tensor<TDataType, MR>( { input_shape_[ 0 ], 128 } );
                hidden2_ = Tensor<TDataType, MR>( { input_shape_[ 0 ], 64 } );
            }
    };
}