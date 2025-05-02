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

    export template<typename TPrecision, DeviceType TDeviceType>
    class MnistClassifier : public BlockModule<TPrecision, TPrecision, TDeviceType> {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, HostMemoryResource>;
        using BlockModuleBase = BlockModule<TPrecision, TPrecision, TDeviceType>;

        MnistClassifier( const std::string& name, const std::string& device_name, size_t batch_size )
            : BlockModuleBase( device_name ) {

            this->setName( name );
            input_shape_ = { batch_size, static_cast<size_t>(MNIST_IMAGE_SIZE) };

            initializeModules();
        }

        MnistClassifier( const std::string& name, std::shared_ptr<DeviceContext> context, size_t batch_size )
            : BlockModuleBase( context ) {

            this->setName( name );
            input_shape_ = { batch_size, static_cast<size_t>(MNIST_IMAGE_SIZE) };

            initializeModules();
        }

        void forward( const Tensor<TPrecision, MR>& input, Tensor<TPrecision, MR>& output ) {
            mlp1_->forward( input, hidden1_ );

            // Second MLP block: hidden1 -> hidden2
            mlp2_->forward( hidden1_, hidden2_ );

            // Output layer: hidden2 -> output (logits)
            output_layer_->forward( hidden2_, output );
        }

        size_t parameterCount() const override {
            size_t total = 0;
            for ( const auto& module : this->getModules() ) {
                total += module->parameterCount();
            }
            return total;
        }

        void save( mz_zip_archive& zip ) const override {
            for ( const auto& module : this->getModules() ) {
                module->save( zip );
            }
        }

        void load( mz_zip_archive& zip ) override {
            for ( const auto& module : this->getModules() ) {
                module->load( zip );
            }
        }

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
            oss << "====================" << std::endl;

            return oss.str();
        }

    private:
        std::vector<size_t> input_shape_;

        // Network layers
        std::shared_ptr<MLP<TPrecision, TDeviceType>> mlp1_;
        std::shared_ptr<MLP<TPrecision, TDeviceType>> mlp2_;
        std::shared_ptr<FullyConnected<TPrecision, TDeviceType>> output_layer_;

        // Intermediate tensors for activations
        Tensor<TPrecision, MR> hidden1_;
        Tensor<TPrecision, MR> hidden2_;

        void initializeModules() {
            // Clear any existing modules
            for ( const auto& [name, _] : this->getNamedModules() ) {
                this->removeModule( name );
            }

            // Layer 1: Input (784) -> Hidden1 (128)
            // Create first MLP block
            mlp1_ = std::make_shared<MLP<TPrecision, TDeviceType>>(
                this->getName() + ".mlp1",
                this->getDeviceContext(),
                input_shape_,
                128,  // Output channels
                true, // Use bias
                this->isTraining() );

            // Layer 2: Hidden1 (128) -> Hidden2 (64)
            // Create second MLP block
            std::vector<size_t> hidden1_shape = { input_shape_[ 0 ], 128 };
            mlp2_ = std::make_shared<MLP<TPrecision, TDeviceType>>(
                this->getName() + ".mlp2",
                this->getDeviceContext(),
                hidden1_shape,
                64,   // Output channels
                true, // Use bias
                this->isTraining() );

            // Layer 3: Hidden2 (64) -> Output (10)
            // Output layer for classification
            output_layer_ = std::make_shared<FullyConnected<TPrecision, TDeviceType>>(
                this->getName() + ".output",
                this->getDeviceContext(),
                64,                    // Input features
                MNIST_NUM_CLASSES,     // Output classes
                true,                  // Use bias
                this->isTraining() );

            // Register modules
            this->addModule( "mlp1", mlp1_ );
            this->addModule( "mlp2", mlp2_ );
            this->addModule( "output", output_layer_ );

            // Initialize intermediate tensors
            hidden1_ = Tensor<TPrecision, MR>( { input_shape_[ 0 ], 128 } );
            hidden2_ = Tensor<TPrecision, MR>( { input_shape_[ 0 ], 64 } );
        }
    };
}