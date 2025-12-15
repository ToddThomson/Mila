/**
 * @file MnistClassifier.ixx
 * @brief MNIST digit classifier using feedforward neural network architecture.
 *
 * Device-templated network implementing a three-layer neural network
 * for MNIST handwritten digit classification (784 -> 128 -> 64 -> 10).
 */

module;
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <ostream>
#include <format>
#include <utility>
#include <optional>

export module Mnist.Classifier;

import Mila;
import Mnist.DataLoader;
import Dnn.Network;

namespace Mila::Mnist
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief MNIST handwritten digit classifier.
     *
     * Three-layer feed-forward neural network architecture:
     *   Input (784) -> Linear (128) -> GELU -> Linear (64) -> GELU -> Linear (10) -> Output
     *
     * Design philosophy:
     * - Three-phase initialization: constructor creates architecture graph; Network base class 
     *   propagates context to children; onBuilding() builds children with shapes and allocates buffers
     * - Context-independent architecture: component graph defined without device knowledge
     * - Shape-agnostic configuration: batch size and MNIST dimensions define structure
     * - Runtime shape determined at onBuilding() time from actual input tensor
     * - Child components stored as concrete types for type safety and direct access
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class MnistClassifier : public Network<TDeviceType, TPrecision>
    {
    public:

        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using NetworkBase = Network<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using GeluType = Gelu<TDeviceType, TPrecision>;
        using ComponentPtr = typename NetworkBase::ComponentPtr;

        /**
         * @brief Construct MNIST classifier network.
         *
         * Architecture is created immediately in the constructor (context-independent).
         * ExecutionContext is created and bound by the Network base class.
         *
         * Construction sequence:
         * 1. Validate batch size
         * 2. Create architecture graph (no device required)
         * 3. Network base creates and binds ExecutionContext
         * 4. Context automatically propagated to children
         *
         * @param device_id Device identifier for network execution.
         * @param name Classifier name for identification.
         * @param batch_size Batch size for training/inference.
         *
         * @throws std::invalid_argument if batch_size <= 0.
         * @throws std::invalid_argument if device_id.type does not match TDeviceType.
         * @throws std::runtime_error if ExecutionContext creation fails.
         *
         * @note Architecture is inspectable immediately after construction.
         *       Network base class handles ExecutionContext creation and propagation.
         */
        explicit MnistClassifier(
            DeviceId device_id,
            const std::string& name,
            int64_t batch_size )
            : NetworkBase( device_id, name ), batch_size_( batch_size )
        {
            if ( batch_size_ <= 0 )
            {
                throw std::invalid_argument( "Batch size must be greater than zero." );
            }

            createGraph();
        }

        ~MnistClassifier() override = default;

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to child components.
         *
         * All setup and validation was done in onBuilding(). This method chains
         * forward calls through the classifier structure using pre-allocated buffers.
         *
         * @param input Input tensor containing flattened MNIST images (batch_size, 784)
         * @param output Output tensor for class logits (batch_size, 10)
         */
        void forward( const ITensor& input, ITensor& output )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "MnistClassifier must be built before calling forward." );
            }

            fc1_->forward( input, *hidden1_pre_act_ );
            gelu1_->forward( *hidden1_pre_act_, *hidden1_ );
            fc2_->forward( *hidden1_, *hidden2_pre_act_ );
            gelu2_->forward( *hidden2_pre_act_, *hidden2_ );
            output_fc_->forward( *hidden2_, output );
        }

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to child components.
         *
         * Chains backward calls through the classifier structure in reverse order.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "MnistClassifier must be built before calling backward." );
            }

            auto device = this->getDeviceId();

            TensorType hidden2_grad( device, hidden2_shape_ );
            zeros( hidden2_grad );
            output_fc_->backward( *hidden2_, output_grad, hidden2_grad );

            TensorType hidden2_pre_grad( device, hidden2_shape_ );
            zeros( hidden2_pre_grad );
            gelu2_->backward( *hidden2_pre_act_, hidden2_grad, hidden2_pre_grad );

            TensorType hidden1_grad( device, hidden1_shape_ );
            zeros( hidden1_grad );
            fc2_->backward( *hidden1_, hidden2_pre_grad, hidden1_grad );

            TensorType hidden1_pre_grad( device, hidden1_shape_ );
            zeros( hidden1_pre_grad );
            gelu1_->backward( *hidden1_pre_act_, hidden1_grad, hidden1_pre_grad );

            fc1_->backward( input, hidden1_pre_grad, input_grad );
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        DeviceId getDeviceId() const override
        {
            return NetworkBase::getDeviceId();
        }

        void synchronize() override
        {
            NetworkBase::synchronize();
        }

        size_t parameterCount() const override
        {
            return NetworkBase::parameterCount();
        }

        std::vector<ITensor*> getParameters() const override
        {
            return NetworkBase::getParameters();
        }

        std::vector<ITensor*> getGradients() const override
        {
            return NetworkBase::getGradients();
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << std::endl;
            oss << "MNIST Classifier: " << this->getName() << std::endl;

            if ( this->hasExecutionContext() )
            {
                oss << "Device: " << getDeviceId().toString() << std::endl;
            }
            else
            {
                oss << "Device: (context not set)" << std::endl;
            }

            oss << "Architecture:" << std::endl;
            oss << "  Input:   784 features (28x28 flattened)" << std::endl;
            oss << "  Layer 1: 784 -> " << HIDDEN1_SIZE << " + GELU" << std::endl;
            oss << "  Layer 2: " << HIDDEN1_SIZE << " -> " << HIDDEN2_SIZE << " + GELU" << std::endl;
            oss << "  Output:  " << HIDDEN2_SIZE << " -> " << MNIST_NUM_CLASSES << " classes" << std::endl;

            if ( this->isBuilt() )
            {
                oss << "Parameters: " << parameterCount() << std::endl;
            }

            oss << "Batch size: " << batch_size_ << std::endl;

            if ( this->isBuilt() )
            {
                oss << "Input shape: (";
                for ( size_t i = 0; i < input_shape_.size(); ++i )
                {
                    oss << input_shape_[ i ];
                    if ( i != input_shape_.size() - 1 )
                        oss << ", ";
                }
                oss << ")" << std::endl;

                oss << "Output shape: (";
                for ( size_t i = 0; i < output_shape_.size(); ++i )
                {
                    oss << output_shape_[ i ];
                    if ( i != output_shape_.size() - 1 )
                        oss << ", ";
                }
                oss << ")" << std::endl;
            }

            oss << "Components: " << std::endl;

            if ( fc1_ )
            {
                oss << "  - fc1: " << fc1_->getName() << std::endl;
            }

            if ( gelu1_ )
            {
                oss << "  - gelu1: " << gelu1_->getName() << std::endl;
            }

            if ( fc2_ )
            {
                oss << "  - fc2: " << fc2_->getName() << std::endl;
            }

            if ( gelu2_ )
            {
                oss << "  - gelu2: " << gelu2_->getName() << std::endl;
            }

            if ( output_fc_ )
            {
                oss << "  - output: " << output_fc_->getName() << std::endl;
            }

            oss << std::endl;

            return oss.str();
        }

    protected:

        /**
         * @brief Create the MNIST classifier network graph (context-independent).
         *
         * Defines the computational graph:
         *   fc1 -> gelu1 -> fc2 -> gelu2 -> output
         *
         * Components are created in shared mode (no ExecutionContext).
         * Context binding happens automatically via Network base class which
         * creates ExecutionContext and propagates to children.
         *
         * Called from constructor before ExecutionContext is created.
         * This enables architecture introspection without requiring a device.
         */
        void createGraph()
        {
            addLinear( "fc1", MNIST_IMAGE_SIZE, HIDDEN1_SIZE );
            addActivation( "gelu1" );
            addLinear( "fc2", HIDDEN1_SIZE, HIDDEN2_SIZE );
            addActivation( "gelu2" );
            addLinear( "output", HIDDEN2_SIZE, MNIST_NUM_CLASSES );
        }

        /**
         * @brief Hook invoked during build() to initialize network with input shape.
         *
         * Validates input shape, computes per-layer shapes, caches typed pointers to children,
         * builds all child components with appropriate shapes, and allocates intermediate buffers.
         *
         * All children have ExecutionContext at this point (propagated by Network base).
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            input_shape_ = input_shape;

            hidden1_shape_ = input_shape;
            hidden1_shape_.back() = HIDDEN1_SIZE;

            hidden2_shape_ = input_shape;
            hidden2_shape_.back() = HIDDEN2_SIZE;

            output_shape_ = input_shape;
            output_shape_.back() = MNIST_NUM_CLASSES;

            fc1_ = this->template getComponentAs<LinearType>( this->getName() + ".fc1" );
            fc1_->build( input_shape );

            gelu1_ = this->template getComponentAs<GeluType>( this->getName() + ".gelu1" );
            gelu1_->build( hidden1_shape_ );

            fc2_ = this->template getComponentAs<LinearType>( this->getName() + ".fc2" );
            fc2_->build( hidden1_shape_ );

            gelu2_ = this->template getComponentAs<GeluType>( this->getName() + ".gelu2" );
            gelu2_->build( hidden2_shape_ );

            output_fc_ = this->template getComponentAs<LinearType>( this->getName() + ".output" );
            output_fc_->build( hidden2_shape_ );

            auto device = this->getDeviceId();

            hidden1_pre_act_ = std::make_shared<TensorType>( device, hidden1_shape_ );
            hidden1_pre_act_->setName( this->getName() + ".hidden1_pre_act" );

            hidden1_ = std::make_shared<TensorType>( device, hidden1_shape_ );
            hidden1_->setName( this->getName() + ".hidden1" );

            hidden2_pre_act_ = std::make_shared<TensorType>( device, hidden2_shape_ );
            hidden2_pre_act_->setName( this->getName() + ".hidden2_pre_act" );

            hidden2_ = std::make_shared<TensorType>( device, hidden2_shape_ );
            hidden2_->setName( this->getName() + ".hidden2" );
        }

    private:
        static constexpr int64_t HIDDEN1_SIZE = 128;
        static constexpr int64_t HIDDEN2_SIZE = 64;

        int64_t batch_size_;

        shape_t input_shape_;
        shape_t hidden1_shape_;
        shape_t hidden2_shape_;
        shape_t output_shape_;

        std::shared_ptr<LinearType> fc1_{ nullptr };
        std::shared_ptr<GeluType> gelu1_{ nullptr };
        std::shared_ptr<LinearType> fc2_{ nullptr };
        std::shared_ptr<GeluType> gelu2_{ nullptr };
        std::shared_ptr<LinearType> output_fc_{ nullptr };

        std::shared_ptr<TensorType> hidden1_pre_act_{ nullptr };
        std::shared_ptr<TensorType> hidden1_{ nullptr };
        std::shared_ptr<TensorType> hidden2_pre_act_{ nullptr };
        std::shared_ptr<TensorType> hidden2_{ nullptr };

        /**
         * @brief Helper to create and register a linear layer child component.
         *
         * @param suffix Component name suffix (will be prefixed with classifier name)
         * @param in_features Input feature dimension
         * @param out_features Output feature dimension
         */
        void addLinear( const std::string& suffix, dim_t in_features, dim_t out_features )
        {
            auto cfg = LinearConfig( in_features, out_features ).withBias( false );

            auto component = std::make_shared<LinearType>( cfg, std::nullopt );
            component->setName( this->getName() + "." + suffix );

            this->addComponent( component );
        }

        /**
         * @brief Helper to create and register an activation layer child component.
         *
         * @param suffix Component name suffix
         */
        void addActivation( const std::string& suffix )
        {
            auto cfg = GeluConfig();

            auto component = std::make_shared<GeluType>( cfg );
            component->setName( this->getName() + "." + suffix );

            this->addComponent( component );
        }

        /**
         * @brief Validate input shape for MNIST classifier.
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.empty() )
            {
                throw std::invalid_argument( "MnistClassifier: input must have rank >= 1" );
            }

            int64_t input_features = input_shape.back();

            if ( input_features != MNIST_IMAGE_SIZE )
            {
                std::ostringstream oss;
                oss << "MnistClassifier: input feature dimension mismatch. Expected "
                    << MNIST_IMAGE_SIZE << " (28x28 flattened), got " << input_features;
                throw std::invalid_argument( oss.str() );
            }
        }
    };
}