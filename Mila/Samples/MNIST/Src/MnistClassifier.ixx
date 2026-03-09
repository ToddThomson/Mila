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
     * Construction Pattern:
     * 1. Constructor creates and owns ExecutionContext for specified device
     * 2. Build component graph (context-independent)
     * 3. Propagate context to self and all children
     * 4. onBuilding() hook builds children with shapes and allocates buffers
     *
     * Serialization Contract:
     * - Implements save_() override to write type identifier and configuration
     * - Provides static Load() factory method for type-safe deserialization
     * - Base Network class handles component graph topology serialization
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
         * Follows the concrete network construction pattern:
         * 1. Create and own ExecutionContext for specified device
         * 2. Build component graph (context-independent)
         * 3. Propagate context to self and all children
         *
         * @param name Classifier name for identification and serialization
         * @param batch_size Batch size for training/inference
         * @param device_id Device identifier for network execution
         *
         * @throws std::invalid_argument if batch_size <= 0
         * @throws std::invalid_argument if device_id.type does not match TDeviceType
         * @throws std::runtime_error if ExecutionContext creation fails
         */
        explicit MnistClassifier( const std::string& name, int64_t batch_size, DeviceId device_id )
            : NetworkBase( name ), owned_context_( createExecutionContext( device_id ) ), batch_size_( batch_size )
        {
            if ( batch_size_ <= 0 )
            {
                throw std::invalid_argument( "MnistClassifier: batch size must be positive" );
            }

            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format( "MnistClassifier: device type mismatch: expected {}, got {}",
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type ) ) );
            }

            createGraph();

            // Propagate owned ExecutionContext to self and children
            this->setExecutionContext( owned_context_.get() );
        }

        ~MnistClassifier() override = default;

        /**
         * @brief Load MnistClassifier from archive.
         *
         * Static factory method for type-safe deserialization. Reconstructs
         * the classifier by:
         * 1. Reading configuration from archive metadata
         * 2. Constructing via normal constructor (creates graph + context)
         * 3. Building with saved input shape
         * 4. Loading component weights into built components
         *
         * @param archive Archive containing serialized classifier
         * @param device_id Device for execution (may differ from saved device)
         * @return Unique pointer to reconstructed MnistClassifier
         *
         * @throws std::runtime_error if archive is malformed
         * @throws std::runtime_error if configuration is invalid
         */
        static std::unique_ptr<MnistClassifier> Load( ModelArchive& archive, DeviceId device_id )
        {
            auto scope = ModelArchive::ScopedScope( archive, "network" );

            SerializationMetadata meta = archive.readMetadata( "classifier_meta.json" );

            std::string name = meta.getString( "name" );
            int64_t batch_size = meta.getInt( "batch_size" );

            int64_t saved_hidden1 = meta.getInt( "hidden1_size" );
            int64_t saved_hidden2 = meta.getInt( "hidden2_size" );
            int64_t saved_classes = meta.getInt( "num_classes" );

            if ( saved_hidden1 != HIDDEN1_SIZE || saved_hidden2 != HIDDEN2_SIZE ||
                saved_classes != MNIST_NUM_CLASSES )
            {
                throw std::runtime_error(
                    std::format( "MnistClassifier::Load: architecture mismatch. "
                        "Archive has [{}, {}, {}], expected [{}, {}, {}]",
                        saved_hidden1, saved_hidden2, saved_classes,
                        HIDDEN1_SIZE, HIDDEN2_SIZE, MNIST_NUM_CLASSES ) );
            }

            auto classifier = std::make_unique<MnistClassifier>( name, batch_size, device_id );

            shape_t input_shape = meta.getShape( "input_shape" );
            classifier->build( input_shape );

            loadComponentWeights( archive, classifier.get() );

            return classifier;
        }

        // ====================================================================
        // Compute operation dispatch (new API: component-owned outputs)
        // ====================================================================

        /**
         * @brief Forward pass using child components' new forward() API.
         *
         * Chains component-owned outputs without allocating intermediate network
         * activation buffers. Final logits are copied into network-owned output
         * buffer to preserve the external return semantics.
         *
         * @param input Forward input tensor
         * @return Pointer to network-owned output tensor (logits)
         */
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "MnistClassifier: must be built before forward pass" );
            }

            auto& fc1_out = fc1_->forward( input );
            this->getExecutionContext()->synchronize();

            fc1_out_ptr_ = &fc1_out;

            auto& gelu1_out = gelu1_->forward( *fc1_out_ptr_ );
            this->getExecutionContext()->synchronize();

            gelu1_out_ptr_ = &gelu1_out;

            auto& fc2_out = fc2_->forward( *gelu1_out_ptr_ );
            this->getExecutionContext()->synchronize();

            fc2_out_ptr_ = &fc2_out;

            auto& gelu2_out = gelu2_->forward( *fc2_out_ptr_ );
            this->getExecutionContext()->synchronize();

            gelu2_out_ptr_ = &gelu2_out;

            auto& logits = output_fc_->forward( *gelu2_out_ptr_ );
            this->getExecutionContext()->synchronize();

            logits_ptr_ = &logits;

            // Copy final logits into network-owned output buffer and return it
            //copy( *logits_ptr_, *owned_output_ );
            //this->getExecutionContext()->synchronize();

            return *logits_ptr_;
        }

        /**
         * @brief Backward pass using child components' new backward() API.
         *
         * Chains component backward calls using the forward inputs/outputs that
         * were cached during the forward() call (component-owned pointers).
         * Returns the component-owned input-gradient produced by the first layer.
         *
         * @param input Original forward input tensor
         * @param output_grad Gradient w.r.t. classifier output
         * @return Pointer to component-owned input gradient tensor
         */
        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "MnistClassifier: must be built before backward pass" );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "MnistClassifier: backward requires training mode (setTraining(true))." );
            }

            // Validate we have cached forward activation pointers
            if ( !gelu2_out_ptr_ || !fc2_out_ptr_ || !gelu1_out_ptr_ || !fc1_out_ptr_ )
            {
                throw std::runtime_error( "MnistClassifier: forward activations not present for backward. Call forward() before backward()." );
            }

            // output_fc backward -> gradient w.r.t. gelu2_out (component-owned)
            auto& hidden2_grad_ptr = output_fc_->backward( *gelu2_out_ptr_, output_grad );
            this->getExecutionContext()->synchronize();

            // gelu2 backward -> gradient w.r.t. fc2_out (component-owned)
            auto& hidden2_pre_grad_ptr = gelu2_->backward( *fc2_out_ptr_, hidden2_grad_ptr );
            this->getExecutionContext()->synchronize();

            // fc2 backward -> gradient w.r.t. gelu1_out (component-owned)
            auto& hidden1_grad_ptr = fc2_->backward( *gelu1_out_ptr_, hidden2_pre_grad_ptr );
            this->getExecutionContext()->synchronize();

            // gelu1 backward -> gradient w.r.t. fc1_out (component-owned)
            auto& hidden1_pre_grad_ptr = gelu1_->backward( *fc1_out_ptr_, hidden1_grad_ptr );
            this->getExecutionContext()->synchronize();

            // fc1 backward -> gradient w.r.t. input (component-owned)
            auto& input_grad_ptr = fc1_->backward( input, hidden1_pre_grad_ptr );
            this->getExecutionContext()->synchronize();

            // Return the component-owned input gradient
            return input_grad_ptr;
        }

        void zeroGradients() override
        {
            // If not built, nothing to clear
            if ( !this->isBuilt() )
            {
                return;
            }

            // Zero gradients in child components only; intermediate activation
            // buffers and grads are owned and managed by components now.
            fc1_->zeroGradients();
            fc2_->zeroGradients();
            output_fc_->zeroGradients();
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
            {
                stats += child->getMemoryStats();
            }

            if ( owned_output_ != nullptr )
            {
                stats.device_state_bytes += owned_output_->getStorageSize();
            }

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << std::endl;
            oss << "MNIST Classifier: " << this->getName() << std::endl;
            oss << "Device: " << this->getDeviceId().toString() << std::endl;

            oss << "Architecture:" << std::endl;
            oss << "  Input:   784 features (28x28 flattened)" << std::endl;
            oss << "  Layer 1: 784 -> " << HIDDEN1_SIZE << " + GELU" << std::endl;
            oss << "  Layer 2: " << HIDDEN1_SIZE << " -> " << HIDDEN2_SIZE << " + GELU" << std::endl;
            oss << "  Output:  " << HIDDEN2_SIZE << " -> " << MNIST_NUM_CLASSES << " classes" << std::endl;

            if ( this->isBuilt() )
            {
                oss << "Parameters: " << this->parameterCount() << std::endl;
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

        IExecutionContext* getExecutionContext() const
        {
            return NetworkBase::getExecutionContext();
        }

    protected:

        /**
         * @brief Save classifier-specific configuration (required by Network base).
         *
         * Implements the serialization contract by writing type identifier
         * and configuration metadata to enable reconstruction via Load(). 
         *
         * @param archive Archive to write to
         * @param mode Serialization mode (passed from Network::save())
         */
        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
        {
            SerializationMetadata meta;
            meta.set( "type", "MnistClassifier" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() )
                .set( "batch_size", batch_size_ );

            if ( this->isBuilt() )
            {
                meta.set( "input_shape", input_shape_ )
                    .set( "output_shape", output_shape_ )
                    .set( "hidden1_shape", hidden1_shape_ )
                    .set( "hidden2_shape", hidden2_shape_ );
            }

            meta.set( "hidden1_size", HIDDEN1_SIZE )
                .set( "hidden2_size", HIDDEN2_SIZE )
                .set( "num_classes", MNIST_NUM_CLASSES );

            archive.writeMetadata( "classifier_meta.json", meta );
        }

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

            // Cache typed pointers to children
            fc1_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_1" );
            fc1_->build( input_shape );

            gelu1_ = this->template getComponentAs<GeluType>( this->getName() + ".gelu_1" );
            gelu1_->build( hidden1_shape_ );

            fc2_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_2" );
            fc2_->build( hidden1_shape_ );

            gelu2_ = this->template getComponentAs<GeluType>( this->getName() + ".gelu_2" );
            gelu2_->build( hidden2_shape_ );

            output_fc_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_output" );
            output_fc_->build( hidden2_shape_ );

            // Allocate network-owned output buffer (logits)
            auto device = this->getDeviceId();

            owned_output_ = std::make_shared<TensorType>( device, output_shape_ );
            owned_output_->setName( this->getName() + ".output" );

            // Clear cached component pointers used for chaining (will be set during forward)
            fc1_out_ptr_ = nullptr;
            gelu1_out_ptr_ = nullptr;
            fc2_out_ptr_ = nullptr;
            gelu2_out_ptr_ = nullptr;
            logits_ptr_ = nullptr;
        }

    private:

        static constexpr int64_t HIDDEN1_SIZE = 128;
        static constexpr int64_t HIDDEN2_SIZE = 64;

        // Owned ExecutionContext
        std::unique_ptr<IExecutionContext> owned_context_{ nullptr };

        // Configuration
        int64_t batch_size_;

        // Computed shapes (cached during build)
        shape_t input_shape_;
        shape_t hidden1_shape_;
        shape_t hidden2_shape_;
        shape_t output_shape_;

        // Typed component pointers (cached during onBuilding)
        std::shared_ptr<LinearType> fc1_{ nullptr };
        std::shared_ptr<GeluType> gelu1_{ nullptr };
        std::shared_ptr<LinearType> fc2_{ nullptr };
        std::shared_ptr<GeluType> gelu2_{ nullptr };
        std::shared_ptr<LinearType> output_fc_{ nullptr };

        // Component-owned activation pointers cached during forward
        TensorType* fc1_out_ptr_{ nullptr };
        TensorType* gelu1_out_ptr_{ nullptr };
        TensorType* fc2_out_ptr_{ nullptr };
        TensorType* gelu2_out_ptr_{ nullptr };
        TensorType* logits_ptr_{ nullptr };

        // Network-owned forward output buffer (new API)
        std::shared_ptr<TensorType> owned_output_{ nullptr };

        /**
         * @brief Create the MNIST classifier network graph (context-independent).
         *
         * Defines the computational graph:
         *   fc1 -> gelu1 -> fc2 -> gelu2 -> output
         *
         * Components are created without ExecutionContext (shared mode).
         * Context will be propagated after this method returns via setExecutionContext().
         */
        void createGraph()
        {
            addLinear( "fc_1", MNIST_IMAGE_SIZE, HIDDEN1_SIZE );
            addActivation( "gelu_1" );
            addLinear( "fc_2", HIDDEN1_SIZE, HIDDEN2_SIZE );
            addActivation( "gelu_2" );
            addLinear( "fc_output", HIDDEN2_SIZE, MNIST_NUM_CLASSES );
        }

        /**
         * @brief Helper to create and register a linear layer child component.
         *
         * @param suffix Component name suffix (will be prefixed with classifier name)
         * @param in_features Input feature dimension
         * @param out_features Output feature dimension
         */
        void addLinear( const std::string& suffix, dim_t in_features, dim_t out_features )
        {
            auto cfg = LinearConfig( in_features, out_features )
                .withBias( false );

            auto linear = std::make_shared<LinearType>(
                this->getName() + "." + suffix, cfg, std::nullopt );

            this->addComponent( linear );
        }

        /**
         * @brief Helper to create and register an activation layer child component.
         *
         * @param suffix Component name suffix
         */
        void addActivation( const std::string& suffix )
        {
            auto cfg = GeluConfig();

            auto gelu = std::make_shared<GeluType>(
                this->getName() + "." + suffix, cfg, std::nullopt );

            this->addComponent( gelu );
        }

        /**
         * @brief Validate input shape for MNIST classifier.
         *
         * @throws std::invalid_argument if shape is invalid
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
                throw std::invalid_argument(
                    std::format( "MnistClassifier: input feature dimension mismatch. "
                        "Expected {} (28x28 flattened), got {}",
                        MNIST_IMAGE_SIZE, input_features ) );
            }
        }

        /**
         * @brief Load component weights from archive.
         *
         * Helper method for Load() to populate weights into already-built components.
         * Base Network class handles component graph traversal; this method loads
         * weights into each component.
         *
         * @param archive Archive containing serialized weights
         * @param classifier Classifier instance with built components
         *
         * @note This is a placeholder - actual implementation will iterate over
         *       components and load their weights from the archive.
         */
        static void loadComponentWeights( ModelArchive& /*archive*/,
            MnistClassifier* /*classifier*/ )
        {
            // TODO: Implement weight loading
            // For each component in classifier->getComponents():
            //   - Read component weights from archive
            //   - Set weights on component via setParameters()
        }
    };
}