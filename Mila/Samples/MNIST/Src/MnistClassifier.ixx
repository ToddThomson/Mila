/**
 * @file MnistClassifier.ixx
 * @brief MNIST digit classifier using multi-layer perceptron architecture.
 *
 * Device-templated composite module implementing a three-layer neural network
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

export module Mnist.Classifier;

import Mila;
import Mnist.DataLoader;

namespace Mila::Mnist
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief MNIST handwritten digit classifier.
     *
     * Three-layer neural network architecture:
     *   Input (784) -> MLP1 (128) -> MLP2 (64) -> Linear (10) -> Output
     *
     * Design philosophy:
     * - Two-phase initialization: build() performs shape validation and buffer allocation
     * - Composite module pattern: manages child modules (2x MLP + 1x Linear)
     * - Shape-agnostic configuration: batch size and MNIST dimensions define structure
     * - Runtime shapes determined at build() time from actual input tensor
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class MnistClassifier : public CompositeModule<TDeviceType>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using CompositeModuleBase = CompositeModule<TDeviceType>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * @brief Construct classifier with execution context.
         *
         * @param exec_context Shared execution context for device resources.
         * @param name Classifier name for identification.
         * @param batch_size Batch size for training/inference.
         */
        explicit MnistClassifier(
            std::shared_ptr<ExecutionContextType> exec_context, const std::string& name, int64_t batch_size )
            : exec_context_( exec_context ), name_( name ), batch_size_( batch_size )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            if (batch_size <= 0)
            {
                throw std::invalid_argument( "Batch size must be greater than zero." );
            }

            createModules();
        }

        ~MnistClassifier() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        bool isBuilt() const override
        {
            return built_ && mlp1_ && mlp2_ && output_fc_;
        }

        /**
         * @brief Build the classifier for a concrete input shape.
         *
         * This is the COLD PATH where all setup happens ONCE:
         * - Validates input shape matches MNIST_IMAGE_SIZE (784)
         * - Builds all child modules with appropriate shapes
         * - Allocates intermediate buffer tensors for forward/backward passes
         *
         * After build(), forward() becomes pure dispatch.
         */
        void build( const shape_t& input_shape ) override
        {
            if (built_)
                return;

            validateInputShape( input_shape );

            cached_input_shape_ = input_shape;

            // Compute intermediate shapes
            cached_hidden1_shape_ = input_shape;
            cached_hidden1_shape_.back() = HIDDEN1_SIZE;

            cached_hidden2_shape_ = input_shape;
            cached_hidden2_shape_.back() = HIDDEN2_SIZE;

            cached_output_shape_ = input_shape;
            cached_output_shape_.back() = MNIST_NUM_CLASSES;

            // Build child modules
            mlp1_->build( input_shape );
            mlp2_->build( cached_hidden1_shape_ );
            output_fc_->build( cached_hidden2_shape_ );

            // Allocate intermediate buffers
            auto device = exec_context_->getDevice();

            hidden1_ = std::make_shared<TensorType>( device, cached_hidden1_shape_ );
            hidden1_->setName( name_ + ".hidden1" );

            hidden2_ = std::make_shared<TensorType>( device, cached_hidden2_shape_ );
            hidden2_->setName( name_ + ".hidden2" );

            built_ = true;
        }

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to child modules.
         *
         * All setup and validation was done in build(). This method chains
         * forward calls through the classifier structure using pre-allocated buffers.
         *
         * @param input Input tensor containing flattened MNIST images (batch_size, 784)
         * @param output Output tensor for class logits (batch_size, 10)
         */
        void forward( const ITensor& input, ITensor& output ) override
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "MnistClassifier must be built before calling forward." );
            }

            // Input (784) ? MLP1 ? Hidden1 (128)
            mlp1_->forward( input, *hidden1_ );

            // Hidden1 (128) ? MLP2 ? Hidden2 (64)
            mlp2_->forward( *hidden1_, *hidden2_ );

            // Hidden2 (64) ? Linear ? Output (10 logits)
            output_fc_->forward( *hidden2_, output );
        }

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to child modules.
         *
         * Chains backward calls through the classifier structure in reverse order.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "MnistClassifier must be built before calling backward." );
            }

            auto device = exec_context_->getDevice();

            // Backprop through output layer
            TensorType hidden2_grad( device, cached_hidden2_shape_ );
			zeros( hidden2_grad );

            output_fc_->backward( *hidden2_, output_grad, hidden2_grad );

            // Backprop through MLP2
            TensorType hidden1_grad( device, cached_hidden1_shape_ );
			zeros( hidden1_grad );
            mlp2_->backward( *hidden1_, hidden2_grad, hidden1_grad );

            // Backprop through MLP1
            mlp1_->backward( input, hidden1_grad, input_grad );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save( ModelArchive& archive ) const override
        {
            for (const auto& module : this->getModules())
            {
                module->save( archive );
            }
        }

        void load( ModelArchive& archive ) override
        {
            for (const auto& module : this->getModules())
            {
                module->load( archive );
            }
        }

        // ====================================================================
        // Module interface
        // ====================================================================

        std::string getName() const override
        {
            return name_;
        }

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return exec_context_->getDevice();
		}

        void synchronize() override
        {
            if (exec_context_)
            {
                exec_context_->synchronize();
            }

            for (const auto& module : this->getModules())
            {
                module->synchronize();
            }
        }

        void setTraining( bool is_training ) override
        {
            CompositeModuleBase::setTraining( is_training );

            for (auto& module : this->getModules())
            {
                module->setTraining( is_training );
            }
        }

        bool isTraining() const override
        {
            return CompositeModuleBase::isTraining();
        }

        size_t parameterCount() const override
        {
            size_t total = 0;
            for (const auto& module : this->getModules())
            {
                total += module->parameterCount();
            }
            return total;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "====================" << std::endl;
            oss << "MNIST Classifier: " << name_ << std::endl;
            oss << "Architecture:" << std::endl;
            oss << "  Input:  784 features (28x28 flattened)" << std::endl;
            oss << "  MLP1:   784 -> " << HIDDEN1_SIZE << " (with GELU activation)" << std::endl;
            oss << "  MLP2:   " << HIDDEN1_SIZE << " -> " << HIDDEN2_SIZE << " (with GELU activation)" << std::endl;
            oss << "  Output: " << HIDDEN2_SIZE << " -> " << MNIST_NUM_CLASSES << " classes" << std::endl;
            oss << "Parameters: " << parameterCount() << std::endl;

            if (exec_context_ && exec_context_->getDevice())
            {
                oss << "Device: " << deviceTypeToString( exec_context_->getDevice()->getDeviceType() ) << std::endl;
            }

            oss << "Batch size: " << batch_size_ << std::endl;

            if (isBuilt())
            {
                oss << "Input shape: (";
                for (size_t i = 0; i < cached_input_shape_.size(); ++i)
                {
                    oss << cached_input_shape_[i];
                    if (i != cached_input_shape_.size() - 1)
                        oss << ", ";
                }
                oss << ")" << std::endl;

                oss << "Output shape: (";
                for (size_t i = 0; i < cached_output_shape_.size(); ++i)
                {
                    oss << cached_output_shape_[i];
                    if (i != cached_output_shape_.size() - 1)
                        oss << ", ";
                }
                oss << ")" << std::endl;
            }

            oss << "Sub-Modules:" << std::endl;
            for (const auto& [name, module] : this->getNamedModules())
            {
                oss << "  - " << name << std::endl;
            }

            oss << "====================" << std::endl;

            return oss.str();
        }

    protected:

        /**
         * @brief Override buildImpl for sequential shape propagation.
         *
         * This is called by the base CompositeModule::build() after validation.
         * Delegates to the public build() method which handles all setup.
         */
        void buildImpl( const shape_t& input_shape ) override
        {
            // Build is already handled by public build() method
            // This override satisfies CompositeModule interface
        }

    private:
        // Architecture constants
        static constexpr int64_t HIDDEN1_SIZE = 128;
        static constexpr int64_t HIDDEN2_SIZE = 64;

        // Configuration
        std::string name_;
        int64_t batch_size_;
        bool built_{ false };
        std::shared_ptr<ExecutionContextType> exec_context_;

        // Cached shapes determined at build time
        shape_t cached_input_shape_;
        shape_t cached_hidden1_shape_;
        shape_t cached_hidden2_shape_;
        shape_t cached_output_shape_;

        // Child modules
        std::shared_ptr<Module<TDeviceType>> mlp1_{ nullptr };
        std::shared_ptr<Module<TDeviceType>> mlp2_{ nullptr };
        std::shared_ptr<Module<TDeviceType>> output_fc_{ nullptr };

        // Intermediate buffer tensors (allocated at build time)
        std::shared_ptr<TensorType> hidden1_{ nullptr };
        std::shared_ptr<TensorType> hidden2_{ nullptr };

        /**
         * @brief Validate input shape for MNIST classifier.
         *
         * Ensures the last dimension matches MNIST_IMAGE_SIZE (784).
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.empty())
            {
                throw std::invalid_argument( "MnistClassifier: input must have rank >= 1" );
            }

            int64_t input_features = input_shape.back();

            if (input_features != MNIST_IMAGE_SIZE)
            {
                std::ostringstream oss;
                oss << "MnistClassifier: input feature dimension mismatch. Expected "
                    << MNIST_IMAGE_SIZE << " (28x28 flattened), got " << input_features;
                throw std::invalid_argument( oss.str() );
            }
        }

        /**
         * @brief Create and register child modules.
         *
         * Called from constructor to instantiate all child modules and register
         * them with the composite module base for uniform management.
         */
        void createModules()
        {
            // MLP1: 784 -> 128
            auto mlp1_config = MLPConfig( MNIST_IMAGE_SIZE, HIDDEN1_SIZE );
            mlp1_config.withName( name_ + ".mlp1" )
                .withBias( false /* was true */ )
                .withActivation( ActivationType::Gelu )
                .withLayerNorm( false );

            mlp1_ = std::make_shared<MLP<TDeviceType, TPrecision>>( exec_context_, mlp1_config );
            this->addModule( "mlp1", mlp1_ );

            // MLP2: 128 -> 64
            auto mlp2_config = MLPConfig( HIDDEN1_SIZE, HIDDEN2_SIZE );
            mlp2_config.withName( name_ + ".mlp2" )
                .withBias( false /* was true */ )
                .withActivation( ActivationType::Gelu )
                .withLayerNorm( false );

            mlp2_ = std::make_shared<MLP<TDeviceType, TPrecision>>( exec_context_, mlp2_config );
            this->addModule( "mlp2", mlp2_ );

            // Output layer: 64 -> 10
            auto output_config = LinearConfig( HIDDEN2_SIZE, MNIST_NUM_CLASSES );
            output_config.withName( name_ + ".output" )
                .withBias( false /* was true */ );

            output_fc_ = std::make_shared<Linear<TDeviceType, TPrecision>>( exec_context_, output_config );
            this->addModule( "fc", output_fc_ );
        }
    };

    // Convenience aliases for common usages
    export template<TensorDataType TPrecision>
        using CpuMnistClassifier = MnistClassifier<DeviceType::Cpu, TPrecision>;

    export template<TensorDataType TPrecision>
        using CudaMnistClassifier = MnistClassifier<DeviceType::Cuda, TPrecision>;
}