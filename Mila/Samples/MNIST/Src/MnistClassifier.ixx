/**
 * @file MnistClassifier.ixx
 * @brief MNIST digit classifier using feedforward neural network architecture.
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
#include <ostream>

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
     * Three-layer feed-forward neural network architecture:
     *   Input (784) -> Linear (128) -> GELU -> Linear (64) -> GELU -> Linear (10) -> Output
     *
     * Design philosophy:
     * - Two-phase initialization: build() performs shape validation and buffer allocation
     * - Composite module pattern: manages child modules (3x Linear + 2x Gelu)
     * - Shape-agnostic configuration: batch size and MNIST dimensions define structure
     * - Runtime shapes determined at build() time from actual input tensor
     * - Child modules stored as concrete types for type safety and direct access
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
        using LinearType = Linear<TDeviceType, TPrecision>;
        using GeluType = Gelu<TDeviceType, TPrecision>;

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
            return built_ && fc1_ && gelu1_ && fc2_ && gelu2_ && output_fc_;
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

            cached_hidden1_shape_ = input_shape;
            cached_hidden1_shape_.back() = HIDDEN1_SIZE;

            cached_hidden2_shape_ = input_shape;
            cached_hidden2_shape_.back() = HIDDEN2_SIZE;

            cached_output_shape_ = input_shape;
            cached_output_shape_.back() = MNIST_NUM_CLASSES;

            fc1_->build( input_shape );
            gelu1_->build( cached_hidden1_shape_ );
            fc2_->build( cached_hidden1_shape_ );
            gelu2_->build( cached_hidden2_shape_ );
            output_fc_->build( cached_hidden2_shape_ );

            auto device = exec_context_->getDevice();

            hidden1_pre_act_ = std::make_shared<TensorType>( device, cached_hidden1_shape_ );
            hidden1_pre_act_->setName( name_ + ".hidden1_pre_act" );

            hidden1_ = std::make_shared<TensorType>( device, cached_hidden1_shape_ );
            hidden1_->setName( name_ + ".hidden1" );

            hidden2_pre_act_ = std::make_shared<TensorType>( device, cached_hidden2_shape_ );
            hidden2_pre_act_->setName( name_ + ".hidden2_pre_act" );

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
        void forward( const ITensor& input, ITensor& output )
        {
            if (!isBuilt())
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
         * @brief Backward pass - HOT PATH, pure dispatch to child modules.
         *
         * Chains backward calls through the classifier structure in reverse order.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "MnistClassifier must be built before calling backward." );
            }

            auto device = exec_context_->getDevice();

            TensorType hidden2_grad( device, cached_hidden2_shape_ );
            zeros( hidden2_grad );
            output_fc_->backward( *hidden2_, output_grad, hidden2_grad );

            TensorType hidden2_pre_grad( device, cached_hidden2_shape_ );
            zeros( hidden2_pre_grad );
            gelu2_->backward( *hidden2_pre_act_, hidden2_grad, hidden2_pre_grad );

            TensorType hidden1_grad( device, cached_hidden1_shape_ );
            zeros( hidden1_grad );
            fc2_->backward( *hidden1_, hidden2_pre_grad, hidden1_grad );

            TensorType hidden1_pre_grad( device, cached_hidden1_shape_ );
            zeros( hidden1_pre_grad );
            gelu1_->backward( *hidden1_pre_act_, hidden1_grad, hidden1_pre_grad );

            fc1_->backward( input, hidden1_pre_grad, input_grad );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save( ModelArchive& archive ) const override
        {
            fc1_->save( archive );
            gelu1_->save( archive );
            fc2_->save( archive );
            gelu2_->save( archive );
            output_fc_->save( archive );
        }

        void load( ModelArchive& archive ) override
        {
            fc1_->load( archive );
            gelu1_->load( archive );
            fc2_->load( archive );
            gelu2_->load( archive );
            output_fc_->load( archive );
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

            fc1_->synchronize();
            gelu1_->synchronize();
            fc2_->synchronize();
            gelu2_->synchronize();
            output_fc_->synchronize();
        }

        void setTraining( bool is_training ) override
        {
            CompositeModuleBase::setTraining( is_training );

            fc1_->setTraining( is_training );
            gelu1_->setTraining( is_training );
            fc2_->setTraining( is_training );
            gelu2_->setTraining( is_training );
            output_fc_->setTraining( is_training );
        }

        bool isTraining() const override
        {
            return CompositeModuleBase::isTraining();
        }

        size_t parameterCount() const override
        {
            size_t total = 0;

            total += fc1_->parameterCount();
            total += gelu1_->parameterCount();
            total += fc2_->parameterCount();
            total += gelu2_->parameterCount();
            total += output_fc_->parameterCount();

            return total;
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;

            auto fc1_params = fc1_->getParameters();
            params.insert( params.end(), fc1_params.begin(), fc1_params.end() );

            auto gelu1_params = gelu1_->getParameters();
            params.insert( params.end(), gelu1_params.begin(), gelu1_params.end() );

            auto fc2_params = fc2_->getParameters();
            params.insert( params.end(), fc2_params.begin(), fc2_params.end() );

            auto gelu2_params = gelu2_->getParameters();
            params.insert( params.end(), gelu2_params.begin(), gelu2_params.end() );

            auto output_params = output_fc_->getParameters();
            params.insert( params.end(), output_params.begin(), output_params.end() );

            return params;
        }

        std::vector<ITensor*> getParameterGradients() const override
        {
            std::vector<ITensor*> grads;

            auto fc1_grads = fc1_->getParameterGradients();
            grads.insert( grads.end(), fc1_grads.begin(), fc1_grads.end() );

            auto gelu1_grads = gelu1_->getParameterGradients();
            grads.insert( grads.end(), gelu1_grads.begin(), gelu1_grads.end() );

            auto fc2_grads = fc2_->getParameterGradients();
            grads.insert( grads.end(), fc2_grads.begin(), fc2_grads.end() );

            auto gelu2_grads = gelu2_->getParameterGradients();
            grads.insert( grads.end(), gelu2_grads.begin(), gelu2_grads.end() );

            auto output_grads = output_fc_->getParameterGradients();
            grads.insert( grads.end(), output_grads.begin(), output_grads.end() );

            return grads;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << std::endl;
            oss << "MNIST Classifier: " << name_ << std::endl;
            oss << "Architecture:" << std::endl;
            oss << "  Input:   784 features (28x28 flattened)" << std::endl;
            oss << "  Layer 1: 784 -> " << HIDDEN1_SIZE << " + GELU" << std::endl;
            oss << "  Layer 2: " << HIDDEN1_SIZE << " -> " << HIDDEN2_SIZE << " + GELU" << std::endl;
            oss << "  Output:  " << HIDDEN2_SIZE << " -> " << MNIST_NUM_CLASSES << " classes" << std::endl;
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
            oss << "  - fc1: " << fc1_->getName() << std::endl;
            oss << "  - gelu1: " << gelu1_->getName() << std::endl;
            oss << "  - fc2: " << fc2_->getName() << std::endl;
            oss << "  - gelu2: " << gelu2_->getName() << std::endl;
            oss << "  - output: " << output_fc_->getName() << std::endl;

            oss << std::endl;

            return oss.str();
        }

        // ====================================================================
        // Child module accessors
        // ====================================================================

        std::shared_ptr<LinearType> getFC1() const noexcept
        {
            return fc1_;
        }

        std::shared_ptr<GeluType> getGelu1() const noexcept
        {
            return gelu1_;
        }

        std::shared_ptr<LinearType> getFC2() const noexcept
        {
            return fc2_;
        }

        std::shared_ptr<GeluType> getGelu2() const noexcept
        {
            return gelu2_;
        }

        std::shared_ptr<LinearType> getOutputFC() const noexcept
        {
            return output_fc_;
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
        }

    private:
        static constexpr int64_t HIDDEN1_SIZE = 128;
        static constexpr int64_t HIDDEN2_SIZE = 64;

        std::string name_;
        int64_t batch_size_;
        bool built_{ false };
        std::shared_ptr<ExecutionContextType> exec_context_;

        shape_t cached_input_shape_;
        shape_t cached_hidden1_shape_;
        shape_t cached_hidden2_shape_;
        shape_t cached_output_shape_;

        std::shared_ptr<LinearType> fc1_{ nullptr };
        std::shared_ptr<GeluType> gelu1_{ nullptr };
        std::shared_ptr<LinearType> fc2_{ nullptr };
        std::shared_ptr<GeluType> gelu2_{ nullptr };
        std::shared_ptr<LinearType> output_fc_{ nullptr };

        std::shared_ptr<TensorType> hidden1_pre_act_{ nullptr };
        std::shared_ptr<TensorType> hidden1_{ nullptr };
        std::shared_ptr<TensorType> hidden2_pre_act_{ nullptr };
        std::shared_ptr<TensorType> hidden2_{ nullptr };

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

        void createModules()
        {
            auto fc1_config = LinearConfig( MNIST_IMAGE_SIZE, HIDDEN1_SIZE )
                .withName( name_ + ".fc1" )
                .withBias( false );

            fc1_ = std::make_shared<LinearType>( exec_context_, fc1_config );

            auto gelu1_config = GeluConfig();
            gelu1_config.withName( name_ + ".gelu1" );

            gelu1_ = std::make_shared<GeluType>( exec_context_, gelu1_config );

            auto fc2_config = LinearConfig( HIDDEN1_SIZE, HIDDEN2_SIZE );
            fc2_config.withName( name_ + ".fc2" )
                .withBias( false );

            fc2_ = std::make_shared<LinearType>( exec_context_, fc2_config );

            auto gelu2_config = GeluConfig();
            gelu2_config.withName( name_ + ".gelu2" );

            gelu2_ = std::make_shared<GeluType>( exec_context_, gelu2_config );

            auto output_config = LinearConfig( HIDDEN2_SIZE, MNIST_NUM_CLASSES );
            output_config.withName( name_ + ".output" )
                .withBias( false );

            output_fc_ = std::make_shared<LinearType>( exec_context_, output_config );
        }
    };

    export template<TensorDataType TPrecision>
        using CpuMnistClassifier = MnistClassifier<DeviceType::Cpu, TPrecision>;

    export template<TensorDataType TPrecision>
        using CudaMnistClassifier = MnistClassifier<DeviceType::Cuda, TPrecision>;
}