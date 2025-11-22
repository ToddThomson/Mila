/**
 * @file MLP.ixx
 * @brief Multi-Layer Perceptron (MLP) block for neural networks.
 *
 * Device-templated composite module that follows the two-phase initialization pattern.
 * Structure: Input ? Linear(in, hidden) ? [LayerNorm] ? Activation ? Linear(hidden, in) ? Output
 */

module;
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <cstdint>

export module Dnn.Blocks.MLP;
export import :Config;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Module;
import Dnn.CompositeModule;
import Dnn.ActivationType;
import Compute.Precision;
import Compute.MemoryResource;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.OperationRegistry;
import Dnn.Modules.Linear;
import Dnn.Modules.Gelu;
import Dnn.Modules.LayerNorm;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Multi-Layer Perceptron (MLP) block for neural networks.
     *
     * Device-templated composite module that implements a standard MLP structure:
     *   Input -> Linear(in_features, hidden_size) -> [LayerNorm] -> Activation -> Linear(hidden_size, in_features) -> Output
     *
     * Design philosophy:
     * - Two-phase initialization: build() does all setup, forward()/backward() are pure dispatch
     * - Composite module pattern: manages child modules (Linear, activation, LayerNorm)
     * - Shape-agnostic configuration: input_features and hidden_size define architecture
     * - Runtime shape determined at build() time from actual input tensor
     * - Child modules stored as concrete types for type safety and direct access
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class MLP : public CompositeModule<TDeviceType>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using CompositeModuleBase = CompositeModule<TDeviceType>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using GeluType = Gelu<TDeviceType, TPrecision>;
        using LayerNormType = LayerNorm<TDeviceType, TPrecision>;

        /**
         * @brief Construct with an existing execution context.
         *
         * @param exec_context Shared execution context for device resources.
         * @param config MLP configuration.
         */
        explicit MLP( std::shared_ptr<ExecutionContextType> context, const MLPConfig& config )
            : exec_context_( context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            createModules();
        }

        ~MLP() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        bool isBuilt() const override
        {
            return built_ && fc1_ && activation_ && fc2_ &&
                (!config_.useLayerNorm() || norm_);
        }

        /**
         * @brief Build the MLP block for a concrete input shape.
         *
         * This is the COLD PATH where all setup happens ONCE:
         * - Validates input shape compatibility with configured input_features
         * - Builds all child modules with appropriate shapes
         * - Allocates intermediate buffer tensors for forward/backward passes
         *
         * After build(), forward() and backward() become pure dispatch methods.
         */
        void build( const shape_t& input_shape ) override
        {
            if (built_)
                return;

            validateInputShape( input_shape );

            cached_input_shape_ = input_shape;

            cached_hidden_shape_ = input_shape;
            cached_hidden_shape_.back() = config_.getHiddenSize();

            fc1_->build( input_shape );

            if (config_.useLayerNorm())
            {
                norm_->build( cached_hidden_shape_ );
            }

            activation_->build( cached_hidden_shape_ );
            fc2_->build( cached_hidden_shape_ );

            auto device = exec_context_->getDevice();

            fc1_output_ = std::make_shared<TensorType>( device, cached_hidden_shape_ );
            fc1_output_->setName( this->getName() + ".fc1_output" );

            if (config_.useLayerNorm())
            {
                norm_output_ = std::make_shared<TensorType>( device, cached_hidden_shape_ );
                norm_output_->setName( this->getName() + ".norm_output" );
            }

            act_output_ = std::make_shared<TensorType>( device, cached_hidden_shape_ );
            act_output_->setName( this->getName() + ".act_output" );

            built_ = true;
        }

        // ====================================================================
        // Compute operation forward and backward dispatch
        // ====================================================================

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to child modules.
         *
         * All setup and validation was done in build(). This method chains
         * forward calls through the MLP structure using pre-allocated buffers.
         */
        void forward( const ITensor& input, ITensor& output )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "MLP module must be built before calling forward." );
            }

            fc1_->forward( input, *fc1_output_ );

            if (config_.useLayerNorm())
            {
                norm_->forward( *fc1_output_, *norm_output_ );
                activation_->forward( *norm_output_, *act_output_ );
            }
            else
            {
                activation_->forward( *fc1_output_, *act_output_ );
            }

            fc2_->forward( *act_output_, output );
        }

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to child modules.
         *
         * Chains backward calls through the MLP structure in reverse order,
         * using pre-allocated gradient buffers.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "MLP module must be built before calling backward." );
            }

            auto device = exec_context_->getDevice();

            TensorType fc2_grad( device, cached_hidden_shape_ );
            fc2_->backward( *act_output_, output_grad, fc2_grad );

            if (config_.useLayerNorm())
            {
                TensorType act_grad( device, cached_hidden_shape_ );
                activation_->backward( *norm_output_, fc2_grad, act_grad );

                TensorType norm_grad( device, cached_hidden_shape_ );
                norm_->backward( *fc1_output_, act_grad, norm_grad );

                fc1_->backward( input, norm_grad, input_grad );
            }
            else
            {
                TensorType act_grad( device, cached_hidden_shape_ );
                activation_->backward( *fc1_output_, fc2_grad, act_grad );

                fc1_->backward( input, act_grad, input_grad );
            }
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            fc1_->save_( archive, mode );
            
            if (norm_)
            {
                norm_->save_( archive, mode );
            }
            
            activation_->save_( archive, mode );
            fc2_->save_( archive, mode );
        }

        /*void load( ModelArchive& archive, SerializationMode mode ) override
        {
            fc1_->load( archive, mode );
            
            if (norm_)
            {
                norm_->load( archive, mode );
            }
            
            activation_->load( archive, mode );
            fc2_->load( archive, mode );
        }*/

        // ====================================================================
        // Module interface
        // ====================================================================

        std::string getName() const override
        {
            return config_.getName();
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
            
            if (norm_)
            {
                norm_->synchronize();
            }
            
            activation_->synchronize();
            fc2_->synchronize();
        }

        // Delegate training-mode handling to Module::onTrainingChanging via CompositeModule
        size_t parameterCount() const override
        {
            size_t total = 0;
            
            total += fc1_->parameterCount();
            
            if (norm_)
            {
                total += norm_->parameterCount();
            }
            
            total += activation_->parameterCount();
            total += fc2_->parameterCount();
            
            return total;
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;
            
            auto fc1_params = fc1_->getParameters();
            params.insert( params.end(), fc1_params.begin(), fc1_params.end() );
            
            if (norm_)
            {
                auto norm_params = norm_->getParameters();
                params.insert( params.end(), norm_params.begin(), norm_params.end() );
            }
            
            auto act_params = activation_->getParameters();
            params.insert( params.end(), act_params.begin(), act_params.end() );
            
            auto fc2_params = fc2_->getParameters();
            params.insert( params.end(), fc2_params.begin(), fc2_params.end() );
            
            return params;
        }

        std::vector<ITensor*> getGradients() const override
        {
            std::vector<ITensor*> grads;
            
            auto fc1_grads = fc1_->getGradients();
            grads.insert( grads.end(), fc1_grads.begin(), fc1_grads.end() );
            
            if (norm_)
            {
                auto norm_grads = norm_->getGradients();
                grads.insert( grads.end(), norm_grads.begin(), norm_grads.end() );
            }
            
            auto act_grads = activation_->getGradients();
            grads.insert( grads.end(), act_grads.begin(), act_grads.end() );
            
            auto fc2_grads = fc2_->getGradients();
            grads.insert( grads.end(), fc2_grads.begin(), fc2_grads.end() );
            
            return grads;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "====================" << std::endl;
            oss << "MLP: " << getName() << std::endl;
            oss << "Input features: " << config_.getInputFeatures() << std::endl;
            oss << "Hidden size: " << config_.getHiddenSize() << std::endl;
            oss << "Bias: " << (config_.hasBias() ? "enabled" : "disabled") << std::endl;
            oss << "Activation: " << activationTypeToString( config_.getActivationType() ) << std::endl;
            oss << "Layer Norm: " << (config_.useLayerNorm() ? "enabled" : "disabled") << std::endl;

            if (exec_context_ && exec_context_->getDevice())
            {
                oss << "Device: " << deviceTypeToString( exec_context_->getDevice()->getDeviceType() ) << std::endl;
            }

            oss << "Parameter count: " << parameterCount() << std::endl;

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

                oss << "Hidden shape: (";
                for (size_t i = 0; i < cached_hidden_shape_.size(); ++i)
                {
                    oss << cached_hidden_shape_[i];
                    if (i != cached_hidden_shape_.size() - 1)
                        oss << ", ";
                }
                oss << ")" << std::endl;
            }

            oss << "Sub-Modules:" << std::endl;
            oss << "  - fc1: " << fc1_->getName() << std::endl;
            
            if (norm_)
            {
                oss << "  - norm: " << norm_->getName() << std::endl;
            }
            
            oss << "  - activation: " << activation_->getName() << std::endl;
            oss << "  - fc2: " << fc2_->getName() << std::endl;

            return oss.str();
        }

        // ====================================================================
        // Configuration accessor
        // ====================================================================

        const MLPConfig& getConfig() const noexcept
        {
            return config_;
        }

        // ====================================================================
        // Child module accessors
        // ====================================================================

        std::shared_ptr<LinearType> getFC1() const noexcept
        {
            return fc1_;
        }

        std::shared_ptr<LinearType> getFC2() const noexcept
        {
            return fc2_;
        }

        std::shared_ptr<GeluType> getActivation() const noexcept
        {
            return activation_;
        }

        std::shared_ptr<LayerNormType> getNorm() const noexcept
        {
            return norm_;
        }

    protected:
        ///**
        // * @brief Override buildImpl for sequential shape propagation.
        // *
        // * This is called by the base CompositeModule::build() after validation.
        // * Delegates to the public build() method which handles all setup.
        // */
        //void buildImpl( const shape_t& input_shape ) override
        //{
        //    // Build is already handled by public build() method
        //}

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagate new training mode to child modules. Called with the
         * CompositeModule's training mutex held; do not call setTraining() here.
         */
        void onTrainingChanging( bool is_training ) override
        {
            if (fc1_) fc1_->setTraining( is_training );
            
            if (norm_) norm_->setTraining( is_training );
            
            if (activation_) activation_->setTraining( is_training );
            
            if (fc2_) fc2_->setTraining( is_training );
        }

    private:
        MLPConfig config_;
        bool built_{ false };
        std::shared_ptr<ExecutionContextType> exec_context_;

        shape_t cached_input_shape_;
        shape_t cached_hidden_shape_;

        std::shared_ptr<LinearType> fc1_{ nullptr };
        std::shared_ptr<GeluType> activation_{ nullptr };
        std::shared_ptr<LinearType> fc2_{ nullptr };
        std::shared_ptr<LayerNormType> norm_{ nullptr };

        std::shared_ptr<TensorType> fc1_output_{ nullptr };
        std::shared_ptr<TensorType> norm_output_{ nullptr };
        std::shared_ptr<TensorType> act_output_{ nullptr };

        /**
         * @brief Validate input shape for MLP operation.
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.empty())
            {
                throw std::invalid_argument( "MLP: input must have rank >= 1" );
            }

            int64_t input_features = input_shape.back();

            if (input_features != config_.getInputFeatures())
            {
                std::ostringstream oss;
                oss << "MLP: input feature dimension mismatch. Expected "
                    << config_.getInputFeatures() << ", got " << input_features;
                throw std::invalid_argument( oss.str() );
            }
        }

        /**
         * @brief Create and configure child modules.
         *
         * Ensure children are synchronized with this CompositeModule's training mode.
         */
        void createModules()
        {
            auto fc1_config = LinearConfig( config_.getInputFeatures(), config_.getHiddenSize() );
            fc1_config.withName( config_.getName() + ".fc1" )
                .withBias( config_.hasBias() );

            fc1_ = std::make_shared<LinearType>( exec_context_, fc1_config );
            fc1_->setTraining( this->isTraining() );

            if (config_.useLayerNorm())
            {
                auto norm_config = LayerNormConfig();
                norm_config.withAxis( -1 )
                    .withName( config_.getName() + ".norm" );

                norm_ = std::make_shared<LayerNormType>( exec_context_, norm_config );
                norm_->setTraining( this->isTraining() );
            }

            switch (config_.getActivationType())
            {
                case ActivationType::Gelu:
                {
                    auto gelu_config = GeluConfig();
                    gelu_config.withName( config_.getName() + ".gelu" );

                    activation_ = std::make_shared<GeluType>( exec_context_, gelu_config );
                    activation_->setTraining( this->isTraining() );
                    break;
                }
                default:
                    throw std::invalid_argument( "MLP: unsupported activation type" );
            }

            auto fc2_config = LinearConfig( config_.getHiddenSize(), config_.getInputFeatures() );
            fc2_config.withName( config_.getName() + ".fc2" )
                .withBias( config_.hasBias() );

            fc2_ = std::make_shared<LinearType>( exec_context_, fc2_config );
            fc2_->setTraining( this->isTraining() );
        }
    };
}