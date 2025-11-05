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
     *   Input ? Linear(in_features, hidden_size) ? [LayerNorm] ? Activation ? Linear(hidden_size, in_features) ? Output
     *
     * Design philosophy:
     * - Two-phase initialization: build() does all setup, forward()/backward() are pure dispatch
     * - Composite module pattern: manages child modules (Linear, activation, LayerNorm)
     * - Shape-agnostic configuration: input_features and hidden_size define architecture
     * - Runtime shape determined at build() time from actual input tensor
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

        /**
         * @brief Construct with an existing execution context.
         *
         * @param exec_context Shared execution context for device resources.
         * @param config MLP configuration.
         */
        explicit MLP( std::shared_ptr<ExecutionContextType> exec_context, const MLPConfig& config )
            : exec_context_( exec_context ), config_( config )
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

            // Compute hidden shape: same leading dims, last dim = hidden_size
            cached_hidden_shape_ = input_shape;
            cached_hidden_shape_.back() = config_.getHiddenSize();

            // Build child modules
            fc1_->build( input_shape );

            if (config_.useLayerNorm())
            {
                norm_->build( cached_hidden_shape_ );
            }

            activation_->build( cached_hidden_shape_ );
            fc2_->build( cached_hidden_shape_ );

            // Allocate intermediate buffers
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
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to child modules.
         *
         * All setup and validation was done in build(). This method chains
         * forward calls through the MLP structure using pre-allocated buffers.
         */
        void forward( const ITensor& input, ITensor& output ) override
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
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
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
            for (const auto& [name, module] : this->getNamedModules())
            {
                oss << "  - " << name << std::endl;
            }

            return oss.str();
        }

        // ====================================================================
        // Configuration accessor
        // ====================================================================

        const MLPConfig& getConfig() const noexcept
        {
            return config_;
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
        MLPConfig config_;
        bool built_{ false };
        std::shared_ptr<ExecutionContextType> exec_context_;

        // Cached shapes determined at build time
        shape_t cached_input_shape_;
        shape_t cached_hidden_shape_;

        // Child modules (stored as Module<TDeviceType> for uniform handling)
        std::shared_ptr<Module<TDeviceType>> fc1_{ nullptr };
        std::shared_ptr<Module<TDeviceType>> activation_{ nullptr };
        std::shared_ptr<Module<TDeviceType>> fc2_{ nullptr };
        std::shared_ptr<Module<TDeviceType>> norm_{ nullptr };

        // Intermediate buffer tensors (allocated at build time)
        std::shared_ptr<TensorType> fc1_output_{ nullptr };
        std::shared_ptr<TensorType> norm_output_{ nullptr };
        std::shared_ptr<TensorType> act_output_{ nullptr };

        /**
         * @brief Validate input shape for MLP operation.
         *
         * Ensures the last dimension matches the configured input_features.
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
         * @brief Create and register child modules.
         *
         * Called from constructor to instantiate all child modules and register
         * them with the composite module base for uniform management.
         */
        void createModules()
        {
            // fc1: input_features ? hidden_size
            auto fc1_config = LinearConfig( config_.getInputFeatures(), config_.getHiddenSize() );
            fc1_config.withName( config_.getName() + ".fc1" )
                .withBias( config_.hasBias() );

            fc1_ = std::make_shared<Linear<TDeviceType, TPrecision>>( exec_context_, fc1_config );
            this->addModule( "fc1", fc1_ );

            // Optional layer norm
            if (config_.useLayerNorm())
            {
                auto norm_config = LayerNormConfig();
                norm_config.withAxis( -1 )
                    .withName( config_.getName() + ".norm" );

                norm_ = std::make_shared<LayerNorm<TDeviceType, TPrecision>>( exec_context_, norm_config );
                this->addModule( "norm", norm_ );
            }

            // Activation
            switch (config_.getActivationType())
            {
                case ActivationType::Gelu:
                {
                    auto gelu_config = GeluConfig();
                    gelu_config.withName( config_.getName() + ".gelu" );

                    activation_ = std::make_shared<Gelu<TDeviceType, TPrecision>>( exec_context_, gelu_config );
                    break;
                }
                default:
                    throw std::invalid_argument( "MLP: unsupported activation type" );
            }

            this->addModule( "activation", activation_ );

            // fc2: hidden_size ? input_features
            auto fc2_config = LinearConfig( config_.getHiddenSize(), config_.getInputFeatures() );
            fc2_config.withName( config_.getName() + ".fc2" )
                .withBias( config_.hasBias() );

            fc2_ = std::make_shared<Linear<TDeviceType, TPrecision>>( exec_context_, fc2_config );
            this->addModule( "fc2", fc2_ );
        }
    };

    // Convenience aliases for common usages
    export template<TensorDataType TPrecision>
        using CpuMLP = MLP<DeviceType::Cpu, TPrecision>;

    export template<TensorDataType TPrecision>
        using CudaMLP = MLP<DeviceType::Cuda, TPrecision>;
}