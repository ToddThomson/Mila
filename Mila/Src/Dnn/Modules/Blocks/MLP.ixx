/**
 * @file MLP.ixx
 * @brief Multi-Layer Perceptron (MLP) block for neural networks.
 *
 * Refactored to use the new CompositeModule<TDeviceType> and ExecutionContext.
 */

module;
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <stdexcept>

export module Dnn.Blocks.MLP;
export import :Config;

import Dnn.ITensor;
import Dnn.Tensor;
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
     * Device-templated MLP that composes linear / activation / optional layer-norm
     * modules. Uses TensorDataType precision parameter to match other modules (see Gelu).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class MLP : public CompositeModule<TDeviceType>
    {
    public:
        // TODO: The MR here must support any compute backend DeviceType
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using CompositeModuleBase = CompositeModule<TDeviceType>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * Construct with an existing execution context.
         *
         * @param exec_context Shared execution context used to create tensors and child modules.
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
            
            initializeModules();
        }

        ~MLP() override = default;

        // Module interface (ITensor-based): forward/backward/synchronize

        void forward( const ITensor& input, ITensor& output ) override
        {
            const auto& in = dynamic_cast<const TensorType&>(input);
            auto& out = dynamic_cast<TensorType&>(output);

            fc1_->forward( in, fc1_output_ );

            if (config_.useLayerNorm())
            {
                norm1_->forward( fc1_output_, norm1_output_ );
                activation_->forward( norm1_output_, act_output_ );
            }
            else
            {
                activation_->forward( fc1_output_, act_output_ );
            }

            fc2_->forward( act_output_, out );
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
        {
            const auto& in = dynamic_cast<const TensorType&>(input);
            const auto& out_grad = dynamic_cast<const TensorType&>(output_grad);
            auto& in_grad = dynamic_cast<TensorType&>(input_grad);

            TensorType fc2_grad( exec_context_->getDevice(), act_output_.shape() );
            fc2_->backward( act_output_, out_grad, fc2_grad );

            TensorType act_grad( exec_context_->getDevice(), config_.useLayerNorm() ? norm1_output_.shape() : fc1_output_.shape() );
            activation_->backward( config_.useLayerNorm() ? norm1_output_ : fc1_output_, fc2_grad, act_grad );

            if (config_.useLayerNorm())
            {
                TensorType norm1_grad( exec_context_->getDevice(), fc1_output_.shape() );
                norm1_->backward( fc1_output_, act_grad, norm1_grad );
                fc1_->backward( in, norm1_grad, in_grad );
            }
            else
            {
                fc1_->backward( in, act_grad, in_grad );
            }
        }

        void synchronize() override
        {
            if (exec_context_)
            {
                exec_context_->synchronize();
            }

            for (const auto& m : this->getModules())
            {
                m->synchronize();
            }
        }

        size_t parameterCount() const override
        {
            size_t total_parameters = 0;
            for (const auto& module : this->getModules())
            {
                total_parameters += module->parameterCount();
            }
            return total_parameters;
        }

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

        std::string getName() const override
        {
            return "MLP";
		}

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "====================" << std::endl;
            oss << "MLP: " << this->getName() << std::endl;

            const auto& input_shape = config_.getInputShape();
            oss << "Input shape: (";
            for (size_t i = 0; i < input_shape.size(); ++i)
            {
                oss << input_shape[i];
                if (i != input_shape.size() - 1)
                {
                    oss << ",";
                }
            }
            oss << ")" << std::endl;

            oss << "Input features: " << config_.getInputFeatures() << std::endl;
            oss << "Hidden size: " << config_.getHiddenSize() << std::endl;
            oss << "Bias: " << (config_.hasBias() ? "enabled" : "disabled") << std::endl;
            oss << "Activation: " << activationTypeToString( config_.getActivationType() ) << std::endl;

            if (config_.useLayerNorm())
            {
                oss << "Layer Norm: enabled" << std::endl;
            }

            if (exec_context_ && exec_context_->getDevice())
            {
                oss << "Device: " << deviceTypeToString( exec_context_->getDevice()->getDeviceType() ) << std::endl;
            }

            oss << "Parameter count: " << parameterCount() << std::endl;
            oss << "Sub-Modules..." << std::endl;

            for (const auto& [name, module] : this->getNamedModules())
            {
                oss << "  - " << name << ": " << module->toString() << std::endl;
            }

            return oss.str();
        }

    private:
        MLPConfig config_;
        std::shared_ptr<ExecutionContextType> exec_context_{ nullptr };

        // store children as Module<TDeviceType> pointers for uniform handling
        std::shared_ptr<Module<TDeviceType>> fc1_{ nullptr };
        std::shared_ptr<Module<TDeviceType>> activation_{ nullptr };
        std::shared_ptr<Module<TDeviceType>> fc2_{ nullptr };
        std::shared_ptr<Module<TDeviceType>> norm1_{ nullptr };

        TensorType fc1_output_{ exec_context_ ? exec_context_->getDevice() : std::shared_ptr<Compute::ComputeDevice>( nullptr ), std::vector<size_t>{} };
        TensorType norm1_output_{ exec_context_ ? exec_context_->getDevice() : std::shared_ptr<Compute::ComputeDevice>( nullptr ), std::vector<size_t>{} };
        TensorType act_output_{ exec_context_ ? exec_context_->getDevice() : std::shared_ptr<Compute::ComputeDevice>( nullptr ), std::vector<size_t>{} };

        void initializeModules()
        {
            // Clear any previously registered children
            for (const auto& [name, _] : this->getNamedModules())
            {
                this->removeModule( name );
            }

            // fc1
            auto fc1_config = LinearConfig( config_.getInputFeatures(), config_.getHiddenSize() )
                .withName( this->getName() + ".fc1" )
                .withBias( config_.hasBias() )
                .withTraining( this->isTraining() );

            fc1_ = std::static_pointer_cast<Module<TDeviceType>>(
                std::make_shared<Linear<TDeviceType, TPrecision>>( exec_context_, fc1_config )
            );
            this->addModule( "fc1", fc1_ );

            // optional layer norm
            if (config_.useLayerNorm())
            {
                auto norm1_config = LayerNormConfig( config_.getHiddenSize() )
                    .withName( this->getName() + ".norm1" )
                    .withTraining( this->isTraining() );

                norm1_ = std::static_pointer_cast<Module<TDeviceType>>(
                    std::make_shared<LayerNorm<TDeviceType, TPrecision>>( exec_context_, norm1_config )
                );
                
                this->addModule( "norm1", norm1_ );
            }

            // activation
            switch (config_.getActivationType())
            {
                case ActivationType::Gelu:
                {
                    auto gelu_config = GeluConfig()
                        .withName( this->getName() + ".gelu" )
                        .withTraining( this->isTraining() );

                    activation_ = std::static_pointer_cast<Module<TDeviceType>>(
                        std::make_shared<Gelu<TDeviceType, TPrecision>>( exec_context_, gelu_config )
                    );
                    break;
                }
                default:
                    break;
            }

            this->addModule( "activation", activation_ );

            // fc2
            auto fc2_config = LinearConfig( config_.getHiddenSize(), config_.getInputFeatures() )
                .withName( this->getName() + ".fc2" )
                .withBias( config_.hasBias() )
                .withTraining( this->isTraining() );

            fc2_ = std::static_pointer_cast<Module<TDeviceType>>(
                std::make_shared<Linear<TDeviceType, TPrecision>>( exec_context_, fc2_config )
            );
            this->addModule( "fc2", fc2_ );

            // prepare intermediate buffers using execution context's device
            const auto& input_shape = config_.getInputShape();
            std::vector<size_t> hidden_shape = input_shape;
            if (!hidden_shape.empty())
            {
                hidden_shape.back() = config_.getHiddenSize();
            }
            else
            {
                hidden_shape = { config_.getHiddenSize() };
            }

            if (exec_context_ && exec_context_->getDevice())
            {
                auto device = exec_context_->getDevice();
                fc1_output_ = TensorType( device, hidden_shape );
                if (config_.useLayerNorm())
                {
                    norm1_output_ = TensorType( device, hidden_shape );
                }
                act_output_ = TensorType( device, hidden_shape );
            }
        }
    };
}