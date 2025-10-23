/**
 * @file Linear.ixx
 * @brief Device-templated Linear (fully connected) module.
 *
 * Refactored to follow the pattern used by Gelu.ixx:
 * - Uses abstract TensorDataType (TPrecision)
 * - Accepts a shared ExecutionContext<TDeviceType> at construction
 * - Delegates compute to a UnaryOperation<DeviceType, TPrecision> backend
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>

export module Dnn.Modules.Linear;
export import :Config;
//export import :Attributes;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Device-templated Linear (fully connected) module.
     *
     * Delegates computation to a compute device specific implementation
     * registered in the OperationRegistry.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Linear : public Module<TDeviceType>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;

        /**
         * Construct with an existing execution context.
         *
         * @param config Linear configuration.
         * @param exec_context Shared execution context for device resources.
         */
        explicit Linear( std::shared_ptr<ExecutionContextType> exec_context, const LinearConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            initializeParameters();

            if (this->isTraining())
            {
                //initializeParameterGradients();
            }

            createOperation();
        }

        ~Linear() override = default;

        size_t parameterCount() const override
        {
            size_t num_params = weight_ ? weight_->size() : 0;
            if (config_.hasBias() && bias_) num_params += bias_->size();
            return num_params;
        }

        void forward( const ITensor& input, ITensor& output ) override
        {
            operation_->forward( input, parameters_, output, output_state_ );
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
        {
            // Ensure parameter_grads vector exists when training
            /* FIXME: operation_->backward(
                input,
                output_grad,
                parameters_,
                output_state_,
                input_grad,
                parameter_grads_
            );*/
        }

        void synchronize() override
        {
            exec_context_->synchronize();
        }

        std::shared_ptr<TensorType> getWeight()
        {
            return weight_;
        }

        std::optional<std::shared_ptr<TensorType>> getBias()
        {
            return config_.hasBias() ? std::optional{ bias_ } : std::nullopt;
        }

        bool hasBias() const
        {
            return config_.hasBias();
        }

        void save( ModelArchive& zip ) const override
        {
            // No-op placeholder; serialize parameter tensors if needed
        }

        void load( ModelArchive& archive ) override
        {
            // No-op placeholder; deserialize parameter tensors if needed
        }

        void setTraining( bool is_training ) override
        {
            training_mode_ = is_training;
        }

        bool isTraining() const override
        {
            return training_mode_;
        }

        std::string getName() const override
        {
            return config_.getName();
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Linear: " << this->getName() << std::endl;
            oss << "Input features: " << config_.getInputFeatures();
            oss << ", Output features: " << config_.getOutputFeatures() << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceType() ) << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;
            return oss.str();
        }

    private:
        LinearConfig config_;
		bool training_mode_{ false };

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        // Module-visible parameter containers (backends expect ITensor-like parameters)
        std::vector<std::shared_ptr<TensorType>> parameters_;
        // Gradients for parameters (allocated when training)
        std::vector<std::shared_ptr<TensorType>> parameter_grads_;
        // Cached forward-state tensors (if backend requires them)
        OutputState output_state_;

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        void initializeParameters()
        {
            parameters_.clear();

            size_t input_features = config_.getInputFeatures();
            size_t output_features = config_.getOutputFeatures();

            // Construct tensors bound to the execution context's device.
            auto device = exec_context_->getDevice();

            weight_ = std::make_shared<TensorType>( device, std::vector<size_t>{ output_features, input_features } );
            weight_->setName( this->getName() + ".weight" );

            xavier<TPrecision, MR>( *weight_, input_features, output_features );

            parameters_.emplace_back( weight_ );
            //this->parameter_map_["weight"] = weight_;

            if (config_.hasBias())
            {
                bias_ = std::make_shared<TensorType>( device, std::vector<size_t>{ output_features } );
                bias_->setName( this->getName() + ".bias" );
                parameters_.emplace_back( bias_ );
                //this->parameter_map_["bias"] = bias_;
            }
        }

        void initializeParameterGradients()
        {
            parameter_grads_.clear();

            size_t input_features = config_.getInputFeatures();
            size_t output_features = config_.getOutputFeatures();

            auto device = exec_context_->getDevice();

            auto weight_grad = std::make_shared<TensorType>( device, std::vector<size_t>{ output_features, input_features } );
            weight_grad->setName( this->getName() + ".weight_grad" );
            parameter_grads_.push_back( weight_grad );

            if (config_.hasBias())
            {
                auto bias_grad = std::make_shared<TensorType>( device, std::vector<size_t>{ output_features } );
                bias_grad->setName( this->getName() + ".bias_grad" );
                parameter_grads_.emplace_back( bias_grad );
            }
        }

        void createOperation()
        {
            // Create backend operation using the ExecutionContext
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "LinearOp",
                    exec_context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Failed to create Linear compute backend operation." );
            }
        }
    };

    // Convenience aliases
    export template<TensorDataType TPrecision>
        using CpuLinear = Linear<DeviceType::Cpu, TPrecision>;

    export template<TensorDataType TPrecision>
        using CudaLinear = Linear<DeviceType::Cuda, TPrecision>;
}