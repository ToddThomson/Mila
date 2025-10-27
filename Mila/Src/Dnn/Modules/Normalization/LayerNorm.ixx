/**
 * @file LayerNorm.ixx
 * @brief Device-templated Layer Normalization module.
 *
 * Refactored to follow the Gelu module pattern:
 * - Use abstract TensorDataType (TPrecision)
 * - Accept a shared ExecutionContext<TDeviceType> at construction
 * - Delegate compute to UnaryOperation<DeviceType, TPrecision> backend
 */

module;
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>
#include <cstdint>
#include <stdexcept>

export module Dnn.Modules.LayerNorm;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorHostTypeMap;
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
     * @brief Layer Normalization module (device-templated).
     *
     * Delegates computation to a device-specific UnaryOperation implementation
     * registered in the OperationRegistry.
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LayerNorm : public Module<TDeviceType>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;
        using Parameters = std::vector<std::shared_ptr<TensorType>>;
        using OutputState = std::vector<std::shared_ptr<TensorType>>;
        using HostType = typename TensorHostTypeMap<TPrecision>::host_type;

        /**
         * Construct with an existing execution context.
         *
         * @param config LayerNorm configuration.
         * @param exec_context Shared execution context for device resources.
         */
        explicit LayerNorm( std::shared_ptr<ExecutionContextType> exec_context, const LayerNormConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();
            initializeTensors();
            createOperation();
        }

        ~LayerNorm() override = default;

        size_t parameterCount() const override
        {
            size_t count = 0;
            if (weight_) count += weight_->size();
            if (config_.hasBias() && bias_) count += bias_->size();
            return count;
        }

        void forward( const ITensor& input, ITensor& output ) override
        {
            operation_->forward( input, parameters_, output, output_state_ );
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) override
        {
            
            // prepare parameter gradients containers
            //std::vector<std::shared_ptr<ITensor>> parameter_grads;
            //parameter_grads.resize( parameters_.size() );
            //for (size_t i = 0; i < parameters_.size(); ++i)
            //{
            //    // allocate gradient tensor with same shape as parameter
            //    auto p = std::static_pointer_cast<TensorType>( parameters_[i] );
            //    parameter_grads[i] = std::make_shared<TensorType>( p->shape() );
            //}

            // delegate to backend
            // FIXME: operation_->backward(
            //    input,
            //    output_grad,
            //    parameters_,
            //    // cast parameter_grads to proper type for backend signature
            //    reinterpret_cast<std::vector<std::shared_ptr<TensorType>>&>(parameter_grads),
            //    input_grad,
            //    output_state_
            //);
        }

        void save( ModelArchive& /*archive*/ ) const override
        {
            // No-op: stateless activation
        }

        void load( ModelArchive& /*archive*/ ) override
        {
            // No-op: stateless activation
        }

        std::string getName() const override
        {
            return config_.getName();
        }

        void synchronize() override
        {
            exec_context_->synchronize();
        }

        void setTraining( bool is_training ) override
        {
            training_mode_ = is_training;
        }

        bool isTraining() const override
        {
            return training_mode_;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "LayerNorm: " << getName() << std::endl;
            oss << "Axis: " << config_.getAxis() << std::endl;
            const auto& shape = config_.getInputShape();
            oss << "Input shape: (";
            for (size_t i = 0; i < shape.size(); ++i)
            {
                oss << shape[i];
                if (i + 1 < shape.size()) oss << ",";
            }
            oss << ")" << std::endl;
            oss << "Epsilon: " << config_.getEpsilon() << std::endl;
            oss << "Has Bias: " << (config_.hasBias() ? "Yes" : "No") << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;
            return oss.str();
        }

    private:
        LayerNormConfig config_;
		bool training_mode_{ false };

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };
        std::shared_ptr<TensorType> mean_{ nullptr };
        std::shared_ptr<TensorType> rstd_{ nullptr };

        std::vector<std::shared_ptr<TensorType>> parameters_;
        OutputState output_state_;

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        void initializeTensors()
        {
            parameters_.clear();
            output_state_.clear();
            //this->parameter_map_.clear();
            //this->state_map_.clear();

            const auto& input_shape = config_.getInputShape();
            if (input_shape.size() < 3)
            {
                throw std::invalid_argument( "Input shape must have at least 3 dimensions [batch, seq_len, features]" );
            }

            size_t batch_size = input_shape[0];
            size_t seq_len = input_shape[1];
            size_t channels = input_shape[2];

            // Construct tensors bound to the execution context's device.
            auto device = exec_context_->getDevice();

            // weight defaults to ones
            weight_ = std::make_shared<TensorType>( device, std::vector<size_t>{ channels } );
            weight_->setName( this->getName() + ".weight" );

            if (config_.hasBias())
            {
                bias_ = std::make_shared<TensorType>( device, std::vector<size_t>{ channels } );
                bias_->setName( this->getName() + ".bias" );
            }

            mean_ = std::make_shared<TensorType>( device, std::vector<size_t>{ batch_size, seq_len } );
            mean_->setName( this->getName() + ".mean" );

            rstd_ = std::make_shared<TensorType>( device, std::vector<size_t>{ batch_size, seq_len } );
            rstd_->setName( this->getName() + ".rstd" );

            parameters_.emplace_back( weight_ );
            //this->parameter_map_["weight"] = weight_;
            
            if (config_.hasBias())
            {
                parameters_.emplace_back( bias_ );
                //this->parameter_map_["bias"] = bias_;
            }

            output_state_.emplace_back( mean_ );
            output_state_.emplace_back( rstd_ );
            //this->state_map_["mean"] = mean_;
            //this->state_map_["rstd"] = rstd_;
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "LayerNormOp",
                    exec_context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Failed to create LayerNorm compute backend operation." );
            }
        }
    };
}