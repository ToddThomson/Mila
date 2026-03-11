/**
 * @file RmsNorm.ixx
 * @brief RMS Normalization component.
 *
 * Normalizes inputs using root-mean-square across the normalized dimensions and
 * applies a learned affine transform (weight and optional bias). Follows the
 * lifecycle and API patterns used by LayerNorm.
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
#include <mutex>
#include <utility>
#include <optional>

export module Dnn.Components.RmsNorm;

export import :Config;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorPartitioning;
import Dnn.TensorInitializers;
import Compute.Precision;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Device-templated RMS Normalization component.
     *
     * Delegates heavy compute to a UnaryOperation backend (registered as "RmsNormOp").
     * Parameters (weight/bias) and parameter gradients are owned by the component.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class RmsNorm : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        explicit RmsNorm( const std::string& name, const RmsNormConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "RmsNorm: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~RmsNorm() override = default;

        // Forward pass returns component-owned output tensor reference.
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "RmsNorm module must be built before calling forward." );
            }

            validateInputShape( input.shape() );

            if ( !operation_ )
            {
                throw std::runtime_error( "RmsNorm: operation backend not initialized" );
            }

            if ( !owned_output_ )
            {
                throw std::runtime_error( "RmsNorm: owned output buffer not allocated" );
            }

            operation_->forward( input, *owned_output_ );

            return *owned_output_;
        }

        // Backward pass returns component-owned input-gradient tensor reference.
        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "RmsNorm module must be built before calling backward." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "RmsNorm must be in training mode to call backward." );
            }

            if ( !operation_ )
            {
                throw std::runtime_error( "RmsNorm: operation backend not initialized" );
            }

            if ( !owned_input_grad_ )
            {
                throw std::runtime_error( "RmsNorm: owned input-grad buffer not allocated" );
            }

            // Zero input gradient buffer before backward pass to avoid gradient accumulation.
            zero( *owned_input_grad_ );

            operation_->backward( input, output_grad, *owned_input_grad_ );

            return *owned_input_grad_;
        }

        void zeroGradients() override
        {
            if ( weight_grad_ )
            {
                zero( *weight_grad_ );
            }

            if ( config_.hasBias() && bias_grad_ )
            {
                zero( *bias_grad_ );
            }
        }

        // Serialization (placeholder to match LayerNorm pattern)
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)archive;
            (void)mode;
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;

            if ( weight_ )
            {
                params.push_back( weight_.get() );
            }

            if ( bias_ )
            {
                params.push_back( bias_.get() );
            }

            return params;
        }

        std::vector<ITensor*> getGradients() const override
        {
            if ( !this->isTraining() )
            {
                throw std::runtime_error( "RmsNorm: getGradients called when not in training mode" );
            }

            std::vector<ITensor*> grads;

            if ( weight_grad_ )
            {
                grads.push_back( weight_grad_.get() );
            }

            if ( bias_grad_ )
            {
                grads.push_back( bias_grad_.get() );
            }

            return grads;
        }

        size_t parameterCount() const override
        {
            size_t count = 0;

            if ( weight_ )
            {
                count += weight_->size();
            }

            if ( config_.hasBias() && bias_ )
            {
                count += bias_->size();
            }

            return count;
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        const ComponentType getType() const override
        {
            return ComponentType::RmsNorm;
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            if ( weight_ != nullptr )
            {
                stats.device_parameter_bytes += weight_->getStorageSize();
            }

            if ( bias_ != nullptr )
            {
                stats.device_parameter_bytes += bias_->getStorageSize();
            }

            if ( owned_output_ != nullptr )
            {
                stats.device_state_bytes += owned_output_->getStorageSize();
            }

            if ( owned_input_grad_ != nullptr )
            {
                stats.device_gradient_bytes += owned_input_grad_->getStorageSize();
            }

            if ( weight_grad_ != nullptr )
            {
                stats.device_gradient_bytes += weight_grad_->getStorageSize();
            }

            if ( bias_grad_ != nullptr )
            {
                stats.device_gradient_bytes += bias_grad_->getStorageSize();
            }

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "RmsNorm: " << this->getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Epsilon: " << config_.getEpsilon() << std::endl;
            oss << "Has Bias: " << (config_.hasBias() ? "Yes" : "No") << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

    protected:
        // Lifecycle hooks follow LayerNorm patterns.

        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            allocateParameters( &input_shape );

            operation_->setParameters( weight_.get(), bias_.get() );

            operation_->build( input_shape );

            auto device = this->getExecutionContext()->getDeviceId();

            owned_output_ = std::make_unique<TensorType>( device, input_shape, this->getName() + ".output" );
            //owned_output_->setName( this->getName() + ".output" );

            //owned_input_grad_ = std::make_unique<TensorType>( device, input_shape, this->getName() + ".input.grad" );
            //owned_input_grad_->setName( this->getName() + ".input.grad" );
            //zero( *owned_input_grad_ );
        }

        void onTrainingChanging( bool is_training ) override
        {
            operation_->setTraining( is_training );

            if ( is_training )
            {
                if ( !weight_grad_ || (config_.hasBias() && !bias_grad_) )
                 {
                    // First transition to training mode — allocate gradient buffers.
                    initializeGradients();
                    operation_->setGradients( weight_grad_.get(), bias_grad_.get() );
                }

                if ( !owned_input_grad_ )
                {
                    auto device = this->getExecutionContext()->getDeviceId();
                    owned_input_grad_ = std::make_unique<TensorType>( device, owned_output_->shape(), this->getName() + ".input.grad" );
                    //owned_input_grad_->setName( this->getName() + ".input.grad" );
                    zero( *owned_input_grad_ );
                }
            }
            else
            {
                operation_->clearGradients();

                if ( weight_grad_ )
                {
                    zeros( *weight_grad_ );
                }

                if ( bias_grad_ )
                {
                    zeros( *bias_grad_ );
                }
            }
        }

    private:
        RmsNormConfig config_;
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        std::vector<int64_t> outer_shape_;

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        std::shared_ptr<TensorType> weight_grad_{ nullptr };
        std::shared_ptr<TensorType> bias_grad_{ nullptr };

        std::unique_ptr<TensorType> owned_output_{ nullptr };
        std::unique_ptr<TensorType> owned_input_grad_{ nullptr };

        void validateInputShape( const shape_t& input_shape ) const
        {
            const auto& norm_shape = config_.getNormalizedShape();

            if ( input_shape.size() < norm_shape.size() )
            {
                throw std::invalid_argument( "Input rank must be >= normalized_shape rank" );
            }

            size_t offset = input_shape.size() - norm_shape.size();

            for ( size_t i = 0; i < norm_shape.size(); ++i )
            {
                if ( input_shape[ offset + i ] != norm_shape[ i ] )
                {
                    throw std::invalid_argument( "Input trailing dimensions don't match normalized_shape" );
                }
            }
        }

        void allocateParameters( const shape_t* input_shape )
        {
            if ( weight_ )
            {
                return;
            }

            int64_t channels = 1;

            if ( config_.getAxis().has_value() )
            {
                const dim_t axis = config_.getAxis().value();
                AxisPartition ap = computeAxisPartition( *input_shape, axis, "RmsNorm" );

                channels = ap.axis_size;

                outer_shape_.clear();

                if ( ap.normalized_axis > 0 )
                {
                    outer_shape_.insert( outer_shape_.end(),
                        input_shape->begin(),
                        input_shape->begin() + ap.normalized_axis );
                }

                if ( ap.normalized_axis + 1 < static_cast<int64_t>(input_shape->size()) )
                {
                    outer_shape_.insert( outer_shape_.end(),
                        input_shape->begin() + ap.normalized_axis + 1,
                        input_shape->end() );
                }
            }
            else
            {
                const auto& normalized_shape = config_.getNormalizedShape();

                if ( normalized_shape.empty() )
                {
                    throw std::invalid_argument( "RmsNorm: cannot allocate parameters without normalized_shape or axis" );
                }

                if ( input_shape )
                {
                    MultiAxisPartition mp = computeNormalizedShapePartition( *input_shape, normalized_shape, "RmsNorm" );

                    channels = mp.normalized_size;
                    outer_shape_ = std::move( mp.outer_shape );
                }
                else
                {
                    for ( const auto& dim : normalized_shape )
                    {
                        channels *= dim;
                    }

                    outer_shape_.clear();
                }
            }

            auto device = this->getExecutionContext()->getDeviceId();

            weight_ = std::make_shared<TensorType>( device, shape_t{ channels }, this->getName() + ".weight" );
            //weight_->setName( this->getName() + ".weight" );
            ones( *weight_ );

            if ( config_.hasBias() )
            {
                bias_ = std::make_shared<TensorType>( device, shape_t{ channels }, this->getName() + ".bias" );
                //bias_->setName( this->getName() + ".bias" );
                zero( *bias_ );
            }
        }

        void initializeGradients()
        {
            auto device_id = this->getExecutionContext()->getDeviceId();

            if ( !weight_grad_ && weight_ )
            {
                weight_grad_ = std::make_shared<TensorType>( device_id, weight_->shape() );
                weight_grad_->setName( this->getName() + ".weight.grad" );
                zeros( *weight_grad_ );
            }

            if ( config_.hasBias() && !bias_grad_ && bias_ )
            {
                bias_grad_ = std::make_shared<TensorType>( device_id, bias_->shape() );
                bias_grad_->setName( this->getName() + ".bias.grad" );
                zeros( *bias_grad_ );
            }
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "RmsNormOp",
                    this->getExecutionContext(),
                    config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Failed to create RmsNorm compute backend operation." );
            }
        }
    };
}