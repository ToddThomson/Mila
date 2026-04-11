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

        /**
         * @brief Run forward pass and return a reference to the component-owned output tensor.
         *
         * Dispatches to the operation via a cached output view. The view is pre-initialized
         * at build time and rebuilt only when the input shape changes (e.g. prefill vs decode).
         * Zero heap allocation in steady-state.
         *
         * @param input Input tensor bound to the component device.
         * @return Reference to the cached output view.
         */
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "RmsNorm::forward: must be built." );
            }

            const auto& input_shape = input.shape();

            if ( output_view_->shape() != input_shape )
            {
                output_view_.emplace( output_->view( input_shape ) );
            }

            operation_->forward( input, *output_view_ );

            // DEBUG:
            //std::cout << std::format(
            //    "RmsNorm::forward: Output View: " ) << std::endl;
            //std::cout << toHost<TensorDataType::FP32>( *output_view_ ).toString( true ) << std::endl;
            // END DEBUG
            
            return *output_view_;
        }

        /**
         * @brief Run backward pass and return a reference to the component-owned input-gradient tensor.
         *
         * @param input       Original forward input tensor.
         * @param output_grad Gradient with respect to the component output.
         * @return Reference to the component-owned input-gradient tensor.
         */
        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "RmsNorm: must be built before calling backward." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "RmsNorm: must be in training mode to call backward." );
            }

            if ( !operation_ )
            {
                throw std::runtime_error( "RmsNorm: operation backend not initialized." );
            }

            if ( !input_grad_ )
            {
                throw std::runtime_error( "RmsNorm: input-grad buffer not allocated." );
            }

            // Zero input gradient buffer before backward pass to avoid gradient accumulation.
            zero( *input_grad_ );

            operation_->backward( input, output_grad, *input_grad_ );

            return *input_grad_;
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

        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
        {
            if ( name == "weight" )
            {
                this->loadParameterFromBlob( "weight", blob, *weight_, weight_->shape() );
            }
            else if ( name == "bias" )
            {
                if ( !config_.hasBias() )
                {
                    throw std::runtime_error(
                        std::format( "Component '{}' was configured without bias", this->getName() ) );
                }

                this->loadParameterFromBlob( "bias", blob, *bias_, bias_->shape() );
            }
            else
            {
                this->loadParameter( name, blob );
            }
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

            if ( output_ != nullptr )
            {
                stats.device_state_bytes += output_->getStorageSize();
            }

            if ( input_grad_ != nullptr )
            {
                stats.device_gradient_bytes += input_grad_->getStorageSize();
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

        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const BuildContext& build_context ) override
        {
            const auto& input_shape = build_context.inputShape();

            validateInputShape( input_shape );
            allocateParameters( &input_shape );

            operation_->setParameters( weight_.get(), bias_.get() );
            operation_->build( build_context );

            auto device = this->getExecutionContext()->getDeviceId();

            output_ = std::make_unique<TensorType>( device, input_shape, this->getName() + ".output" );
            output_view_.emplace( output_->view( input_shape ) );

            if ( build_context.isTrainingMode() )
            {
                initializeGradients();
                operation_->setGradients( weight_grad_.get(), bias_grad_.get() );

                input_grad_ = std::make_unique<TensorType>( device, input_shape, this->getName() + ".input_grad" );
                zero( *input_grad_ );
            }
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            operation_->setTrainingMode( training_mode );

            auto is_training = (training_mode == TrainingMode::Normal);

            if ( !is_training )
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
        shape_t outer_shape_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        std::shared_ptr<TensorType> weight_grad_{ nullptr };
        std::shared_ptr<TensorType> bias_grad_{ nullptr };

        std::unique_ptr<TensorType> output_{ nullptr };
        std::unique_ptr<TensorType> input_grad_{ nullptr };
        std::optional<TensorType> output_view_;

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

                outer_shape_ = {};
                uint8_t out_ndim = 0;

                for ( uint8_t i = 0; i < input_shape->ndim; ++i )
                {
                    if ( static_cast<int64_t>( i ) != ap.normalized_axis )
                    {
                        outer_shape_.dims[ out_ndim++ ] = (*input_shape)[ i ];
                    }
                }

                outer_shape_.ndim = out_ndim;
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
                    outer_shape_ = shape_t{ mp.outer_shape.data(), mp.outer_shape.data() + mp.outer_shape.size() };
                }
                else
                {
                    for ( const auto& dim : normalized_shape )
                    {
                        channels *= dim;
                    }

                    outer_shape_ = {};
                }
            }

            auto device = this->getExecutionContext()->getDeviceId();

            weight_ = std::make_shared<TensorType>( device, shape_t{ channels }, this->getName() + ".weight" );
            ones( *weight_ );

            if ( config_.hasBias() )
            {
                bias_ = std::make_shared<TensorType>( device, shape_t{ channels }, this->getName() + ".bias" );
                zero( *bias_ );
            }
        }

        void initializeGradients()
        {
            auto device_id = this->getExecutionContext()->getDeviceId();

            if ( !weight_grad_ && weight_ )
            {
                weight_grad_ = std::make_shared<TensorType>( device_id, weight_->shape(), this->getName() + ".weight.grad" );
                zeros( *weight_grad_ );
            }

            if ( config_.hasBias() && !bias_grad_ && bias_ )
            {
                bias_grad_ = std::make_shared<TensorType>( device_id, bias_->shape(), this->getName() + ".bias.grad" );
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