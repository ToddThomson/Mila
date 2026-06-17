/**
 * @file LayerNorm.ixx
 * @brief Layer Normalization component.
 *
 * Normalizes inputs across specified dimensions and applies learned affine
 * transformation (weight and bias). Delegates compute to a device-specific
 * UnaryOperation backend.
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
#include <numeric>
#include <algorithm>

export module Dnn.Components.LayerNorm;
export import Dnn.Components.LayerNormConfig;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorPartitioning;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.OperationTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Serialization.ModelArchive;
import Serialization.Mode;
import Serialization.Tensor;

// DEBUG:
import Dnn.TensorOps;
import Dnn.TensorHelpers;
import Logging.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Device-templated Layer Normalization component.
     *
     * Provides forward and backward APIs that operate on concrete Tensor types.
     * Delegates heavy compute to a UnaryOperation backend. Parameters (weight/bias)
     * and parameter gradients are owned by the component.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LayerNorm : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * @brief Construct LayerNorm with optional ExecutionContext ownership.
         *
         * @param name Component name (used for tensor names).
         * @param config LayerNorm configuration (normalized_shape, axis, epsilon, bias).
         * @param device_id If provided, component creates and owns an ExecutionContext
         *                  bound to this device; otherwise a parent must supply one
         *                  before building.
         *
         * @throws std::invalid_argument if provided device_id type does not match template.
         */
        explicit LayerNorm( const std::string& name, const LayerNormConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "LayerNorm: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );

                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~LayerNorm() override = default;

        /**
         * @brief Run forward pass and return a reference to the component-owned output tensor.
         *
         * The returned reference refers to a Tensor owned by this component. The
         * backend `operation_->forward` will write into the provided output tensor.
         *
         * Preconditions:
         *  - Component must be built.
         *  - Backend operation must be initialized.
         *  - Component-owned output buffer must be allocated (done during build).
         *
         * @param input Input Tensor bound to the component device.
         * @return Reference to the component-owned output Tensor.
         *
         * @throws std::runtime_error on precondition violations.
         */
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "LayerNorm::forward: must be built." );
            }

            operation_->forward( input, *output_ );

            const auto& input_shape = input.shape();

            if ( input_shape == output_->shape() )
            {
                return *output_;
            }

            if ( !output_view_.has_value() || output_view_->shape() != input_shape )
            {
                output_view_.emplace( output_->view( input_shape ) );
            }

            return *output_view_;
        }

        /**
         * @brief Run backward pass and return a reference to the component-owned input-gradient tensor.
         *
         * The returned reference refers to a Tensor owned by this component. The
         * backend `operation_->backward` will write/accumulate into the provided
         * input-gradient tensor.
         *
         * Preconditions:
         *  - Component must be built and in training mode.
         *  - Backend operation must be initialized.
         *  - Component-owned input-gradient buffer must be allocated (done during build or on training start).
         *
         * @param input Original forward input tensor (device-bound).
         * @param output_grad Gradient with respect to the component output (device-bound).
         * @return Reference to the component-owned input-gradient Tensor.
         *
         * @throws std::runtime_error on precondition violations.
         */
        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "LayerNorm module must be built before calling backward." );
            }

            if ( this->isInferenceMode() )
            {
                throw std::runtime_error( "LayerNorm::backward: must be in training mode" );
            }

            if ( !input_grad_ )
            {
                throw std::runtime_error( "LayerNorm: owned input-grad buffer not allocated" );
            }

            // Zero input gradient buffer before backward pass. No exeptions.
            // Backend ops use accumulation (atomicAdd/+=) which requires pre-zeroed buffers
            // to prevent gradient buildup across calls. Without this, gradients grow linearly
            // with each call -> explosion.
            zero( *input_grad_ );

            operation_->backward( input, output_grad, *input_grad_ );

            return *input_grad_;
        }

        void zeroGradients() override
        {
            if ( weight_grad_ )
            {
                zero( *weight_grad_ /*, this->getExecutionContext() */);
            }

            if ( config_.hasBias() && bias_grad_ )
            {
                zero( *bias_grad_ /*, this->getExecutionContext()*/ );
            }
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)archive;
            (void)mode;
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

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
            // Gradient buffers exist only when built for training; inference
            // yields an empty vector (see Component::getGradients contract).
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
                        std::format( "Component '{}' was configured without bias", this->getName() )
                    );
                }

                this->loadParameterFromBlob( "bias", blob, *bias_, bias_->shape() );
            }
            else
            {
                // Throw by default for unknown parameter names
                this->loadParameter( name, blob );
            }
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::LayerNorm;
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
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
            oss << "LayerNorm: " << this->getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Epsilon: " << config_.getEpsilon() << std::endl;
            oss << "Has Bias: " << (config_.hasBias() ? "Yes" : "No") << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

    protected:

        // ====================================================================
        // Component Lifecycle hooks
        // ====================================================================

        /**
         * @brief Hook invoked after ExecutionContext is set.
         *
         * Creates the backend operation and performs any eager parameter allocation
         * if normalized_shape was supplied at construction time.
         */
        void onExecutionContextSet() override
        {
            createOperation();
        }

        /**
         * @brief Hook invoked during build() to initialize component with input shape.
         *
         * Validates input shape, allocates parameters if needed, binds parameters
         * to the backend operation, triggers backend build, and allocates the
         * component-owned forward output and input-gradient tensors.
         */
        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const auto& input_shape = context.inputShape();

            initializeParameters( input_shape );

            if ( context.shouldInitializeParameters() )
            {
                fill( *weight_, 1.0f, this->getExecutionContext() );

                if ( bias_ )
                {
                    zero( *bias_, this->getExecutionContext() );
                }
            }

            operation_->setParameters( weight_.get(), bias_.get() );
            operation_->build( context );

            auto device = this->getExecutionContext()->getDeviceId();

            /**
             * Output buffer — allocated at the full input shape.
             *
             * LayerNorm is a general component with no knowledge of sequence
             * dimensions or inference decode paths. The parent Network or
             * Transformer is responsible for passing the correct input shape
             * via BuildContext:
             *
             *   Training   — full sequence shape e.g. [B, T, features]
             *   Inference  — decode shape e.g. [1, 1, features] for decode path
             *                or prefill shape e.g. [1, T_chunk, features] for prefill
             *
             * In all cases LayerNorm simply allocates at inputShape() — no
             * special casing for inference or sequence dimensions.
             */
            output_ = std::make_unique<TensorType>( device, input_shape, this->getName() + ".output" );

            if ( context.isTrainingMode() )
            {
                initializeGradients();
                operation_->setGradients( weight_grad_.get(), bias_grad_.get() );

                input_grad_ = std::make_unique<TensorType>( device, input_shape, this->getName() + ".input_grad" );
                zero( *input_grad_ );
            }
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagates training state to the backend operation and allocates or
         * clears parameter gradient buffers as appropriate.
         */
        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            operation_->setTrainingMode( training_mode );

            auto is_eval_mode = (training_mode == TrainingMode::Eval);

            if ( is_eval_mode )
            {
                operation_->clearGradients();

                if ( weight_grad_ )
                {
                    zero( *weight_grad_, this->getExecutionContext() );
                }

                if ( bias_grad_ )
                {
                    zero( *bias_grad_, this->getExecutionContext() );
                }
            }
        }

    private:
        using OpType = typename Compute::OperationTraits<OperationType::LayerNormOp, TDeviceType, TPrecision>::type;

        LayerNormConfig config_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        std::shared_ptr<TensorType> weight_grad_{ nullptr };
        std::shared_ptr<TensorType> bias_grad_{ nullptr };

        // Component-owned forward output and input-gradient tensors (new API).
        // Stored as unique_ptr since the component exclusively owns these buffers.
        std::unique_ptr<TensorType> output_{ nullptr };
        std::unique_ptr<TensorType> input_grad_{ nullptr };
        std::optional<TensorType> output_view_;

        void validateBuildContext( const BuildContext& context ) const
        {
            const auto& input_shape = context.inputShape();

            if ( input_shape.empty() )
            {
                throw std::invalid_argument( "LayerNorm: input shape must not be empty" );
            }

            if ( config_.hasNormalizedShape() )
            {
                const auto& normalized_shape = config_.getNormalizedShape();

                if ( input_shape.size() < normalized_shape.size() )
                {
                    throw std::invalid_argument( std::format(
                        "LayerNorm: input rank {} is less than normalized_shape rank {}",
                        input_shape.size(), normalized_shape.size() ) );
                }

                size_t offset = input_shape.size() - normalized_shape.size();
                for ( size_t i = 0; i < normalized_shape.size(); ++i )
                {
                    if ( input_shape[ offset + i ] != normalized_shape[ i ] )
                    {
                        throw std::invalid_argument( std::format(
                            "LayerNorm: input trailing dim {} is {} but normalized_shape expects {}",
                            i, input_shape[ offset + i ], normalized_shape[ i ] ) );
                    }
                }
            }
            else if ( config_.getAxis().has_value() )
            {
                int64_t axis = config_.getAxis().value();
                int64_t rank = static_cast<int64_t>(input_shape.size());

                // Resolve negative axis
                if ( axis < 0 )
                {
                    axis = rank + axis;
                }

                if ( axis < 0 || axis >= rank )
                {
                    throw std::invalid_argument( std::format(
                        "LayerNorm: axis {} out of range for input rank {}",
                        config_.getAxis().value(), rank ) );
                }
            }
        }

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

        /**
         * @brief Single parameter allocation routine.
         *
         * If `input_shape` is provided the allocator will compute channel count and
         * outer_shape for axis-mode or normalized-shape-mode. If only normalized_shape
         * is available, channels are computed from that shape and outer_shape is left empty.
         *
         * @param input_shape Optional pointer to the build-time input shape.
         */
        void initializeParameters( const shape_t& input_shape )
        {
            const dim_t normalized_features = computeNormalizedFeatureCount( input_shape );

            auto device = this->getExecutionContext()->getDeviceId();

            weight_ = std::make_shared<TensorType>( device, shape_t{ normalized_features }, this->getName() + ".weight" );

            if ( config_.hasBias() )
            {
                bias_ = std::make_shared<TensorType>( device, shape_t{ normalized_features }, this->getName() + ".bias" );
            }
        }

        void initializeGradients()
        {
            auto device_id = this->getExecutionContext()->getDeviceId();

            if ( !weight_grad_ && weight_ )
            {
                weight_grad_ = std::make_shared<TensorType>( device_id, weight_->shape(), this->getName() + ".weight_grad" );
                zero( *weight_grad_, this->getExecutionContext() );
            }

            if ( config_.hasBias() && !bias_grad_ && bias_ )
            {
                bias_grad_ = std::make_shared<TensorType>( device_id, bias_->shape(), this->getName() + ".bias_grad" );
                zero( *bias_grad_, this->getExecutionContext() );
            }
        }

        dim_t computeNormalizedFeatureCount( const shape_t& input_shape ) const
        {
            if ( config_.getAxis().has_value() )
            {
                const dim_t axis = config_.getAxis().value();
                AxisPartition axis_partition = computeAxisPartition( input_shape, axis, "LayerNorm" );

                return axis_partition.axis_size;
            }

            const auto& normalized_shape = config_.getNormalizedShape();

            if ( normalized_shape.empty() )
            {
                throw std::invalid_argument( "LayerNorm: cannot allocate parameters without normalized_shape or axis" );
            }

            MultiAxisPartition normalized_partition =
                computeNormalizedShapePartition( input_shape, normalized_shape, "LayerNorm" );

            return normalized_partition.normalized_size;
        }

        void createOperation()
        {
            operation_ = std::make_shared<OpType>( this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Failed to create LayerNorm compute backend operation." );
            }
        }
    };
}