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

// DEBUG:
import Dnn.TensorOps;
import Dnn.TensorHelpers;
import Utils.Logger;

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

        // ====================================================================
        // Compute operation dispatch (new API)
        // ====================================================================

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
                throw std::runtime_error( "LayerNorm module must be built before calling forward." );
            }

            //// DEBUG:

            //// Check input range
            //auto host_input = toHost<TensorDataType::FP32>( input );
            //auto host_input_ptr = host_input.data();
            //const size_t n = host_input.size();
            //auto [min_in, max_in] = std::minmax_element( host_input_ptr, host_input_ptr + n );
            //Utils::Logger::debug( std::format( "LayerNorm {} in:[{:.3f}, {:.3f}] with shape {}",
            //    this->getName(), *min_in, *max_in, shapeToString( input.shape() ) ) );

            //// END DEBUG:

            operation_->forward( input, *owned_output_ );

            // Resolve the output: return the full buffer if the shape matches,
            // otherwise produce a view trimmed to the actual input shape.
            auto input_shape = input.shape();

            TensorType* result = nullptr;

            if ( input_shape == max_input_shape_ )
            {
                result = owned_output_.get();
            }
            else
            {
                current_output_view_ = std::make_unique<TensorType>(
                    owned_output_->view( input_shape )
                );
                result = current_output_view_.get();
            }

            //// DEBUG:

            //// Check output range on the resolved view to avoid reading stale
            //// elements beyond the current input extent in the pre-allocated buffer.
            //auto host_output = toHost<TensorDataType::FP32>( *result );
            //auto host_output_ptr = host_output.data();
            //const size_t m = host_output.size();
            //auto [min_out, max_out] = std::minmax_element( host_output_ptr, host_output_ptr + m );
            //Utils::Logger::debug( std::format( "LayerNorm {} out:[{:.3f}, {:.3f}] with shape {}",
            //    this->getName(), *min_out, *max_out, shapeToString( result->shape() ) ) );

            //// END DEBUG:

            return *result;
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

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "LayerNorm must be in training mode to call backward." );
            }

            if ( !operation_ )
            {
                throw std::runtime_error( "LayerNorm: operation backend not initialized" );
            }

            if ( !owned_input_grad_ )
            {
                throw std::runtime_error( "LayerNorm: owned input-grad buffer not allocated" );
            }

            // Zero input gradient buffer before backward pass. No exeptions.
            // Backend ops use accumulation (atomicAdd/+=) which requires pre-zeroed buffers
            // to prevent gradient buildup across calls. Without this, gradients grow linearly
            // with each call -> explosion.
            zero( *owned_input_grad_ );

            operation_->backward( input, output_grad, *owned_input_grad_ );

            return *owned_input_grad_;
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
            if ( !this->isTraining() )
            {
                throw std::runtime_error( "LayerNorm: getGradients called when not in training mode" );
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

        void loadParameter( const std::string& name, const TensorBlob& blob ) override
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

            // DEBUG: Diagnostics

            if ( name == "weight" )
            {
                // DIAGNOSTIC: Check weight statistics
                auto host_weights = toHost<TensorDataType::FP32>( *weight_ );

                const float* ptr = host_weights.data();
                const size_t n = host_weights.size();

                if ( n > 0 )
                {
                    float min_w = *std::min_element( ptr, ptr + n );
                    float max_w = *std::max_element( ptr, ptr + n );
                    float mean_w = std::accumulate( ptr, ptr + n, 0.0f ) / static_cast<float>(n);

                    Utils::Logger::info( std::format(
                        "LayerNormOp weight stats: min={:.6f} max={:.6f} mean={:.6f}",
                        min_w, max_w, mean_w ) );
                }

            }

            if ( name == "bias" )
            {
                auto host_bias = toHost<TensorDataType::FP32>( *bias_ );
                const float* ptr = host_bias.data();
                const size_t n = host_bias.size();

                if ( n > 0 )
                {
                    float min_b = *std::min_element( ptr, ptr + n );
                    float max_b = *std::max_element( ptr, ptr + n );

                    Utils::Logger::info( std::format(
                        "LayerNormOp bias stats: min={:.6f} max={:.6f}",
                        min_b, max_b ) );
                }
            }

            // END DEBUG:
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
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            max_input_shape_ = input_shape;

            allocateParameters( input_shape );

            operation_->setParameters( weight_.get(), bias_.get() );

            operation_->build( input_shape );

            auto device = this->getExecutionContext()->getDeviceId();

            owned_output_ = std::make_unique<TensorType>( device, input_shape );
            owned_output_->setName( this->getName() + ".output" );

            owned_input_grad_ = std::make_unique<TensorType>( device, input_shape );
            owned_input_grad_->setName( this->getName() + ".input.grad" );
            zero( *owned_input_grad_ );
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagates training state to the backend operation and allocates or
         * clears parameter gradient buffers as appropriate.
         */
        void onTrainingChanging( bool is_training ) override
        {
            operation_->setTraining( is_training );

            if ( is_training )
            {
                initializeGradients();

                operation_->setGradients( weight_grad_.get(), bias_grad_.get() );
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
        LayerNormConfig config_;
        shape_t max_input_shape_;
        
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        std::shared_ptr<TensorType> weight_grad_{ nullptr };
        std::shared_ptr<TensorType> bias_grad_{ nullptr };

        // Component-owned forward output and input-gradient tensors (new API).
        // Stored as unique_ptr since the component exclusively owns these buffers.
        std::unique_ptr<TensorType> owned_output_{ nullptr };
        std::unique_ptr<TensorType> owned_input_grad_{ nullptr };
        std::unique_ptr<TensorType> current_output_view_{ nullptr };

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
        void allocateParameters( const shape_t& input_shape )
        {
            if ( weight_ )
            {
                return;
            }

            const dim_t normalized_features = computeNormalizedFeatureCount( input_shape );

            auto device = this->getExecutionContext()->getDeviceId();

            weight_ = std::make_shared<TensorType>( device, shape_t{ normalized_features } );
            weight_->setName( this->getName() + ".weight" );
            ones( *weight_ );

            if ( config_.hasBias() )
            {
                bias_ = std::make_shared<TensorType>( device, shape_t{ normalized_features } );
                bias_->setName( this->getName() + ".bias" );
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
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "LayerNormOp",
                    this->getExecutionContext(),
                    config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Failed to create LayerNorm compute backend operation." );
            }
        }
    };
}