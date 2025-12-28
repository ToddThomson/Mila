/**
 * @file LayerNorm.ixx
 * @brief Device-templated Layer Normalization module.
 *
 * Delegates compute to a UnaryOperation backend and owns weight/bias parameters.
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

export module Dnn.Components.LayerNorm;
export import :Config;

import Dnn.Component;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorPartitioning;
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
     * @brief Layer Normalization component.
     *
     * Delegates computation to a device-specific UnaryOperation implementation
     * registered in the OperationRegistry. Normalizes input across specified
     * dimensions and applies learned affine transformation (weight and bias).
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
         * The component is constructed with a default implementation name and may
         * be renamed by callers via `setName()` prior to building. Supports both
         * standalone (owns context) and shared (parent provides context) modes.
         */
        explicit LayerNorm( const std::string& name, const LayerNormConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            // REVIEW: All of this is common boilerplate. Consider moving to ComponentBase
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
        // Compute operation dispatch
        // ====================================================================

        void forward( const ITensor& input, ITensor& output )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "LayerNorm module must be built before calling forward." );
            }

            validateInputShape( input.shape() );

            operation_->forward( input, output );
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "LayerNorm module must be built before calling backward." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "LayerNorm must be in training mode to call backward." );
            }

            operation_->backward( input, output_grad, input_grad );
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

        // ====================================================================
        // Component interface
        // ====================================================================

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
        // Lifecycle hooks
        // ====================================================================

        /**
         * @brief Hook invoked after ExecutionContext is set.
         *
         * Create backend operation and allocate eager parameters if normalized_shape
         * was provided but parameters haven't yet been allocated.
         */
        void onExecutionContextSet() override
        {
            createOperation();
        }

        /**
         * @brief Hook invoked during build() to initialize component with input shape.
         *
         * Validates input shape, allocates parameters if needed (axis mode),
         * binds parameters to backend operation, and triggers backend build.
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            allocateParameters( &input_shape );

            operation_->setParameters( weight_.get(), bias_.get() );

            operation_->build( input_shape );
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagates training mode to the backend operation and allocates / frees
         * parameter gradient buffers as appropriate.
         */
        void onTrainingChanging( bool is_training ) override
        {
            // REVIEW: What does this mean for operations?
            operation_->setTraining( is_training );

            if ( is_training )
            {
                initializeGradients();
                
                operation_->setGradients( weight_grad_.get(), config_.hasBias() ? bias_grad_.get() : nullptr );
            }
            else
            {
                operation_->clearGradients();

                // Prefer to keep gradient buffers allocated for next training phase.
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
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        std::vector<int64_t> outer_shape_;

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        std::shared_ptr<TensorType> weight_grad_{ nullptr };
        std::shared_ptr<TensorType> bias_grad_{ nullptr };

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
         * `input_shape` is provided: allocate for axis mode (requires input shape),
         *   or use `computeNormalizedShapePartition` to compute outer_shape when both
         *   normalized_shape and input shape are available.
         *
         * This consolidates previous duplicate routines into a single authoritative allocator.
         */
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
                AxisPartition ap = computeAxisPartition( *input_shape, axis, "LayerNorm" );

                channels = ap.axis_size;

                outer_shape_.clear();

                if ( ap.normalized_axis > 0 )
                {
                    outer_shape_.insert( outer_shape_.end(),
                        input_shape->begin(),
                        input_shape->begin() + ap.normalized_axis );
                }

                if ( ap.normalized_axis + 1 < static_cast<int64_t>( input_shape->size() ) )
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
                    throw std::invalid_argument( "LayerNorm: cannot allocate parameters without normalized_shape or axis" );
                }

                if ( input_shape )
                {
                    MultiAxisPartition mp = computeNormalizedShapePartition( *input_shape, normalized_shape, "LayerNorm" );

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

            weight_ = std::make_shared<TensorType>( device, shape_t{ channels } );
            weight_->setName( this->getName() + ".weight" );

            if ( config_.hasBias() )
            {
                bias_ = std::make_shared<TensorType>( device, shape_t{ channels } );
                bias_->setName( this->getName() + ".bias" );
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