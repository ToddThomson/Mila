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
     *
     * Construction Modes:
     * - **Standalone mode**: Construct with DeviceId to create and own an ExecutionContext.
     *   The component manages the context lifetime and uses it for operation execution.
     * - **Shared mode**: Construct without DeviceId; parent (Network/CompositeComponent)
     *   provides ExecutionContext via setExecutionContext() after construction.
     *
     * Ownership:
     * - Standalone mode: Component owns its ExecutionContext (stored in owned_exec_context_).
     * - Shared mode: Component borrows ExecutionContext from parent; lifecycle managed externally.
     *
     * Parameter allocation strategy:
     * - If normalized_shape provided at construction: parameters allocated immediately
     * - If axis provided (shape-agnostic): parameters allocated during build() based on input shape
     *
     * @tparam TDeviceType Compile-time device type (Cpu, Cuda, Metal, Rocm).
     * @tparam TPrecision Compile-time tensor precision (FP32, FP16, BF16, etc.).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LayerNorm : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * @brief Construct LayerNorm with optional ExecutionContext ownership.
         *
         * Supports two construction modes:
         *
         * **Standalone mode (device_id provided)**:
         * - Creates and owns an ExecutionContext for the specified device.
         * - Registers the owned context with the base Component class via setExecutionContext().
         * - Backend operation is created immediately in onExecutionContextSet() hook.
         * - Use case: Unit tests, standalone component usage.
         *
         * **Shared mode (device_id not provided)**:
         * - Does not create ExecutionContext; expects parent to provide one.
         * - Parent (Network/CompositeComponent) calls setExecutionContext() after construction.
         * - Backend operation created when parent sets context.
         * - Use case: Components added to Network via addComponent<LayerNorm>(...).
         *
         * @param config LayerNorm configuration (normalized_shape or axis, epsilon, bias).
         * @param device_id Optional device identifier. If provided, creates owned ExecutionContext
         *                  for standalone mode. If nullopt, expects shared context from parent.
         *
         * @throws std::invalid_argument if config is invalid (via config.validate()).
         * @throws std::invalid_argument if device_id.type does not match TDeviceType.
         * @throws std::runtime_error if ExecutionContext creation fails (standalone mode).
         *
         * @note In standalone mode, setExecutionContext() is called to register the owned
         *       context with the base class, enabling getExecutionContext() and triggering
         *       the onExecutionContextSet() hook for operation creation.
         *
         * @example
         * // Standalone mode (owns context)
         * LayerNormConfig config;
         * config.withNormalizedShape({768});
         * LayerNorm<DeviceType::Cpu, TensorDataType::FP32> ln(config, Device::Cpu());
         *
         * @example
         * // Shared mode (borrows parent's context)
         * Network<DeviceType::Cpu, TensorDataType::FP32> net(Device::Cpu(), "my_net");
         * LayerNormConfig config;
         * config.withAxis(-1);
         * net.addComponent<LayerNorm>("ln", config);
         */
        explicit LayerNorm( const LayerNormConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : config_( config )
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

            if ( config_.hasNormalizedShape() )
            {
                initializeParameters();
            }
        }

        ~LayerNorm() override = default;

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - delegates to backend operation.
         *
         * Normalizes input across specified dimensions and applies learned
         * affine transformation.
         */
        void forward( const ITensor& input, ITensor& output )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "LayerNorm module must be built before calling forward." );
            }

            validateInputShape( input );

            operation_->forward( input, output );
        }

        /**
         * @brief Backward pass - delegates to backend operation.
         *
         * Computes gradients with respect to input and parameters.
         */
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
                params.push_back( weight_.get() );

            if ( bias_ )
                params.push_back( bias_.get() );

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
                grads.push_back( weight_grad_.get() );

            if ( bias_grad_ )
                grads.push_back( bias_grad_.get() );

            return grads;
        }

        size_t parameterCount() const override
        {
            size_t count = 0;

            if ( weight_ )
                count += weight_->size();

            if ( config_.hasBias() && bias_ )
                count += bias_->size();

            return count;
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        std::string getName() const override
        {
            return config_.getName();
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
            oss << "--------------------" << std::endl;
            oss << "LayerNorm: " << getName() << std::endl;
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
         * Called by Component::setExecutionContext() after the context is
         * registered. Creates the backend UnaryOperation using the OperationRegistry.
         *
         * This hook is triggered in two scenarios:
         * - Standalone mode: Immediately in constructor after owned context creation
         * - Shared mode: When parent calls setExecutionContext() after construction
         *
         * @throws std::runtime_error if operation creation fails.
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
         *
         * If in training mode, also allocates gradient tensors and binds them
         * to the operation.
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            if ( !config_.hasNormalizedShape() )
            {
                allocateParametersForShape( input_shape );
            }

            operation_->setParameters( weight_.get(), bias_.get() );

            if ( this->isTraining() )
            {
                initializeParameterGradients();
                operation_->setGradients( weight_grad_.get(), bias_grad_.get() );
            }

            operation_->build( input_shape );
        }

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagates training mode to the backend operation and allocates / frees
         * parameter gradient buffers as appropriate. Called with Component's
         * training mutex held; do not call setTraining() here.
         *
         * @param is_training New training mode (true = training, false = eval).
         */
        void onTrainingChanging( bool is_training ) override
        {
            operation_->setTraining( is_training );

            if ( is_training )
            {
                if ( this->isBuilt() )
                {
                    initializeParameterGradients();
                    operation_->setGradients( weight_grad_.get(), config_.hasBias() ? bias_grad_.get() : nullptr );
                }
            }
            else
            {
                operation_->clearGradients();

                weight_grad_.reset();
                bias_grad_.reset();
            }
        }

    private:
        LayerNormConfig config_;
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::unique_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        std::vector<int64_t> outer_shape_;

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        std::shared_ptr<TensorType> weight_grad_{ nullptr };
        std::shared_ptr<TensorType> bias_grad_{ nullptr };

        /**
         * @brief Validate input shape against normalized_shape configuration.
         */
        void validateInputShape( const ITensor& input ) const
        {
            const auto& norm_shape = config_.getNormalizedShape();
            const auto& input_shape = input.shape();

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
         * @brief Validate input shape against normalized_shape configuration.
         */
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
         * @brief Allocate parameters based on input shape (axis mode).
         *
         * Called when config uses axis instead of normalized_shape.
         * Computes parameter size from input shape and allocates weight/bias.
         */
        void allocateParametersForShape( const shape_t& input_shape )
        {
            int64_t channels = 1;

            if ( config_.getAxis().has_value() )
            {
                const dim_t axis = config_.getAxis().value();
                AxisPartition ap = computeAxisPartition( input_shape, axis, "LayerNorm" );

                channels = ap.axis_size;

                outer_shape_.clear();

                if ( ap.normalized_axis > 0 )
                {
                    outer_shape_.insert( outer_shape_.end(),
                        input_shape.begin(),
                        input_shape.begin() + ap.normalized_axis );
                }

                if ( ap.normalized_axis + 1 < static_cast<int64_t>(input_shape.size()) )
                {
                    outer_shape_.insert( outer_shape_.end(),
                        input_shape.begin() + ap.normalized_axis + 1,
                        input_shape.end() );
                }
            }
            else
            {
                const auto& normalized_shape = config_.getNormalizedShape();
                MultiAxisPartition mp = computeNormalizedShapePartition( input_shape, normalized_shape, "LayerNorm" );

                channels = mp.normalized_size;
                outer_shape_ = std::move( mp.outer_shape );
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

        /**
         * @brief Allocate parameters based on normalized_shape configuration.
         *
         * Called when config provides normalized_shape at construction time.
         * Enables eager parameter allocation independent of input shape.
         */
        void initializeParameters()
        {
            if ( config_.getAxis().has_value() )
            {
                return;
            }

            const auto& normalized_shape = config_.getNormalizedShape();

            int64_t channels = 1;

            for ( const auto& dim : normalized_shape )
            {
                channels *= dim;
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

        /**
         * @brief Allocate and zero-initialize parameter gradient tensors.
         *
         * Called when entering training mode or during build if already in training mode.
         */
        void initializeParameterGradients()
        {
            auto device = this->getExecutionContext()->getDeviceId();

            if ( !weight_grad_ && weight_ )
            {
                weight_grad_ = std::make_shared<TensorType>( device, weight_->shape() );
                weight_grad_->setName( this->getName() + ".weight.grad" );
                zeros( *weight_grad_ );
            }

            if ( config_.hasBias() && !bias_grad_ && bias_ )
            {
                bias_grad_ = std::make_shared<TensorType>( device, bias_->shape() );
                bias_grad_->setName( this->getName() + ".bias.grad" );
                zeros( *bias_grad_ );
            }
        }

        /**
         * @brief Create the backend compute operation.
         *
         * Uses the shared ExecutionContext from the base class to request a
         * device-specific UnaryOperation from the OperationRegistry.
         */
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