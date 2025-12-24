/**
 * @file Linear.ixx
 * @brief Device-templated Linear (fully connected) component.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>

export module Dnn.Components.Linear;
export import :Config;

import Dnn.Component;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Precision;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContextFactory;
import Compute.IExecutionContext;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;
import Serialization.Mode;
import Serialization.Tensor;
import Serialization.Metadata;
import nlohmann.json;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using json = nlohmann::json;

    /**
     * @brief Linear (fully connected) component.
     *
     * Delegates computation to a device-specific UnaryOperation implementation
     * registered in the OperationRegistry. All Linear instances share an
     * ExecutionContext owned by the parent (Network or test fixture).
     *
     * Ownership model:
     * - Linear NEVER owns ExecutionContext in shared mode
     * - Standalone mode: Linear owns its ExecutionContext (for unit tests)
     * - Context is shared across component hierarchy when part of Network
     *
     * Construction Modes:
     * - **Standalone mode (device_id provided)**: Creates and owns an ExecutionContext
     *   for the specified device. Used in unit tests and standalone component usage.
     * - **Shared mode (device_id not provided)**: Does not create ExecutionContext;
     *   expects parent to provide one via setExecutionContext(). Used when added
     *   to Network via addComponent<Linear>(...).
     *
     * @tparam TDeviceType Compile-time device type (Cpu, Cuda, Metal, Rocm).
     * @tparam TPrecision Compile-time tensor precision (FP32, FP16, BF16, etc.).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Linear final : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * @brief Construct Linear with optional ExecutionContext ownership.
         *
         * Supports two construction modes:
         *
         * **Standalone mode (device_id provided)**:
         * - Creates and owns an ExecutionContext for the specified device.
         * - Registers the owned context with the base Component class via setExecutionContext().
         * - Parameters and backend operation are created in lifecycle hooks.
         * - Use case: Unit tests, standalone component usage.
         *
         * **Shared mode (device_id not provided)**:
         * - Does not create ExecutionContext; expects parent to provide one.
         * - Parent (Network/CompositeComponent) calls setExecutionContext() after construction.
         * - Parameters and operation created when parent sets context.
         * - Use case: Components added to Network via addComponent<Linear>(...).
         *
         * @param config Linear configuration.
         * @param device_id Optional device identifier. If provided, creates owned ExecutionContext
         *                  for standalone mode. If nullopt, expects shared context from parent.
         *
         * @throws std::invalid_argument if config is invalid (via config.validate()).
         * @throws std::invalid_argument if device_id.type does not match TDeviceType.
         * @throws std::runtime_error if ExecutionContext creation fails (standalone mode).
         *
         * @note In standalone mode, setExecutionContext() is called to register the owned
         *       context with the base class, enabling getExecutionContext() and triggering
         *       the onExecutionContextSet() hook for initialization.
         *
         * @example
         * // Standalone mode (owns context)
         * LinearConfig config;
         * config.setInputFeatures(128).setOutputFeatures(64);
         * Linear<DeviceType::Cpu, TensorDataType::FP32> linear(config, Device::Cpu());
         *
         * @example
         * // Shared mode (borrows parent's context)
         * Network<DeviceType::Cpu, TensorDataType::FP32> net(Device::Cpu(), "my_net");
         * net.addComponent<Linear>("fc1", LinearConfig().setInputFeatures(128).setOutputFeatures(64));
         */
        explicit Linear( const std::string& name, const LinearConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "Linear: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );

                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~Linear() override = default;

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - delegates to backend operation.
         *
         * Computes y = x * W^T + b (if bias is enabled).
         */
        void forward( const ITensor& input, ITensor& output )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Linear Component must be built before calling forward." );
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
                throw std::runtime_error( "Linear Component must be built before calling backward." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Linear Component must be in training mode to call backward. Call setTraining(true) first." );
            }

            if ( !weight_grad_ )
            {
                throw std::runtime_error( "Linear Component weight gradients not initialized. This is a bug." );
            }

            if ( config_.hasBias() && !bias_grad_ )
            {
                throw std::runtime_error( "Linear Component bias gradients not initialized. This is a bug." );
            }

            operation_->backward( input, output_grad, input_grad );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

                /**
         * @brief Save component state to a ModelArchive.
         *
         * This method writes relative entries into the archive. Callers are
         * expected to scope the archive (for example "components/<name>/")
         * before invoking `save_()` so leaf implementations only emit
         * component-local paths such as "meta.json" and "tensors/weight".
         *
         * @param archive ModelArchive to write to (scoped by caller)
         * @param mode SerializationMode (currently unused)
         */
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)mode;

            SerializationMetadata meta;
            meta.set( "type", "Linear" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() );

            archive.writeMetadata( "meta.json", meta );

            SerializationMetadata cfg;
            cfg.set( "input_features", config_.getInputFeatures() )
                .set( "output_features", config_.getOutputFeatures() )
                .set( "has_bias", config_.hasBias() );

            archive.writeMetadata( "config.json", cfg );

            if ( weight_ )
            {
                TensorMetadata tmeta;
                tmeta.dtype = weight_->getDataTypeName();
                tmeta.shape = weight_->shape();
                tmeta.byte_size = static_cast<size_t>(weight_->size()) * weight_->elementSize();
                tmeta.layout = "row_major";
                tmeta.byte_order = "little";

                if constexpr ( std::is_same_v<MR, CpuMemoryResource> )
                {
                    const void* data_ptr = weight_->rawData();
                    writeTensorBlob( archive, "tensors/weight", tmeta, data_ptr, tmeta.byte_size );
                }
                else
                {
                    using HostTensorType = Tensor<dtype_t::FP32, CpuMemoryResource>;
                    HostTensorType host_weight( Device::Cpu(), weight_->shape() );

                    copy( *weight_, host_weight );

                    const void* host_ptr = host_weight.rawData();
                    writeTensorBlob( archive, "tensors/weight", tmeta, host_ptr, tmeta.byte_size );
                }
            }

            if ( config_.hasBias() && bias_ )
            {
                TensorMetadata bmeta;
                bmeta.dtype = bias_->getDataTypeName();
                bmeta.shape = bias_->shape();
                bmeta.byte_size = static_cast<size_t>(bias_->size()) * bias_->elementSize();
                bmeta.layout = "row_major";
                bmeta.byte_order = "little";

                if constexpr ( std::is_same_v<MR, CpuMemoryResource> )
                {
                    const void* data_ptr = bias_->rawData();
                    writeTensorBlob( archive, "tensors/bias", bmeta, data_ptr, bmeta.byte_size );
                }
                else
                {
                    using HostTensorType = Tensor<dtype_t::FP32, CpuMemoryResource>;
                    HostTensorType host_bias( Device::Cpu(), bias_->shape() );

                    copy( *bias_, host_bias );

                    const void* host_ptr = host_bias.rawData();
                    writeTensorBlob( archive, "tensors/bias", bmeta, host_ptr, bmeta.byte_size );
                }
            }
        }

        // ====================================================================
        // Parameters and Gradients
        // ====================================================================

        size_t parameterCount() const override
        {
            size_t count = 0;

            if ( weight_ )
                count += weight_->size();

            if ( config_.hasBias() && bias_ )
                count += bias_->size();

            return count;
        }

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
                throw std::runtime_error( "Linear: getGradients called when not in training mode" );
            }

            std::vector<ITensor*> grads;

            if ( weight_grad_ )
                grads.push_back( weight_grad_.get() );

            if ( bias_grad_ )
                grads.push_back( bias_grad_.get() );

            return grads;
        }

        /**
         * @brief Get weight gradient tensor.
         *
         * @return Shared pointer to weight gradient, or nullptr if not in training mode
         */
        std::shared_ptr<TensorType> getWeightGrad() const noexcept
        {
            return weight_grad_;
        }

        /**
         * @brief Get bias gradient tensor.
         *
         * @return Shared pointer to bias gradient, or nullptr if bias disabled or not in training mode
         */
        std::shared_ptr<TensorType> getBiasGrad() const noexcept
        {
            return bias_grad_;
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
            oss << "--------------------" << std::endl;
            oss << "Linear: " << this->getName() << std::endl;
            oss << "Input features: " << config_.getInputFeatures();
            oss << ", Output features: " << config_.getOutputFeatures() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Has Bias: " << (config_.hasBias() ? "Yes" : "No") << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

        // ====================================================================
        // Parameter accessors
        // ====================================================================

        /**
         * @brief Return shared ownership of the weight tensor.
         *
         * @returns Shared pointer to the weight tensor.
         */
        std::shared_ptr<TensorType> getWeight() const noexcept
        {
            return weight_;
        }

        /**
         * @brief Return shared ownership of the bias tensor.
         *
         * @returns Shared pointer to the bias tensor, or nullptr if no bias.
         */
        std::shared_ptr<TensorType> getBias() const noexcept
        {
            return bias_;
        }

        /**
         * @brief Check whether the component has a bias term.
         *
         * @returns True if bias is enabled in the configuration.
         */
        bool hasBias() const noexcept
        {
            return config_.hasBias();
        }

        /**
         * @brief Get the configuration.
         *
         * @returns Reference to the LinearConfig.
         */
        const LinearConfig& getConfig() const noexcept
        {
            return config_;
        }

    protected:

        /**
         * @brief Hook invoked after ExecutionContext is set.
         *
         * Called by Component::setExecutionContext() after the context is
         * registered. Initializes parameters and creates the backend operation.
         *
         * This hook is triggered in two scenarios:
         * - Standalone mode: Immediately in constructor after owned context creation
         * - Shared mode: When parent calls setExecutionContext() after construction
         *
         * @throws std::runtime_error if parameter initialization or operation creation fails.
         */
        void onExecutionContextSet() override
        {
            initializeParameters();
            createOperation();
        }

        /**
         * @brief Build the Component using an input shape.
         *
         * Linear layer parameters are eagerly created in onExecutionContextSet() based
         * on the configuration. This method binds parameters to the backend
         * operation and triggers backend-specific setup.
         *
         * If in training mode, also initializes gradient tensors and binds them
         * to the operation.
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            operation_->setParameters( weight_.get(), bias_.get() );
            operation_->setTraining( this->isTraining() );

            if ( this->isTraining() )
            {
                initializeGradients();
                operation_->setGradients( weight_grad_.get(), bias_grad_.get() );
            }

            operation_->build( input_shape );
        }


        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagate training mode to the backend operation and allocate / free
         * parameter gradient buffers as appropriate. Called with Component's
         * training mutex held; do not call setTraining() here.
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

                if ( weight_grad_ ) zeros( *weight_grad_ );
                if ( bias_grad_ ) zeros( *bias_grad_ );
            }
        }

    private:

        LinearConfig config_;
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        std::shared_ptr<TensorType> weight_grad_{ nullptr };
        std::shared_ptr<TensorType> bias_grad_{ nullptr };

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        /**
         * @brief Validate input shape for linear operation.
         *
         * Ensures the last dimension matches the configured input_features.
         */
        void validateInputShape( const ITensor& input ) const
        {
            const auto& input_shape = input.shape();
            validateInputShape( input_shape );
        }

        /**
         * @brief Validate input shape for linear operation.
         *
         * Ensures the last dimension matches the configured input_features.
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.empty() )
            {
                throw std::invalid_argument( "Linear: input must have rank >= 1" );
            }

            int64_t input_features = input_shape.back();

            if ( input_features != config_.getInputFeatures() )
            {
                throw std::invalid_argument(
                    std::format( "Linear: input feature dimension mismatch. Expected {}, got {}",
                        config_.getInputFeatures(), input_features )
                );
            }
        }

        /**
         * @brief Ensure gradient tensors are allocated with correct shapes.
         */
        void initializeGradients()
        {
            auto device = this->getExecutionContext()->getDeviceId();

            if ( !weight_grad_ )
            {
                weight_grad_ = std::make_shared<TensorType>( device, weight_->shape() );
                weight_grad_->setName( this->getName() + ".weight.grad" );

                zeros( *weight_grad_ );
            }

            if ( config_.hasBias() && !bias_grad_ )
            {
                bias_grad_ = std::make_shared<TensorType>( device, bias_->shape() );
                bias_grad_->setName( this->getName() + ".bias.grad" );

                zeros( *bias_grad_ );
            }
        }

        /**
         * @brief Allocate and initialize weight and optional bias tensors.
         *
         * Called from onExecutionContextSet() hook once device is known.
         * Tensors are created on the execution context device and initialized
         * using Xavier initialization for weights. Bias is zero-initialized.
         */
        void initializeParameters()
        {
            int64_t input_features = config_.getInputFeatures();
            int64_t output_features = config_.getOutputFeatures();

            auto device = this->getExecutionContext()->getDeviceId();

            weight_ = std::make_shared<TensorType>( device, shape_t{ output_features, input_features } );
            weight_->setName( this->getName() + ".weight" );

            xavier<TPrecision, MR>( *weight_, input_features, output_features );

            if ( config_.hasBias() )
            {
                bias_ = std::make_shared<TensorType>( device, shape_t{ output_features } );
                bias_->setName( this->getName() + ".bias" );

                zeros( *bias_ );
            }
        }

        /**
         * @brief Create the backend compute operation.
         *
         * Called from onExecutionContextSet() hook. Uses the shared ExecutionContext
         * to request a device-specific UnaryOperation from the OperationRegistry.
         */
        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "LinearOp",
                    this->getExecutionContext(),
                    config_
                );

            if ( !operation_ )
            {
                throw std::runtime_error( "Failed to create Linear compute backend operation." );
            }
        }
    };
}