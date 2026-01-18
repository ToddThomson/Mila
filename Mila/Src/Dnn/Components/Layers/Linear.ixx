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
import Dnn.ComponentType;
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

// DEBUG:
import Dnn.TensorHelpers;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using json = nlohmann::json;

    /**
     * @brief Device-templated fully connected (linear) component.
     *
     * Delegates compute to a device-specific UnaryOperation implementation.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Linear : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * @brief Construct a Linear component.
         *
         * Constructs with a name and configuration. If `device_id` is provided,
         * the component creates and owns an ExecutionContext (standalone mode)
         * and registers it with the base Component via setExecutionContext().
         * If `device_id` is not provided, the component expects a shared
         * ExecutionContext to be provided later via setExecutionContext().
         *
         * @param name Component name.
         * @param config Layer configuration (validated on construction).
         * @param device_id Optional device identifier. When present the component
         *                  creates an owned ExecutionContext for the device.
         *
         * @throws std::invalid_argument if config is invalid or device type mismatches.
         * @throws std::runtime_error if ExecutionContext creation fails.
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
         * @brief Perform forward pass.
         *
         * Uses a component-owned output buffer (allocated in onBuilding()) and
         * delegates computation to the backend operation.
         *
         * @param input Input tensor (device-bound).
         * @return Reference to the component-owned output tensor.
         *
         * @throws std::runtime_error if component not built, backend not initialized,
         *         or owned output buffer not allocated.
         */
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Linear Component must be built before calling forward." );
            }

            validateInputShape( input.shape() );

            operation_->forward( input, *owned_output_ );

            return *owned_output_;
        }

        /**
         * @brief Perform backward pass.
         *
         * Uses a component-owned input-gradient buffer (allocated in onBuilding())
         * and delegates computation to the backend operation.
         *
         * @param input Original forward input tensor.
         * @param output_grad Gradient with respect to the component output.
         * @return Reference to the component-owned input-gradient tensor.
         *
         * @throws std::runtime_error if component not built, not in training mode,
         *         or backend not initialized.
         */
        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Linear Component must be built before calling backward." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Linear Component must be in training mode to call backward. Call setTraining(true) first." );
            }

            // Zero input gradient buffer before backward pass. No exeptions.
            // Backend ops use accumulation (atomicAdd/+=) which requires pre-zeroed buffers
            // to prevent gradient buildup across calls. Without this, gradients grow linearly
            // with each call -> explosion.
            zero( *owned_input_grad_ );

            // DEBUG: Dump W and B and input
            //debugDumpTensor<TPrecision,MR>( *weight_, "weight" );

            if ( bias_ )
            {
                //debugDumpTensor<TPrecision,MR>( *bias_, "bias" );
            }

            //debugDumpTensor<TPrecision,MR>( input, "input" );
            //debugDumpTensor<TPrecision,MR>( output_grad, "output_grad" );

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

            // REVIEW: Not strictly necessary, but zero input gradients for safety during testing
            if ( owned_input_grad_ )
            {
                zero( *owned_input_grad_ );
            }
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Save component state to a ModelArchive.
         *
         * Writes component-local metadata and parameter tensors into the provided archive.
         * Callers should scope the archive before invoking this method.
         *
         * @param archive ModelArchive to write to (scoped by caller).
         * @param mode Serialization mode (currently unused).
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

        const ComponentType getType() const override
        {
            return ComponentType::Linear;
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
            oss << "Linear: " << this->getName() << std::endl;
            oss << "Input features: " << config_.getInputFeatures();
            oss << ", Output features: " << config_.getOutputFeatures() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Has Bias: " << (config_.hasBias() ? "Yes" : "No") << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

        /**
         * @brief Whether the component contains a bias parameter.
         *
         * @return True if bias is enabled in the configuration.
         */
        bool hasBias() const noexcept
        {
            return config_.hasBias();
        }

        /**
         * @brief Access the component configuration.
         *
         * @return Reference to the LinearConfig.
         */
        const LinearConfig& getConfig() const noexcept
        {
            return config_;
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
                throw std::runtime_error( "Linear: getGradients called when not in training mode" );
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

    protected:

        /**
         * @brief Lifecycle hook executed after ExecutionContext is set.
         *
         * Initializes parameter tensors and creates the backend operation.
         *
         * @throws std::runtime_error if initialization fails.
         */
        void onExecutionContextSet() override
        {
            initializeParameters();

            createOperation();
        }

        /**
         * @brief Build the component for a given input shape.
         *
         * Binds parameters and gradients (if training) to the backend operation,
         * invokes operation-specific build, and allocates component-owned forward
         * and input-gradient tensors. Owned tensors use std::unique_ptr to express
         * exclusive ownership by the component.
         *
         * @param input_shape Shape of the incoming tensor.
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            operation_->setParameters( weight_.get(), bias_.get() );
            operation_->setTraining( this->isTraining() );

            // REVIEW: training is never set before build in current Component API.
            if ( this->isTraining() )
            {
                initializeGradients();
                operation_->setGradients( weight_grad_.get(), bias_grad_.get() );
            }

            operation_->build( input_shape );

            // Allocate and cache component-owned output and input-gradient tensors.
            auto device_id = this->getExecutionContext()->getDeviceId();

            shape_t output_shape = input_shape;
            if ( !output_shape.empty() )
            {
                output_shape.back() = config_.getOutputFeatures();
            }

            owned_output_ = std::make_unique<TensorType>( device_id, output_shape );
            owned_output_->setName( this->getName() + ".output" );

            owned_input_grad_ = std::make_unique<TensorType>( device_id, input_shape );
            owned_input_grad_->setName( this->getName() + ".input.grad" );
            zero( *owned_input_grad_ );
        }

        /**
         * @brief Hook invoked when training mode is changing.
         *
         * Propagates training mode to the backend operation and allocates or
         * clears gradient buffers as necessary. Called with the Component's
         * training mutex held.
         *
         * @param is_training New training mode.
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

                // Prefer to keep gradient buffers allocated for next training phase.
                if ( weight_grad_ )
                {
                    zero( *weight_grad_ );
                }
                
                if ( bias_grad_ )
                {
                    zero( *bias_grad_ );
                }
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

        // Component-owned forward output and input-gradient tensors (exclusive ownership)
        std::unique_ptr<TensorType> owned_output_{ nullptr };
        std::unique_ptr<TensorType> owned_input_grad_{ nullptr };

        /**
         * @brief Validate input shape for the linear operation.
         *
         * Ensures the last dimension matches the configured input_features.
         *
         * @param input_shape Shape to validate.
         * @throws std::invalid_argument if rank < 1 or feature dim mismatches.
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
         *
         * Allocates weight and bias gradient tensors when needed and zeroes them.
         */
        void initializeGradients()
        {
            auto device_id = this->getExecutionContext()->getDeviceId();

            if ( !weight_grad_ )
            {
                weight_grad_ = std::make_shared<TensorType>( device_id, weight_->shape() );
                weight_grad_->setName( this->getName() + ".weight.grad" );

                zero( *weight_grad_ );
            }

            if ( config_.hasBias() && !bias_grad_ )
            {
                bias_grad_ = std::make_shared<TensorType>( device_id, bias_->shape() );
                bias_grad_->setName( this->getName() + ".bias.grad" );

                zero( *bias_grad_ );
            }
        }

        /**
         * @brief Allocate and initialize weight and optional bias tensors.
         *
         * Tensors are created on the ExecutionContext device and weights are
         * initialized using Xavier initialization. Bias is zero-initialized.
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

                zero( *bias_ );
            }
        }

        /**
         * @brief Create the backend compute operation.
         *
         * Requests a device-specific UnaryOperation from the OperationRegistry.
         *
         * @throws std::runtime_error if operation creation fails.
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
                throw std::runtime_error( "Failed to create Linear operation." );
            }
        }
    };
}