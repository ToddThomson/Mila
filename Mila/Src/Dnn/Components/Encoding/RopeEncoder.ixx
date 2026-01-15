/**
 * @file RopeEncoder.ixx
 * @brief Rotary positional embedding (RoPE) component.
 *
 * Applies rotary position embeddings to input embeddings. Delegates compute
 * to a device-specific UnaryOperation backend registered as "RopeOp".
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
#include <optional>

export module Dnn.Components.RopeEncoder;
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
     * @brief Device-templated RoPE component.
     *
     * Delegates heavy compute to a UnaryOperation backend (registered as "RopeOp").
     * This component does not own trainable parameters; it owns only forward/backward
     * temporary buffers that mirror the input shape.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class RopeEncoder : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        explicit RopeEncoder( const std::string& name, const RopeConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "Rope: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~RopeEncoder() override = default;

        /**
         * @brief Forward: applies rotary position embedding to `input`.
         *
         * Preconditions:
         *  - component must be built
         *  - backend operation must be initialized
         *  - owned output buffer allocated during build
         */
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Rope module must be built before calling forward." );
            }

            validateInputShape( input.shape() );

            if ( !operation_ )
            {
                throw std::runtime_error( "Rope: operation backend not initialized" );
            }

            if ( !owned_output_ )
            {
                throw std::runtime_error( "Rope: owned output buffer not allocated" );
            }

            operation_->forward( input, *owned_output_ );

            return *owned_output_;
        }

        /**
         * @brief Backward: computes gradients w.r.t. input (if supported by backend).
         *
         * Requires training mode and an allocated owned_input_grad_ buffer.
         */
        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Rope module must be built before calling backward." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Rope must be in training mode to call backward." );
            }

            if ( !operation_ )
            {
                throw std::runtime_error( "Rope: operation backend not initialized" );
            }

            if ( !owned_input_grad_ )
            {
                throw std::runtime_error( "Rope: owned input-grad buffer not allocated" );
            }

            // Zero input gradient buffer before backward pass to avoid accumulation.
            zero( *owned_input_grad_ );

            operation_->backward( input, output_grad, *owned_input_grad_ );

            return *owned_input_grad_;
        }

        void zeroGradients() override
        {
            // RoPE has no learnable parameters; nothing to zero.
        }

        // Serialization placeholder to match project patterns
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)archive;
            (void)mode;
        }

        std::vector<ITensor*> getParameters() const override
        {
            return {}; // RoPE has no parameters
        }

        std::vector<ITensor*> getGradients() const override
        {
            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Rope: getGradients called when not in training mode" );
            }

            return {}; // No parameter gradients
        }

        size_t parameterCount() const override
        {
            return 0;
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
            oss << "Rope: " << this->getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Config: " << config_.toString() << std::endl;
            return oss.str();
        }

    protected:
        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            operation_->build( input_shape );

            auto device = this->getExecutionContext()->getDeviceId();

            owned_output_ = std::make_unique<TensorType>( device, input_shape );
            owned_output_->setName( this->getName() + ".output" );

            owned_input_grad_ = std::make_unique<TensorType>( device, input_shape );
            owned_input_grad_->setName( this->getName() + ".input.grad" );
            zero( *owned_input_grad_ );
        }

        void onTrainingChanging( bool is_training ) override
        {
            operation_->setTraining( is_training );

            if ( !is_training )
            {
                // Clear any internal gradient state in backend
                operation_->clearGradients();
            }
        }

    private:
        RopeConfig config_;
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        std::unique_ptr<TensorType> owned_output_{ nullptr };
        std::unique_ptr<TensorType> owned_input_grad_{ nullptr };

        void validateInputShape( const shape_t& input_shape ) const
        {
            // Minimal validation: expect at least [B, T, C] or [B, T, ...]
            if ( input_shape.size() < 2 )
            {
                throw std::invalid_argument( "Rope: input rank must be >= 2 (batch, seq, ...)" );
            }

            // Additional checks may rely on RopeConfig in the future.
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "RopeOp",
                    this->getExecutionContext(),
                    config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Failed to create Rope compute backend operation." );
            }
        }
    };
}