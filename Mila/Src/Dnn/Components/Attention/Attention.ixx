/**
 * @file Attention.ixx
 * @brief Multi-Head Attention module (concatenated QKV input).
 *
 * Module delegates compute to a device-specific UnaryOperation implementation
 * that expects a concatenated QKV input.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <optional>

export module Dnn.Components.Attention;
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
     * @brief Multi-Head Attention module that accepts concatenated QKV input.
     *
     * The module requires a single input tensor in model-layout containing
     * concatenated Q, K and V along the feature axis:
     *   input shape == [B, T, 3 * embedding_dim]
     *
     * The backend compute implementation (registered as "AttentionOp") must
     * accept the concatenated QKV input and produce an output of shape
     *   output shape == [B, T, embedding_dim]
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Attention : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using ComponentBase = Component<TDeviceType, TPrecision>;

        /**
         * @brief Construct Attention component.
         *
         * @param name Component name identifier (mandatory)
         * @param config Attention configuration
         * @param device_id Optional DeviceId to create owned ExecutionContext (standalone mode)
         */
        explicit Attention( const std::string& name, const AttentionConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "Attention: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );

                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~Attention() override = default;

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Run forward pass and return component-owned output tensor.
         *
         * Allocates and reuses an output tensor owned by the component (allocated
         * in onBuilding()). Delegates to backend `operation_->forward`.
         *
         * @param input Concatenated QKV input tensor.
         * @return Reference to component-owned TensorType containing the forward result.
         *
         * @throws std::runtime_error if component not built or backend not initialized.
         */
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Attention module must be built before calling forward." );
            }

            validateConcatenatedQKVShape( input.shape() );

            operation_->forward( input, *owned_output_ );

            return *owned_output_;
        }

        /**
         * @brief Run backward pass and return component-owned input-gradient tensor.
         *
         * Uses an input-gradient tensor owned by the component (allocated in onBuilding())
         * and delegates to backend `operation_->backward`.
         *
         * @param input Concatenated QKV input tensor used by forward.
         * @param output_grad Gradient w.r.t. the module output.
         * @return Reference to component-owned TensorType containing the input gradient.
         *
         * @throws std::runtime_error if component not built, not in training mode, or backend not initialized.
         */
        TensorType& backward(
            const TensorType& input,
            const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Attention must be built before calling backward." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Attention must be in training mode to call backward. Call setTraining(true) first." );
            }

            validateConcatenatedQKVShape( input.shape() );

            // Zero input gradient buffer before backward pass. No exceptions.
            zero( *owned_input_grad_ /*, this->getExecutionContext() */ );

            operation_->backward( input, output_grad, *owned_input_grad_ );

            return *owned_input_grad_;
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
            return {};
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::Attention;
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        size_t parameterCount() const override
        {
            return 0;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Attention: " << this->getName() << std::endl;
            oss << "Device Id: " << this->getExecutionContext()->getDeviceId().toString() << std::endl;
            oss << "Embedding dimension: " << config_.getEmbeddingSize() << std::endl;
            oss << "Number of heads: " << config_.getNumHeads() << std::endl;
            oss << "Head size: " << (config_.getEmbeddingSize() / config_.getNumHeads()) << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

        int64_t getEmbeddingSize() const noexcept
        {
            return config_.getEmbeddingSize();
        }

        int64_t getNumHeads() const noexcept
        {
            return config_.getNumHeads();
        }

        const AttentionConfig& getConfig() const noexcept
        {
            return config_;
        }

    protected:

        // ====================================================================
        // Lifecycle hooks aligned with Component base
        // ====================================================================

        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const shape_t& input_shape ) override
        {
            validateConcatenatedQKVShape( input_shape );

            operation_->setTraining( this->isTraining() );

            operation_->setParameters( nullptr, nullptr );

            operation_->build( input_shape );

            input_shape_ = input_shape;

            // Allocate component-owned forward output and input-grad tensors.
            auto device = this->getExecutionContext()->getDeviceId();

            shape_t out_shape = input_shape_;
            out_shape.back() = config_.getEmbeddingSize();

            owned_output_ = std::make_unique<TensorType>( device, out_shape );
            owned_output_->setName( this->getName() + ".output" );

            owned_input_grad_ = std::make_unique<TensorType>( device, input_shape_ );
            owned_input_grad_->setName( this->getName() + ".input.grad" );
        }

        void onTrainingChanging( bool is_training ) override
        {
            if ( operation_ )
            {
                operation_->setTraining( is_training );
            }
        }

    private:
        AttentionConfig config_;
        shape_t input_shape_;

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        // Component-owned buffers (new API): forward output and input gradient
        std::unique_ptr<TensorType> owned_output_{ nullptr };
        std::unique_ptr<TensorType> owned_input_grad_{ nullptr };

        void validateConcatenatedQKVShape( const shape_t& shape ) const
        {
            if ( shape.size() != 3 )
            {
                throw std::invalid_argument( "Attention: expected 3D model-layout shape" );
            }

            const int64_t trailing = shape.back();
            const int64_t expected = config_.getEmbeddingSize() * 3;

            if ( trailing != expected )
            {
                std::ostringstream oss;
                oss << "Attention: expected concatenated QKV trailing dimension " << expected
                    << " (3 * embedding_dim), got " << trailing;
                throw std::invalid_argument( oss.str() );
            }
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createUnaryOperation<TDeviceType, TPrecision>(
                    "AttentionOp",
                    this->getExecutionContext(),
                    config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Failed to create Attention compute backend operation." );
            }
        }
    };
}