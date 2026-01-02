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
     *
     * Construction modes:
     * - Standalone: provide DeviceId and component will create and own an ExecutionContext.
     * - Deferred/shared: omit DeviceId and caller must call setExecutionContext() before build().
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
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

        void forward( const ITensor& input, ITensor& output )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Attention module must be built before calling forward." );
            }

            validateForwardShapes( input, output );

            operation_->forward( input, output );
        }

        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Attention must be built before calling backward." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Attention must be in training mode to call backward. Call setTraining(true) first." );
            }

            validateBackwardShapes( input, output_grad, input_grad );

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
            return {};
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
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
            oss << "Embedding dimension: " << config_.getEmbeddingDim() << std::endl;
            oss << "Number of heads: " << config_.getNumHeads() << std::endl;
            oss << "Head size: " << (config_.getEmbeddingDim() / config_.getNumHeads()) << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

        int64_t getEmbeddingDim() const noexcept
        {
            return config_.getEmbeddingDim();
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

        void validateConcatenatedQKVShape( const shape_t& shape ) const
        {
            if ( shape.size() != 3 )
            {
                throw std::invalid_argument( "Attention: expected 3D model-layout shape" );
            }

            const int64_t trailing = shape.back();
            const int64_t expected = config_.getEmbeddingDim() * 3;

            if ( trailing != expected )
            {
                std::ostringstream oss;
                oss << "Attention: expected concatenated QKV trailing dimension " << expected
                    << " (3 * embedding_dim), got " << trailing;
                throw std::invalid_argument( oss.str() );
            }
        }

        void validateForwardShapes( const ITensor& input, const ITensor& output ) const
        {
            const auto& in_shape = input.shape();
            const auto& out_shape = output.shape();

            validateConcatenatedQKVShape( in_shape );

            if ( out_shape.size() != 3 )
            {
                throw std::invalid_argument( "Attention: output must be 3D model-layout [B, T, embedding_dim]" );
            }

            const int64_t out_trailing = out_shape.back();
            const int64_t expected_out = config_.getEmbeddingDim();

            if ( out_trailing != expected_out )
            {
                std::ostringstream oss;
                oss << "Attention: expected output trailing dimension " << expected_out
                    << " (embedding_dim), got " << out_trailing;
                throw std::invalid_argument( oss.str() );
            }

            if ( in_shape[ 0 ] != out_shape[ 0 ] || in_shape[ 1 ] != out_shape[ 1 ] )
            {
                throw std::invalid_argument( "Attention: input and output batch/sequence dimensions must match" );
            }
        }

        void validateBackwardShapes(
            const ITensor& input,
            const ITensor& output_grad,
            const ITensor& input_grad ) const
        {
            const auto& in_shape = input.shape();
            validateConcatenatedQKVShape( in_shape );

            const auto& outg_shape = output_grad.shape();
            if ( outg_shape.size() != 3 || outg_shape.back() != config_.getEmbeddingDim() )
            {
                throw std::invalid_argument( "Attention: output_grad must have model-layout trailing dim == embedding_dim" );
            }

            const auto& ing_shape = input_grad.shape();
            if ( ing_shape.size() != 3 || ing_shape.back() != config_.getEmbeddingDim() * 3 )
            {
                throw std::invalid_argument( "Attention: input_grad must have model-layout trailing dim == 3 * embedding_dim" );
            }

            if ( in_shape[ 0 ] != outg_shape[ 0 ] || in_shape[ 1 ] != outg_shape[ 1 ] ||
                in_shape[ 0 ] != ing_shape[ 0 ] || in_shape[ 1 ] != ing_shape[ 1 ] )
            {
                throw std::invalid_argument( "Attention: batch/sequence dimensions must match across input, output_grad and input_grad" );
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