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
import Compute.KVCacheable;
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

            auto input_shape = input.shape();

            if ( input_shape == max_input_shape_ )
            {
                return *owned_output_;
            }

            auto output_shape = input_shape;
            output_shape.back() = config_.getModelDim();
            output_view_ = std::make_unique<TensorType>( owned_output_->view( output_shape ) );

            return *output_view_;
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
        // KV Caching
        // ====================================================================
        void initializeKVCache( int64_t max_seq_len )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Attention must be built before initializeKVCache()." );
            }

            auto* kv_cacheable = dynamic_cast<IKVCacheable*>(operation_.get());

            if ( !kv_cacheable )
            {
                throw std::runtime_error( "Attention: KV cache is not supported by this backend." );
            }

            kv_cacheable->initializeKVCache( static_cast<int>(max_input_shape_[ 0 ]), static_cast<int>(max_seq_len) );
        }

        void resetKVCache()
        {
            auto* kv_cacheable = dynamic_cast<IKVCacheable*>(operation_.get());

            if ( !kv_cacheable )
            {
                throw std::runtime_error( "Attention: KV cache is not supported by this backend." );
            }

            kv_cacheable->resetKVCache();
        }

        TensorType& forwardPrefill( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Attention module must be built before calling forwardPrefill()." );
            }

            validateConcatenatedQKVShape( input.shape() );

            auto* kv_cacheable = dynamic_cast<IKVCacheable*>(operation_.get());

            if ( !kv_cacheable )
            {
                throw std::runtime_error( "Attention: KV cache is not supported by this backend." );
            }

            kv_cacheable->forwardPrefill( input, *owned_output_ );

            auto input_shape = input.shape();

            if ( input_shape == max_input_shape_ )
            {
                return *owned_output_;
            }

            auto output_shape = input_shape;
            output_shape.back() = config_.getModelDim();
            output_view_ = std::make_unique<TensorType>( owned_output_->view( output_shape ) );

            return *output_view_;
        }

        TensorType& forwardDecode( const TensorType& input, int position )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Attention module must be built before calling forwardDecode()." );
            }

            validateConcatenatedQKVShape( input.shape() );

            auto* kv_cacheable = dynamic_cast<IKVCacheable*>(operation_.get());

            if ( !kv_cacheable )
            {
                throw std::runtime_error( "Attention: KV cache is not supported by this backend." );
            }

            kv_cacheable->forwardDecode( input, *owned_decode_output_, position );

            return *owned_decode_output_;
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
            oss << "Model dimension: " << config_.getModelDim() << std::endl;
            oss << "Number of heads: " << config_.getNumHeads() << std::endl;
            oss << "Head size: " << (config_.getModelDim() / config_.getNumHeads()) << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

        int64_t getModelDim() const noexcept
        {
            return config_.getModelDim();
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

            max_input_shape_ = input_shape;

            auto device = this->getExecutionContext()->getDeviceId();

            shape_t out_shape = max_input_shape_;
            out_shape.back() = config_.getModelDim();

            owned_output_ = std::make_unique<TensorType>( device, out_shape );
            owned_output_->setName( this->getName() + ".output" );

            owned_input_grad_ = std::make_unique<TensorType>( device, max_input_shape_ );
            owned_input_grad_->setName( this->getName() + ".input.grad" );

            shape_t decode_output_shape = { max_input_shape_[ 0 ], 1, config_.getModelDim() };
            owned_decode_output_ = std::make_unique<TensorType>( device, decode_output_shape );
            owned_decode_output_->setName( this->getName() + ".output.decode" );
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
        shape_t max_input_shape_;

        std::shared_ptr<UnaryOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        std::unique_ptr<TensorType> owned_output_{ nullptr };
        std::unique_ptr<TensorType> output_view_{ nullptr };
        std::unique_ptr<TensorType> owned_input_grad_{ nullptr };
        std::unique_ptr<TensorType> owned_decode_output_{ nullptr };

        void validateConcatenatedQKVShape( const shape_t& shape ) const
        {
            if ( shape.size() != 3 )
            {
                throw std::invalid_argument( "Attention: expected 3D model-layout shape" );
            }

            const int64_t trailing = shape.back();
            const int64_t expected = config_.getModelDim() * 3;

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