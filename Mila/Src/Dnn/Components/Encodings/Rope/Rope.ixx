/**
 * @file Rope.ixx
 * @brief Rotary positional embedding (RoPE) component.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <format>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <optional>

export module Dnn.Components.Rope;
export import :Config;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Compute.Precision;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.PairedOperation;
import Compute.IPositionalPairedOp;
import Compute.OperationRegistry;
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
     * Rotates Q and K in-place using a PairedOperation backend registered
     * as "RopeOp". The component owns no forward output buffers — rotation
     * writes directly back into the caller-provided tensors (typically views
     * into a fused QKV projection buffer).
     *
     * Supports three dispatch modes:
     *   forward(Q, K)                  — training, positions 0..T-1
     *   prefill(Q, K, position_offset) — chunked prefill, positions offset..offset+T-1
     *   decode(Q, K, position)         — single-token KV-cache decode
     *
     * Positional dispatch (prefill/decode) is available when the backend
     * implements IPositionalPairedOp. The interface pointer is cached at
     * operation creation via dynamic_cast.
     *
     * This component has no trainable parameters.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Rope : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        explicit Rope( const std::string& name, const RopeConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                    throw std::invalid_argument( "Rope: device type mismatch" );

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~Rope() override = default;

        /**
         * @brief Apply rotary position embeddings to Q and K in-place.
         *
         * Uses implicit positions 0..T-1. Suitable for training forward passes.
         *
         * @param Q  Query tensor [B, T, n_heads    * head_dim]. Mutated in-place.
         * @param K  Key tensor   [B, T, n_kv_heads * head_dim]. Mutated in-place.
         */
        void forward( TensorType& Q, TensorType& K )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "Rope must be built before calling forward()." );

            if ( !operation_ )
                throw std::runtime_error( "Rope: operation backend not initialized." );

            operation_->forward( Q, K, Q, K );
        }

        // ====================================================================
        // Backward
        // ====================================================================

        /**
         * @brief Backpropagate gradients through RoPE.
         *
         * RoPE is an orthogonal rotation (R^T = R^{-1}), so input gradients
         * are the upstream gradients rotated by the transpose (negative) angles.
         *
         * @param grad_Q  Upstream gradient w.r.t. rotated Q.
         * @param grad_K  Upstream gradient w.r.t. rotated K.
         * @return Pair of references: (grad_Q_in, grad_K_in).
         */
        std::pair<TensorType&, TensorType&> backward(
            TensorType& grad_Q,
            TensorType& grad_K )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "Rope must be built before calling backward()." );

            if ( !this->isTraining() )
                throw std::runtime_error( "Rope must be in training mode to call backward()." );

            if ( !operation_ )
                throw std::runtime_error( "Rope: operation backend not initialized." );

            zero( *owned_Q_grad_ );
            zero( *owned_K_grad_ );

            operation_->backward( grad_Q, grad_K, *owned_Q_grad_, *owned_K_grad_ );

            return { *owned_Q_grad_, *owned_K_grad_ };
        }

        // ====================================================================
        // Positional inference (IPositionalPairedOp)
        // ====================================================================

        /**
         * @brief Apply rotary position embeddings with an explicit position offset.
         *
         * Each token at chunk-local position t is rotated using the cos/sin cache
         * row at absolute position (t + position_offset). Required for chunked
         * prefill where successive chunks use increasing offsets.
         *
         * @param Q               Query tensor [B, T, n_heads    * head_dim]. Mutated in-place.
         * @param K               Key tensor   [B, T, n_kv_heads * head_dim]. Mutated in-place.
         * @param position_offset Absolute position of the first token in this chunk.
         */
        void prefill( TensorType& Q, TensorType& K, int position_offset )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "Rope must be built before calling prefill()." );

            if ( !positional_op_ )
                throw std::runtime_error( "Rope: backend does not support positional inference." );

            positional_op_->prefill( Q, K, Q, K, position_offset );
        }

        /**
         * @brief Single-token decode with explicit position.
         *
         * Rotates Q and K in-place using the cos/sin cache row at `position`.
         * Required for KV-cache autoregressive generation where T=1.
         *
         * @param Q        Query tensor [B, 1, n_heads    * head_dim]. Mutated in-place.
         * @param K        Key tensor   [B, 1, n_kv_heads * head_dim]. Mutated in-place.
         * @param position Absolute position of the token in the full sequence.
         */
        void decode( TensorType& Q, TensorType& K, int position )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "Rope must be built before calling decode()." );

            if ( !positional_op_ )
                throw std::runtime_error( "Rope: backend does not support positional inference." );

            positional_op_->decode( Q, K, Q, K, position );
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        void zeroGradients() override
        {}

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)archive; (void)mode;
        }

        std::vector<ITensor*> getParameters() const override
        {
            return {};
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        size_t parameterCount() const override
        {
            return 0;
        }

        const ComponentType getType() const override
        {
            return ComponentType::Rope;
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

            if ( owned_Q_grad_ != nullptr )
            {
                stats.device_gradient_bytes += owned_Q_grad_->getStorageSize();
            }

            if ( owned_K_grad_ != nullptr )
            {
                stats.device_gradient_bytes += owned_K_grad_->getStorageSize();
            }

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Rope: " << this->getName() << "\n";
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << "\n";
            oss << "Config: " << config_.toString() << "\n";

            return oss.str();
        }

    protected:

        void onExecutionContextSet() override
        {
            createOperation();
        }

        void onBuilding( const BuildContext& build_context ) override
        {
            validateBuildContext( build_context );
            const auto& input_shape = build_context.inputShape();

            q_shape_ = shape_t{ input_shape[ 0 ], input_shape[ 1 ], static_cast<dim_t>( config_.getNumHeads() * config_.getHeadDim() ) };
            k_shape_ = shape_t{ input_shape[ 0 ], input_shape[ 1 ], static_cast<dim_t>( config_.getNumKVHeads() * config_.getHeadDim() ) };

            if ( build_context.isTrainingMode() )
            {
                auto device = this->getExecutionContext()->getDeviceId();

                owned_Q_grad_ = std::make_unique<TensorType>( device, q_shape_, this->getName() + ".Q_grad" );
                zero( *owned_Q_grad_ );

                owned_K_grad_ = std::make_unique<TensorType>( device, k_shape_, this->getName() + ".K_grad" );
                zero( *owned_K_grad_ );
            }

            operation_->build( build_context );
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {

        }

    private:
        RopeConfig config_;

        std::shared_ptr<PairedOperation<TDeviceType, TPrecision>> operation_{ nullptr };
        IPositionalPairedOp* positional_op_{ nullptr };

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        shape_t q_shape_;
        shape_t k_shape_;

        std::unique_ptr<TensorType> owned_Q_grad_{ nullptr };
        std::unique_ptr<TensorType> owned_K_grad_{ nullptr };

        void validateBuildContext( const BuildContext& context ) const
        {
            if ( context.inputShape().size() < 2 )
                throw std::invalid_argument( "Rope: leading shape must have at least rank 2 [B, T, ...]" );
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createPairedOperation<TDeviceType, TPrecision>(
                    "RopeOp",
                    this->getExecutionContext(),
                    config_ );

            if ( !operation_ )
                throw std::runtime_error( "Rope: failed to create RopeOp backend." );

            positional_op_ = dynamic_cast<IPositionalPairedOp*>( operation_.get() );
        }
    };
}
