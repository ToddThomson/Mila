/**
 * @file Rope.ixx
 * @brief Rotary positional embedding (RoPE) component.
 *
 * Applies rotary position embeddings to Q and K tensors simultaneously,
 * rotating both in-place via a single paired backend dispatch using the
 * same cos/sin cache.
 *
 * Unlike GPT-2's Lpe (learned positional encoding) which is a true unary
 * operation on the full embedding stream, RoPE is inherently paired: Q and K
 * are rotated in-place using the same cos/sin cache in a single kernel
 * dispatch. The operation is in-place by design — the CUDA kernel reads each
 * float2 pair into a local register before writing back, making src == dst
 * safe with no aliasing hazard.
 *
 * Typical usage in LlamaBlock:
 *
 *   // Zero-copy views into the fused QKV projection output
 *   auto Q = qkv_out.view( {B, T, n_heads    * head_dim}, 0      );
 *   auto K = qkv_out.view( {B, T, n_kv_heads * head_dim}, q_size );
 *
 *   rope_->forward( Q, K );    // rotates Q and K in-place inside qkv_out
 *   attn_->forward( qkv_out ); // GQA sees rotated Q, K and untouched V
 *
 * Q shape: [B, T, n_heads    * head_dim]
 * K shape: [B, T, n_kv_heads * head_dim]
 */

module;
#include <memory>
#include <vector>
#include <string>
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

        explicit Rope(
            const std::string& name,
            const RopeConfig& config,
            std::optional<DeviceId> device_id = std::nullopt )
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
         * Both tensors are rotated in a single backend dispatch using the
         * same cos/sin cache. Writes directly back into the provided tensors.
         *
         * Safe to call with views of the same underlying buffer — the CUDA
         * kernel reads each float2 pair into local registers before writing,
         * so src == dst carries no aliasing hazard.
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

            // FIXME: validateShapes( Q.shape(), K.shape() );

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
         * The backend implements this via negate_sin=true in the kernel.
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
        // Component interface
        // ====================================================================

        void zeroGradients() override
        {} // No learnable parameters.

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
            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Rope: getGradients called when not in training mode." );
            }

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

        /**
         * @brief Allocate RoPE forward path buffers.
         *
         * Derives Q and K tensor shapes from the leading shape { B, T } and
         * RopeConfig. All rotation is in-place — no forward output buffers
         * are allocated.
         *
         * @param leading_shape { B, T } — allocation bounds. Trailing dimensions
         *                      are derived from RopeConfig:
         *                        Q: n_heads    * head_dim
         *                        K: n_kv_heads * head_dim
         */
        void onBuilding( const shape_t& leading_shape ) override
        {
            validateLeadingShape( leading_shape );

            q_shape_ = leading_shape;
            q_shape_.push_back( config_.getNumHeads() * config_.getHeadDim() );

            k_shape_ = leading_shape;
            k_shape_.push_back( config_.getNumKVHeads() * config_.getHeadDim() );

            operation_->build( leading_shape );
        }

        /**
         * @brief Manage gradient buffer allocation and state on training mode transitions.
         *
         * RoPE has no learnable parameters. Gradient buffers for Q and K are
         * allocated once on the first transition to training mode and retained
         * for the component lifetime. They are zeroed on every transition in
         * both directions to prevent stale gradients leaking across mode switches.
         *
         * Transition behavior:
         *   setTraining( true )  — first call:  allocate Q and K gradient buffers, zero them
         *                        — subsequent:  zero existing buffers
         *   setTraining( false ) — zero existing buffers, retain allocations
         *
         * @param is_training True when entering training mode.
         */
        void onTrainingChanging( bool is_training ) override
        {
            if ( owned_Q_grad_ == nullptr )
            {
                auto device = this->getExecutionContext()->getDeviceId();

                owned_Q_grad_ = std::make_unique<TensorType>( device, q_shape_ );
                owned_Q_grad_->setName( this->getName() + ".Q_grad" );

                owned_K_grad_ = std::make_unique<TensorType>( device, k_shape_ );
                owned_K_grad_->setName( this->getName() + ".K_grad" );

                zero( *owned_Q_grad_ );
                zero( *owned_K_grad_ );
            }

            operation_->setTraining( is_training );

            if ( !is_training )
                operation_->clearGradients();
        }

    private:
        RopeConfig config_;
        shape_t q_shape_;
        shape_t k_shape_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<PairedOperation<TDeviceType, TPrecision>> operation_{ nullptr };

        // Backward gradient buffers — allocated once at build, reused each step.
        std::unique_ptr<TensorType> owned_Q_grad_{ nullptr };
        std::unique_ptr<TensorType> owned_K_grad_{ nullptr };

        void validateLeadingShape( const shape_t& leading_shape ) const
        {
            if ( leading_shape.size() < 2 )
                throw std::invalid_argument( "Rope: leading shape rank must be >= 2." );
        }

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createPairedOperation<TDeviceType, TPrecision>(
                    "RopeOp",
                    this->getExecutionContext(),
                    config_ );

            if ( !operation_ )
                throw std::runtime_error( "Rope: failed to create backend operation." );
        }
    };
}
