/**
 * @file GatedDeltaRule.ixx
 * @brief Gated delta rule -- the linear-attention mixer of Qwen 3.8's DeltaNet layers.
 *
 * The counterpart to GroupedQueryAttention in the other layer kind, and the first mixer in
 * the tree whose memory is a fixed-size RECURRENT STATE rather than a growing cache. Per
 * value head the state is [head_key_dim, head_value_dim] and each step is
 *
 *   S     <- S * exp(g_t)
 *   delta <- (v_t - k_t^T S) * beta_t
 *   S     <- S + k_t (x) delta
 *   out_t <- q_t^T S
 *
 * with q and k L2-normalized and q scaled by 1/sqrt(head_key_dim).
 *
 * THE STATE DOES NOT GROW WITH CONTEXT. That is the whole point of the layer kind: 48 of
 * Qwen 3.8 27B's 64 layers hold one of these instead of a KV cache, so their cost is flat
 * in sequence length. It inverts the usual long-context budget -- see Qwen3.8.md section 3.
 *
 * g AND beta ARE DERIVED HERE, from the raw a/b projections and this component's own A_log
 * and dt_bias parameters (beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)). The
 * alternative -- computing them in the block -- would put a softplus on the public
 * activation enum for one caller and add two launches over [B, T, heads] tensors.
 *
 * RECURRENT FORM ONLY, for now. Prefill runs the same sequential recurrence as decode, so
 * it is O(T) in sequence steps rather than the chunked (UT-transform) formulation the
 * reference uses for long prefills. That is a throughput gap, not a correctness one, and
 * this form is the oracle the chunked one must be validated against when it is built.
 *
 * Inference-only: no backward.
 */

module;
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <format>
#include <stdexcept>
#include <optional>
#include <cstdint>

export module Dnn.Components.GatedDeltaRule;

export import Dnn.Components.GatedDeltaRuleConfig;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.OperationTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Serialization.ModelArchive;
import Serialization.Metadata;
import Serialization.Mode;
import Serialization.Tensor;
import Serialization.SafeTensors;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GatedDeltaRule : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        // The recurrence accumulates in FP32 regardless of activation precision.
        using StateTensorType = Tensor<TensorDataType::FP32, MR>;

        explicit GatedDeltaRule( const std::string& name, const GatedDeltaRuleConfig& config,
            std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "GatedDeltaRule: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~GatedDeltaRule() override = default;

        /**
         * @brief Advance the recurrence over a chunk of tokens.
         *
         * Prefill and decode are the same call: the state carries either way, and unlike an
         * attention layer there is no separate geometry for a single token. `prefill` and
         * `decode` exist as named entry points so a block reads the same as the attention
         * one, and both forward here.
         */
        TensorType& forward( const TensorType& q, const TensorType& k, const TensorType& v,
            const TensorType& a, const TensorType& b )
        {
            auto& output = run( q, k, v, a, b );

            this->publish( ComputePass::Forward, "output", output );

            return output;
        }

        TensorType& prefill( const TensorType& q, const TensorType& k, const TensorType& v,
            const TensorType& a, const TensorType& b, dim_t position_offset )
        {
            if ( position_offset == 0 )
            {
                resetState();
            }

            auto& output = run( q, k, v, a, b );

            this->publish( ComputePass::Prefill, "output", output );

            return output;
        }

        TensorType& decode( const TensorType& q, const TensorType& k, const TensorType& v,
            const TensorType& a, const TensorType& b, dim_t /*position*/ )
        {
            auto& output = run( q, k, v, a, b );

            this->publish( ComputePass::Decode, "output", output );

            return output;
        }

        /// Zero the recurrent state. The next chunk starts a fresh sequence.
        void resetState()
        {
            if ( state_ )
            {
                zero( *state_, this->getExecutionContext() );
            }
        }

        std::vector<std::string> getParameterNames() const override
        {
            return { "A_log", "dt_bias" };
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> parameters;

            if ( A_log_ )
            {
                parameters.push_back( A_log_.get() );
            }

            if ( dt_bias_ )
            {
                parameters.push_back( dt_bias_.get() );
            }

            return parameters;
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        dim_t parameterCount() const override
        {
            dim_t count = 0;

            if ( A_log_ )
            {
                count += A_log_->size();
            }

            if ( dt_bias_ )
            {
                count += dt_bias_->size();
            }

            return count;
        }

        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
        {
            if ( name == "A_log" )
            {
                this->loadParameterFromBlob( "A_log", blob, *A_log_, A_log_->shape() );
            }
            else if ( name == "dt_bias" )
            {
                this->loadParameterFromBlob( "dt_bias", blob, *dt_bias_, dt_bias_->shape() );
            }
            else
            {
                this->loadParameter( name, blob );
            }
        }

        void saveFlatTensors(
            Serialization::SafeTensorsWriter& writer,
            const std::string& prefix,
            Serialization::TensorSavePass pass ) const override
        {
            if ( A_log_ )
            {
                this->saveParameterToWriter( writer, prefix + ".A_log", *A_log_, pass );
            }

            if ( dt_bias_ )
            {
                this->saveParameterToWriter( writer, prefix + ".dt_bias", *dt_bias_, pass );
            }
        }

        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
        {
            SerializationMetadata meta;
            meta.set( "type", "GatedDeltaRule" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() )
                .set( "num_key_heads", static_cast<int64_t>( config_.getNumKeyHeads() ) )
                .set( "num_value_heads", static_cast<int64_t>( config_.getNumValueHeads() ) )
                .set( "head_key_dim", static_cast<int64_t>( config_.getHeadKeyDim() ) )
                .set( "head_value_dim", static_cast<int64_t>( config_.getHeadValueDim() ) );

            archive.writeMetadata( "meta.json", meta );

            if ( A_log_ )
            {
                this->saveParameterToArchive( archive, "A_log", *A_log_ );
            }

            if ( dt_bias_ )
            {
                this->saveParameterToArchive( archive, "dt_bias", *dt_bias_ );
            }
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        void installSharedOutput( std::shared_ptr<TensorType> output )
        {
            if ( this->isBuilt() )
                throw std::logic_error( std::format(
                    "GatedDeltaRule '{}': installSharedOutput must be called before build()",
                    this->getName() ) );

            output_ = std::move( output );
            output_installed_ = true;
        }

        const ComponentType getType() const override
        {
            return ComponentType::GatedDeltaRule;
        }

        std::vector<const ITensor*> getOutputs() const override
        {
            if ( output_ == nullptr )
            {
                return {};
            }

            return { output_.get() };
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "output",
                ComputePassMask{ ComputePass::Forward, ComputePass::Prefill, ComputePass::Decode } } };
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            if ( A_log_ )
            {
                stats.device_parameter_bytes += A_log_->getStorageSize();
            }

            if ( dt_bias_ )
            {
                stats.device_parameter_bytes += dt_bias_->getStorageSize();
            }

            if ( state_ )
            {
                stats.device_state_bytes += state_->getStorageSize();
            }

            if ( output_ && !output_installed_ )
            {
                stats.device_state_bytes += output_->getStorageSize();
            }

            return stats;
        }

        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            const auto& input_shape = context.inputShape();

            validateInputShape( input_shape );

            MemoryStats stats;

            if ( !A_log_ )
            {
                stats.device_parameter_bytes +=
                    2 * storageBytes<TPrecision>( config_.getNumValueHeads() );
            }

            // Flat in context length -- the property the layer kind exists for.
            if ( !state_ )
            {
                stats.device_state_bytes += storageBytes<TensorDataType::FP32>(
                    input_shape[ 0 ] * config_.getStateElementsPerBatch() );
            }

            if ( !output_installed_ && !context.hasInstalledOutput() )
            {
                stats.device_state_bytes += storageBytes<TPrecision>( elementCount( input_shape ) );
            }

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "GatedDeltaRule: " << this->getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Heads: " << config_.getNumKeyHeads() << " key / "
                << config_.getNumValueHeads() << " value (group "
                << config_.getHeadGroupSize() << ")" << std::endl;
            oss << "Head dims: " << config_.getHeadKeyDim() << " key / "
                << config_.getHeadValueDim() << " value" << std::endl;
            oss << "Recurrent state per batch item: "
                << config_.getStateElementsPerBatch() << " floats" << std::endl;

            return oss.str();
        }

    protected:

        void onExecutionContextSet() override
        {
            operation_ = std::make_shared<OpType>( this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Failed to create GatedDeltaRule compute backend operation." );
            }
        }

        void onBuilding( const BuildContext& build_context ) override
        {
            const auto& input_shape = build_context.inputShape();

            validateInputShape( input_shape );

            auto device = this->getExecutionContext()->getDeviceId();
            const dim_t heads = config_.getNumValueHeads();

            if ( !A_log_ )
            {
                A_log_ = std::make_shared<TensorType>(
                    device, shape_t{ heads }, this->getName() + ".A_log" );
                dt_bias_ = std::make_shared<TensorType>(
                    device, shape_t{ heads }, this->getName() + ".dt_bias" );
            }

            if ( build_context.shouldInitializeParameters() )
            {
                zero( *A_log_, this->getExecutionContext() );
                zero( *dt_bias_, this->getExecutionContext() );
            }

            operation_->setParameters( A_log_.get(), dt_bias_.get() );
            operation_->build( build_context );

            batch_ = input_shape[ 0 ];

            state_ = std::make_shared<StateTensorType>(
                device,
                shape_t{ batch_, heads, config_.getHeadKeyDim(), config_.getHeadValueDim() },
                this->getName() + ".state" );
            zero( *state_, this->getExecutionContext() );

            const shape_t output_shape{ batch_, input_shape[ 1 ], config_.getValueWidth() };

            if ( output_installed_ )
            {
                if ( !output_ || output_->size() < elementCount( output_shape ) )
                    throw std::invalid_argument( std::format(
                        "GatedDeltaRule '{}': installed shared output slot is smaller than the "
                        "build shape requires", this->getName() ) );
            }
            else
            {
                output_ = std::make_shared<TensorType>(
                    device, output_shape, this->getName() + ".output" );
            }

            output_view_.emplace( output_->view( output_shape ) );
        }

        void onTrainingModeChanging( TrainingMode /*training_mode*/ ) override
        {
        }

    private:

        /**
         * @brief The recurrence itself, shared by forward, prefill and decode.
         *
         * Separate so each entry point can publish under its own compute pass; the three
         * differ only in state handling and in what they report, not in the work.
         */
        TensorType& run( const TensorType& q, const TensorType& k, const TensorType& v,
            const TensorType& a, const TensorType& b )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GatedDeltaRule: must be built before use." );
            }

            const auto& v_shape = v.shape();

            if ( output_view_->shape() != v_shape )
            {
                output_view_.emplace( output_->view( v_shape ) );
            }

            operation_->forward( q, k, v, a, b, *state_, *output_view_ );

            return *output_view_;
        }

        using OpType = typename OperationTraits<OperationType::GatedDeltaRuleOp, TDeviceType, TPrecision>::type;

        GatedDeltaRuleConfig config_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };

        std::shared_ptr<TensorType> A_log_{ nullptr };
        std::shared_ptr<TensorType> dt_bias_{ nullptr };

        std::shared_ptr<StateTensorType> state_{ nullptr };
        dim_t batch_{ 0 };

        std::shared_ptr<TensorType> output_{ nullptr };
        bool output_installed_{ false };
        std::optional<TensorType> output_view_;

        /**
         * @brief The build context is sized by the VALUE side: [B, T, value_width].
         *
         * That is what the output and the pooled slot are shaped like. q and k arrive
         * narrower (key_width) and are validated per call by the operation.
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 3 )
            {
                throw std::invalid_argument( std::format(
                    "GatedDeltaRule '{}': build shape must be rank 3 [B, T, value_width], got rank {}",
                    this->getName(), input_shape.size() ) );
            }

            if ( input_shape[ 2 ] != config_.getValueWidth() )
            {
                throw std::invalid_argument( std::format(
                    "GatedDeltaRule '{}': value width mismatch -- expected {}, got {}",
                    this->getName(), config_.getValueWidth(), input_shape[ 2 ] ) );
            }
        }
    };
}
