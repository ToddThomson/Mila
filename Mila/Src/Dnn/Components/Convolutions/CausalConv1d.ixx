/**
 * @file CausalConv1d.ixx
 * @brief Depthwise causal 1-D convolution over the sequence axis.
 *
 * One length-K filter per channel, no cross-channel mixing:
 *
 *   out[b, t, c] = bias[c] + sum(i = 0 .. K-1) weight[c, i] * x[b, t - (K-1) + i, c]
 *
 * The first component in the tree that carries a convolution, and the first whose memory
 * is a rolling window rather than a growing cache. Qwen 3.8's Gated DeltaNet layers run it
 * over the 10240-wide fused q/k/v stream with K = 4 (`linear_conv_kernel_dim`).
 *
 * PREFILL AND DECODE, like the attention components -- and for the same reason. A causal
 * convolution needs the K-1 positions to its left, which during decode are in the past
 * rather than in the input. `prefill` convolves a chunk; `decode` convolves one token
 * against the retained rows. Both refresh the state afterwards, so chunked prefill
 * composes: chunk N+1 sees chunk N's tail exactly as if the sequence had arrived whole.
 *
 * NO ACTIVATION. The reference fuses SiLU into its convolution kernel; this component does
 * not, because the tree already has an elementwise activation and fusing them would need a
 * second operation type for a saving that has not been measured. The block composes the
 * two, matching how AttentionOutputGate composes rather than fuses.
 *
 * Inference-only: no backward. Training the DeltaNet layers is not in scope, and a
 * backward that exists but has never been exercised is worse than one that does not.
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

export module Dnn.Components.CausalConv1d;

export import Dnn.Components.CausalConv1dConfig;

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
    class CausalConv1d : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        explicit CausalConv1d( const std::string& name, const CausalConv1dConfig& config,
            std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "CausalConv1d: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~CausalConv1d() override = default;

        /**
         * @brief Convolve a chunk starting at @p position_offset.
         *
         * At offset 0 the sequence starts here and the left context is zero; past that the
         * retained rows supply it. Refreshes the state on the way out.
         */
        TensorType& prefill( const TensorType& input, dim_t position_offset )
        {
            auto& output = run( input, position_offset > 0 );

            this->publish( ComputePass::Prefill, "output", output );

            return output;
        }

        /**
         * @brief Convolve one token against the retained rows.
         *
         * @p position is unused: unlike RoPE the convolution has no absolute notion of
         * where it is, only of the K-1 rows behind it. It is in the signature so the call
         * site reads like every other decode step, and so a caller cannot pass the two
         * paths different notions of position.
         */
        TensorType& decode( const TensorType& input, dim_t /*position*/ )
        {
            auto& output = run( input, /*use_state*/ true );

            this->publish( ComputePass::Decode, "output", output );

            return output;
        }

        /// Drop the retained rows. The next prefill starts a fresh sequence.
        void resetState()
        {
            if ( state_ )
            {
                zero( *state_, this->getExecutionContext() );
            }

            state_primed_ = false;
        }

        std::vector<std::string> getParameterNames() const override
        {
            if ( config_.hasBias() )
            {
                return { "weight", "bias" };
            }

            return { "weight" };
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> parameters;

            if ( weight_ )
            {
                parameters.push_back( weight_.get() );
            }

            if ( bias_ )
            {
                parameters.push_back( bias_.get() );
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

            if ( weight_ )
            {
                count += weight_->size();
            }

            if ( bias_ )
            {
                count += bias_->size();
            }

            return count;
        }

        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
        {
            if ( name == "weight" )
            {
                this->loadParameterFromBlob( "weight", blob, *weight_, weight_->shape() );
            }
            else if ( name == "bias" )
            {
                if ( !config_.hasBias() )
                {
                    throw std::runtime_error(
                        std::format( "Component '{}' was configured without bias", this->getName() ) );
                }

                this->loadParameterFromBlob( "bias", blob, *bias_, bias_->shape() );
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
            if ( weight_ )
            {
                this->saveParameterToWriter( writer, prefix + ".weight", *weight_, pass );
            }

            if ( bias_ )
            {
                this->saveParameterToWriter( writer, prefix + ".bias", *bias_, pass );
            }
        }

        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
        {
            SerializationMetadata meta;
            meta.set( "type", "CausalConv1d" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() )
                .set( "channels", static_cast<int64_t>( config_.getChannels() ) )
                .set( "kernel_width", static_cast<int64_t>( config_.getKernelWidth() ) )
                .set( "has_bias", config_.hasBias() );

            archive.writeMetadata( "meta.json", meta );

            for ( const auto& parameter_name : getParameterNames() )
            {
                if ( parameter_name == "weight" && weight_ )
                {
                    this->saveParameterToArchive( archive, parameter_name, *weight_ );
                }
                else if ( parameter_name == "bias" && bias_ )
                {
                    this->saveParameterToArchive( archive, parameter_name, *bias_ );
                }
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

        /**
         * @brief Install a shared output slot (activation pooling). Before build().
         */
        void installSharedOutput( std::shared_ptr<TensorType> output )
        {
            if ( this->isBuilt() )
                throw std::logic_error( std::format(
                    "CausalConv1d '{}': installSharedOutput must be called before build()",
                    this->getName() ) );

            output_ = std::move( output );
            output_installed_ = true;
        }

        const ComponentType getType() const override
        {
            return ComponentType::CausalConv1d;
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
            return { { "output", ComputePassMask{ ComputePass::Prefill, ComputePass::Decode } } };
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            if ( weight_ )
            {
                stats.device_parameter_bytes += weight_->getStorageSize();
            }

            if ( bias_ )
            {
                stats.device_parameter_bytes += bias_->getStorageSize();
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

            if ( !weight_ )
            {
                stats.device_parameter_bytes +=
                    storageBytes<TPrecision>( config_.getChannels() * config_.getKernelWidth() );

                if ( config_.hasBias() )
                {
                    stats.device_parameter_bytes += storageBytes<TPrecision>( config_.getChannels() );
                }
            }

            // The rolling window: batch * (K-1) * channels, independent of context length.
            // That independence is the whole point of the recurrence it serves.
            if ( !state_ )
            {
                stats.device_state_bytes += storageBytes<TPrecision>(
                    input_shape[ 0 ] * config_.getStateRows() * config_.getChannels() );
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
            oss << "CausalConv1d: " << this->getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Channels: " << config_.getChannels() << std::endl;
            oss << "Kernel width: " << config_.getKernelWidth()
                << " (retains " << config_.getStateRows() << " rows)" << std::endl;
            oss << "Has Bias: " << (config_.hasBias() ? "Yes" : "No") << std::endl;
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

    protected:

        void onExecutionContextSet() override
        {
            operation_ = std::make_shared<OpType>( this->getExecutionContext(), config_ );

            if ( !operation_ )
            {
                throw std::runtime_error( "Failed to create CausalConv1d compute backend operation." );
            }
        }

        void onBuilding( const BuildContext& build_context ) override
        {
            const auto& input_shape = build_context.inputShape();

            validateInputShape( input_shape );

            auto device = this->getExecutionContext()->getDeviceId();
            const dim_t channels = config_.getChannels();
            const dim_t kernel_width = config_.getKernelWidth();

            if ( !weight_ )
            {
                weight_ = std::make_shared<TensorType>(
                    device, shape_t{ channels, kernel_width }, this->getName() + ".weight" );

                if ( config_.hasBias() )
                {
                    bias_ = std::make_shared<TensorType>(
                        device, shape_t{ channels }, this->getName() + ".bias" );
                }
            }

            if ( build_context.shouldInitializeParameters() )
            {
                fill( *weight_, 0.0f, this->getExecutionContext() );

                if ( bias_ )
                {
                    zero( *bias_, this->getExecutionContext() );
                }
            }

            operation_->setParameters( weight_.get(), bias_.get() );
            operation_->build( build_context );

            batch_ = input_shape[ 0 ];

            state_ = std::make_shared<TensorType>(
                device, shape_t{ batch_, config_.getStateRows(), channels }, this->getName() + ".state" );
            zero( *state_, this->getExecutionContext() );
            state_primed_ = false;

            if ( output_installed_ )
            {
                if ( !output_ || output_->size() < elementCount( input_shape ) )
                    throw std::invalid_argument( std::format(
                        "CausalConv1d '{}': installed shared output slot is smaller than the "
                        "build shape requires", this->getName() ) );
            }
            else
            {
                output_ = std::make_shared<TensorType>(
                    device, input_shape, this->getName() + ".output" );
            }

            output_view_.emplace( output_->view( input_shape ) );
        }

        void onTrainingModeChanging( TrainingMode /*training_mode*/ ) override
        {
            // Inference-only: no gradients to clear and no operation-side training state.
        }

    private:
        using OpType = typename OperationTraits<OperationType::CausalConv1dOp, TDeviceType, TPrecision>::type;

        CausalConv1dConfig config_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        // The rolling window of the last K-1 input rows, [B, K-1, C].
        std::shared_ptr<TensorType> state_{ nullptr };
        bool state_primed_{ false };
        dim_t batch_{ 0 };

        std::shared_ptr<TensorType> output_{ nullptr };
        bool output_installed_{ false };
        std::optional<TensorType> output_view_;

        TensorType& run( const TensorType& input, bool use_state )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "CausalConv1d::run: must be built before use." );
            }

            const auto& input_shape = input.shape();

            validateInputShape( input_shape );

            if ( input_shape[ 0 ] != batch_ )
            {
                throw std::invalid_argument( std::format(
                    "CausalConv1d '{}': batch {} does not match the built batch {}",
                    this->getName(), input_shape[ 0 ], batch_ ) );
            }

            if ( output_view_->shape() != input_shape )
            {
                output_view_.emplace( output_->view( input_shape ) );
            }

            // A caller asking for left context before any has been retained is asking for
            // the previous chunk of a sequence that never had one. Zeros would answer it
            // silently and wrongly, so refuse instead.
            if ( use_state && !state_primed_ )
            {
                throw std::logic_error( std::format(
                    "CausalConv1d '{}': continuation requested before any chunk was seen -- "
                    "call prefill() at position 0 first, or resetState() to start over",
                    this->getName() ) );
            }

            operation_->forward( input, use_state ? state_.get() : nullptr, *output_view_ );
            operation_->updateState( input, *state_ );

            state_primed_ = true;

            return *output_view_;
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 3 )
            {
                throw std::invalid_argument( std::format(
                    "CausalConv1d '{}': input must be rank 3 [B, T, C], got rank {}",
                    this->getName(), input_shape.size() ) );
            }

            if ( input_shape[ 2 ] != config_.getChannels() )
            {
                throw std::invalid_argument( std::format(
                    "CausalConv1d '{}': channel mismatch -- expected {}, got {}",
                    this->getName(), config_.getChannels(), input_shape[ 2 ] ) );
            }
        }
    };
}
