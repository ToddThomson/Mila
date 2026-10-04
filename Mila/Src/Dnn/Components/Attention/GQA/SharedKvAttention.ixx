/**
 * @file SharedKvAttention.ixx
 * @brief Attention for a layer that projects queries only and attends a KV cache another layer owns.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <optional>

export module Dnn.Components.SharedKvAttention;
export import Dnn.Components.GqaConfig;
export import Compute.KvCacheView;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceAllocation;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.OperationTraits;
import Compute.OperationType;
import Compute.Observation;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief One decode step's attention over a cache it reads and never writes.
     *
     * Gemma 4's draft model is four layers with query projections only: each attends the target's cache of the
     * same layer type (Gemma4Mtp.md 4.3). The query is rotated at the decode position before it arrives here, and
     * attends the cache's keys up to the position before it, within the cache's window. Decode only: the reader has
     * no prefill, no cache and no parameters. The config's key/value heads and window must be the cache's.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class SharedKvAttention : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using OpType = typename OperationTraits<OperationType::SharedKvAttentionOp, TDeviceType, TPrecision, void>::type;

        explicit SharedKvAttention( const std::string& name, const GqaConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "SharedKvAttention: device type mismatch" );
                }

                context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( context_.get() );
            }
        }

        ~SharedKvAttention() override = default;

        /**
         * @brief Attend `cache` with the rotated query `q` [B, 1, num_heads * head_size].
         *
         * @return The component-owned output [B, 1, num_heads * head_size].
         */
        TensorType& decode( const TensorType& q, const KvCacheView& cache )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "SharedKvAttention must be built before decode()." );

            operation_->decode( q, cache, *output_ );

            this->publish( ComputePass::Decode, "output", *output_ );

            return *output_;
        }

        std::vector<ITensor*> getParameters() const override
        {
            return {};
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        const ComponentType getType() const override
        {
            return ComponentType::SharedKvAttention;
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        dim_t parameterCount() const override
        {
            return 0;
        }

        /// What onBuilding() would allocate for [B, T, num_heads * head_size], without allocating.
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            const std::size_t granularity = context.getAllocationGranularity();

            MemoryStats stats;
            stats.device_state_bytes += operation_->getRequiredStateMemorySize( context );
            stats.device_state_bytes += occupiedDeviceBytes(
                storageBytes<TPrecision>( context.inputShape()[ 0 ] * config_.getModelDim() ), granularity );
            stats.device_scratch_bytes = occupiedDeviceBytes( operation_->getRequiredScratchBytes( context ), granularity );

            return stats;
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;
            stats.device_state_bytes += operation_->getStateMemorySize();
            stats.device_scratch_bytes =
                occupiedDeviceBytes( operation_->getScratchBytes(), allocationGranularity( this->getDeviceId() ) );

            if ( output_ != nullptr )
                stats.device_state_bytes += occupiedTensorBytes( *output_ );

            return stats;
        }

        std::vector<const ITensor*> getOutputs() const override
        {
            return output_ != nullptr ? std::vector<const ITensor*>{ output_.get() } : std::vector<const ITensor*>{};
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "output", ComputePassMask{ ComputePass::Decode } } };
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "--------------------\n";
            oss << "SharedKvAttention: " << this->getName() << "\n";
            oss << "Num Q heads: " << config_.getNumHeads() << "\n";
            oss << "Num KV heads (read): " << config_.getNumKvHeads() << "\n";
            oss << "Head size: " << config_.getHeadDim() << "\n";
            oss << "Window: " << config_.getWindow() << "\n";

            return oss.str();
        }

        const GqaConfig& getConfig() const noexcept
        {
            return config_;
        }

    protected:

        void save_( ModelArchive&, SerializationMode ) const override
        {
            // No parameters: the projections around it own the weights.
        }

        void onExecutionContextSet() override
        {
            operation_ = std::make_shared<OpType>( this->getExecutionContext(), config_ );
        }

        void onBuilding( const BuildContext& context ) override
        {
            operation_->build( context );

            output_ = std::make_unique<TensorType>( this->getExecutionContext()->getDeviceId(),
                shape_t{ context.inputShape()[ 0 ], 1, config_.getModelDim() }, this->getName() + ".output" );
        }

    private:
        GqaConfig config_;

        std::unique_ptr<IExecutionContext> context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };
        std::unique_ptr<TensorType> output_{ nullptr };
    };
}
