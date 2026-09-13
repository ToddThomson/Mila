/**
 * @file MixtureOfExperts.ixx
 * @brief Mixture-of-experts bank: the stacked experts and the weighted combine, routed from outside.
 *
 * Routing is an input, not a child. Gemma 4 routes on the raw residual while its experts read
 * pre_feedforward_layernorm_2 of it (Specifications/Gemma4MoE.md Phase 1), so a bank that owned its
 * router would need a family-specific two-input signature. The block wires Router beside it.
 */

module;
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../Activations/Activation/Kernels/ElementwiseActivation.h"

export module Dnn.Components.MixtureOfExperts;

export import Dnn.Components.MixtureOfExpertsConfig;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Dnn.ActivationType;
import Dnn.Components.Activation;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.OperationTraits;
import Compute.Observation;
import Serialization.ModelArchive;
import Serialization.Metadata;
import Serialization.Mode;
import Serialization.Tensor;
import Serialization.SafeTensors;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Stacked experts: gate_up_proj [E, 2I, H] and down_proj [E, H, I], checkpoint-named.
     *
     * An expert is a row of these tensors, never an object. Inference-only.
     *
     * @tparam TGate Gate activation: Silu (SwiGLU experts) or Gelu (GeGLU experts, Gemma).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision, ActivationType TGate = ActivationType::Silu>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class MixtureOfExperts : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using IndexTensorType = Tensor<TensorDataType::INT32, MR>;

        explicit MixtureOfExperts( const std::string& name, const MixtureOfExpertsConfig& config,
            std::optional<DeviceId> device_id = std::nullopt )
            : ComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "MixtureOfExperts: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~MixtureOfExperts() override = default;

        /**
         * @brief Run every token through its selected experts and combine.
         *
         * @param input    [..., hidden_size], what the experts read.
         * @param weights  [..., top_k] combine weights, as Router produces them.
         * @param indices  [..., top_k] expert indices, as Router produces them.
         * @return [..., hidden_size], component-owned, valid until the next forward().
         */
        TensorType& forward( const TensorType& input, const TensorType& weights, const IndexTensorType& indices )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "MixtureOfExperts::forward: must be built before use." );
            }

            if ( output_view_->shape() != input.shape() )
            {
                output_view_.emplace( output_->view( input.shape() ) );
            }

            operation_->forward( input, weights, indices, *output_view_ );

            this->publish( ComputePass::Forward, "output", *output_view_ );

            return *output_view_;
        }

        std::vector<std::string> getParameterNames() const override
        {
            return { "gate_up_proj", "down_proj" };
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> parameters;

            if ( gate_up_proj_ )
            {
                parameters.push_back( gate_up_proj_.get() );
            }

            if ( down_proj_ )
            {
                parameters.push_back( down_proj_.get() );
            }

            return parameters;
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        dim_t parameterCount() const override
        {
            return ( gate_up_proj_ ? gate_up_proj_->size() : 0 ) + ( down_proj_ ? down_proj_->size() : 0 );
        }

        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
        {
            if ( name == "gate_up_proj" )
            {
                this->loadParameterFromBlob( "gate_up_proj", blob, *gate_up_proj_, gate_up_proj_->shape() );
            }
            else if ( name == "down_proj" )
            {
                this->loadParameterFromBlob( "down_proj", blob, *down_proj_, down_proj_->shape() );
            }
            else
            {
                throw std::invalid_argument( std::format(
                    "MixtureOfExperts '{}': no parameter named '{}'", this->getName(), name ) );
            }
        }

        void saveFlatTensors(
            Serialization::SafeTensorsWriter& writer,
            const std::string& prefix,
            Serialization::TensorSavePass pass ) const override
        {
            this->saveParameterToWriter( writer, prefix + ".gate_up_proj", *gate_up_proj_, pass );
            this->saveParameterToWriter( writer, prefix + ".down_proj", *down_proj_, pass );
        }

        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
        {
            SerializationMetadata meta = config_.toMetadata();
            meta.set( "type", "MixtureOfExperts" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() );

            archive.writeMetadata( "meta.json", meta );

            this->saveParameterToArchive( archive, "gate_up_proj", *gate_up_proj_ );
            this->saveParameterToArchive( archive, "down_proj", *down_proj_ );
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        const ComponentType getType() const override
        {
            return ComponentType::MixtureOfExperts;
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "output", ComputePassMask{ ComputePass::Forward } } };
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const ITensor* parameter : getParameters() )
            {
                stats.device_parameter_bytes += parameter->getStorageSize();
            }

            if ( output_ )
            {
                stats.device_state_bytes += output_->getStorageSize();
            }

            if ( operation_ )
            {
                stats.device_state_bytes += operation_->getStateMemorySize();
            }

            return stats;
        }

        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            const auto& input_shape = context.inputShape();

            validateInputShape( input_shape );

            MemoryStats stats;

            if ( !gate_up_proj_ )
            {
                const dim_t experts = config_.getNumExperts();
                const dim_t hidden = config_.getHiddenSize();
                const dim_t intermediate = config_.getExpertIntermediateSize();

                stats.device_parameter_bytes += storageBytes<TPrecision>( experts * 3 * intermediate * hidden );
            }

            stats.device_state_bytes += storageBytes<TPrecision>( elementCount( input_shape ) );

            if ( operation_ )
            {
                stats.device_state_bytes += operation_->getRequiredStateMemorySize( context );
            }

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "MixtureOfExperts: " << this->getName() << std::endl;
            oss << "Experts: " << config_.getNumExperts() << " x intermediate " << config_.getExpertIntermediateSize()
                << ", top " << config_.getTopK() << " per token" << std::endl;

            return oss.str();
        }

    protected:

        void onExecutionContextSet() override
        {
            operation_ = std::make_shared<OpType>( this->getExecutionContext(), config_ );
        }

        void onBuilding( const BuildContext& context ) override
        {
            const auto& input_shape = context.inputShape();

            validateInputShape( input_shape );

            const dim_t experts = config_.getNumExperts();
            const dim_t hidden = config_.getHiddenSize();
            const dim_t intermediate = config_.getExpertIntermediateSize();
            auto device = this->getExecutionContext()->getDeviceId();

            if ( !gate_up_proj_ )
            {
                gate_up_proj_ = std::make_shared<TensorType>(
                    device, shape_t{ experts, 2 * intermediate, hidden }, this->getName() + ".gate_up_proj" );
                down_proj_ = std::make_shared<TensorType>(
                    device, shape_t{ experts, hidden, intermediate }, this->getName() + ".down_proj" );
            }

            if ( context.shouldInitializeParameters() )
            {
                zero( *gate_up_proj_, this->getExecutionContext() );
                zero( *down_proj_, this->getExecutionContext() );
            }

            operation_->setParameters( gate_up_proj_.get(), down_proj_.get() );
            operation_->build( context );

            output_ = std::make_shared<TensorType>( device, input_shape, this->getName() + ".output" );
            output_view_.emplace( output_->view( input_shape ) );
        }

        void onTrainingModeChanging( TrainingMode /*training_mode*/ ) override
        {
        }

    private:
        using OpType = typename OperationTraits<OperationType::MoeOp, TDeviceType, TPrecision>::template op_for<functor_of_t<TGate>>;

        MixtureOfExpertsConfig config_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };

        std::shared_ptr<TensorType> gate_up_proj_{ nullptr };
        std::shared_ptr<TensorType> down_proj_{ nullptr };

        std::shared_ptr<TensorType> output_{ nullptr };
        std::optional<TensorType> output_view_;

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.empty() || input_shape.back() != config_.getHiddenSize() )
            {
                throw std::invalid_argument( std::format(
                    "MixtureOfExperts '{}': input must be [..., {}]", this->getName(), config_.getHiddenSize() ) );
            }
        }
    };
}
