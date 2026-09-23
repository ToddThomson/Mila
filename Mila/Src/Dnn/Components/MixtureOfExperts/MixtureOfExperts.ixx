/**
 * @file MixtureOfExperts.ixx
 * @brief Mixture-of-experts bank: the stacked experts and the weighted combine, routed from outside.
 *
 * Routing is an input, not a child. Gemma 4 routes on the raw residual while its experts read
 * pre_feedforward_layernorm_2 of it (Specifications/Gemma4MoE.md Phase 1), so a bank that owned its
 * router would need a family-specific two-input signature. The block wires Router beside it.
 */

module;
#include <cstddef>
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
import Dnn.Quantization.Weight.Policies;
import Compute.DeviceAllocation;
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
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief Stacked experts: gate_up_proj [E, 2I, H] and down_proj [E, H, I], checkpoint-named.
     *
     * An expert is a row of these tensors, never an object. Inference-only. Under a per-group FP4
     * policy each row is stored as a Linear FP4 weight row is -- packed nibbles plus per-group FP32
     * scales, saved as gate_up_proj_scale and down_proj_scale -- and a BF16 source is quantized on load.
     *
     * @tparam TGate Gate activation: Silu (SwiGLU experts) or Gelu (GeGLU experts, Gemma).
     * @tparam TWeightQuantization NoWeightQuant or PerGroupFp4<g>; the operation refuses any other.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision, ActivationType TGate = ActivationType::Silu,
        WeightQuantPolicy TWeightQuantization = NoWeightQuant>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class MixtureOfExperts : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using IndexTensorType = Tensor<TensorDataType::INT32, MR>;

        static constexpr bool kIsQuantized = TWeightQuantization::kIsQuantized;
        static constexpr TensorDataType kWeightDtype = kIsQuantized ? TWeightQuantization::kStorageDtype : TPrecision;

        static constexpr bool kIsFp4 = []() constexpr
        {
            if constexpr ( requires { TWeightQuantization::kIsFp4E2M1; } )
            {
                return TWeightQuantization::kIsQuantized && TWeightQuantization::kIsFp4E2M1;
            }
            else
            {
                return false;
            }
        }();

        using WeightTensorType = Tensor<kWeightDtype, MR>;
        using ScaleTensorType = Tensor<TensorDataType::FP32, MR>;

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

        /// The stored tensors, scales included, so storage accounting sees every byte.
        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> parameters;

            for ( ITensor* tensor : { static_cast<ITensor*>( gate_up_proj_.get() ), static_cast<ITensor*>( gate_up_scale_.get() ),
                                      static_cast<ITensor*>( down_proj_.get() ), static_cast<ITensor*>( down_scale_.get() ) } )
            {
                if ( tensor )
                {
                    parameters.push_back( tensor );
                }
            }

            return parameters;
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        /// Logical expert weights, whatever their storage.
        dim_t parameterCount() const override
        {
            return gate_up_proj_
                ? config_.getNumExperts() * 3 * config_.getExpertIntermediateSize() * config_.getHiddenSize()
                : 0;
        }

        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
        {
            const dim_t hidden = config_.getHiddenSize();
            const dim_t intermediate = config_.getExpertIntermediateSize();

            if ( name == "gate_up_proj" )
            {
                loadProjection( name, blob, *gate_up_proj_, gate_up_scale_.get(), 2 * intermediate, hidden );
            }
            else if ( name == "down_proj" )
            {
                loadProjection( name, blob, *down_proj_, down_scale_.get(), hidden, intermediate );
            }
            else if ( name == "gate_up_proj_scale" || name == "down_proj_scale" )
            {
                ScaleTensorType* scales = name == "gate_up_proj_scale" ? gate_up_scale_.get() : down_scale_.get();

                if ( scales == nullptr )
                {
                    throw std::invalid_argument( std::format(
                        "MixtureOfExperts '{}': received '{}' but this bank stores no scales", this->getName(), name ) );
                }

                this->loadParameterFromBlob( name, blob, *scales, scales->shape() );
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

            if ( gate_up_scale_ )
            {
                this->saveParameterToWriter( writer, prefix + ".gate_up_proj_scale", *gate_up_scale_, pass );
            }

            this->saveParameterToWriter( writer, prefix + ".down_proj", *down_proj_, pass );

            if ( down_scale_ )
            {
                this->saveParameterToWriter( writer, prefix + ".down_proj_scale", *down_scale_, pass );
            }
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

            if ( gate_up_scale_ && down_scale_ )
            {
                this->saveParameterToArchive( archive, "gate_up_proj_scale", *gate_up_scale_ );
                this->saveParameterToArchive( archive, "down_proj_scale", *down_scale_ );
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
                stats.device_parameter_bytes += occupiedTensorBytes( *parameter );
            }

            stats.device_inactive_parameter_bytes = inactiveBytes( stats.device_parameter_bytes );

            if ( output_ )
            {
                stats.device_state_bytes += occupiedTensorBytes( *output_ );
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

            const std::size_t granularity = context.getAllocationGranularity();

            if ( !gate_up_proj_ )
            {
                const dim_t hidden = config_.getHiddenSize();
                const dim_t intermediate = config_.getExpertIntermediateSize();

                stats.device_parameter_bytes += projectionBytes( 2 * intermediate, hidden, granularity )
                    + projectionBytes( hidden, intermediate, granularity );
                stats.device_inactive_parameter_bytes = inactiveBytes( stats.device_parameter_bytes );
            }

            stats.device_state_bytes +=
                occupiedDeviceBytes( storageBytes<TPrecision>( elementCount( input_shape ) ), granularity );

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
                gate_up_proj_ = std::make_shared<WeightTensorType>(
                    device, shape_t{ experts, 2 * intermediate, storedColumns( hidden ) }, this->getName() + ".gate_up_proj" );
                down_proj_ = std::make_shared<WeightTensorType>(
                    device, shape_t{ experts, hidden, storedColumns( intermediate ) }, this->getName() + ".down_proj" );

                if constexpr ( kIsFp4 )
                {
                    gate_up_scale_ = std::make_shared<ScaleTensorType>(
                        device, shape_t{ experts, 2 * intermediate, groupsPerRow( hidden ) }, this->getName() + ".gate_up_proj_scale" );
                    down_scale_ = std::make_shared<ScaleTensorType>(
                        device, shape_t{ experts, hidden, groupsPerRow( intermediate ) }, this->getName() + ".down_proj_scale" );
                }
            }

            // Packed storage has no meaningful initial value: a quantized bank is filled only by a load.
            if constexpr ( !kIsQuantized )
            {
                if ( context.shouldInitializeParameters() )
                {
                    zero( *gate_up_proj_, this->getExecutionContext() );
                    zero( *down_proj_, this->getExecutionContext() );
                }
            }

            operation_->setParameters( gate_up_proj_.get(), down_proj_.get() );

            if constexpr ( kIsFp4 )
            {
                operation_->setWeightScales( gate_up_scale_.get(), down_scale_.get() );
            }

            operation_->build( context );

            output_ = std::make_shared<TensorType>( device, input_shape, this->getName() + ".output" );
            output_view_.emplace( output_->view( input_shape ) );
        }

        void onTrainingModeChanging( TrainingMode /*training_mode*/ ) override
        {
        }

    private:
        using OpType = typename OperationTraits<OperationType::MoeOp, TDeviceType, TPrecision>::template op_for<functor_of_t<TGate>, TWeightQuantization>;

        MixtureOfExpertsConfig config_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };

        std::shared_ptr<WeightTensorType> gate_up_proj_{ nullptr };
        std::shared_ptr<WeightTensorType> down_proj_{ nullptr };
        // Per-group FP4 only.
        std::shared_ptr<ScaleTensorType> gate_up_scale_{ nullptr };
        std::shared_ptr<ScaleTensorType> down_scale_{ nullptr };

        std::shared_ptr<TensorType> output_{ nullptr };
        std::optional<TensorType> output_view_;

        // A packed blob loads as stored; under FP4 a BF16 blob is quantized into the packed storage.
        void loadProjection( const std::string& name, const ITensorBlob& blob, WeightTensorType& target,
            ScaleTensorType* scales, dim_t rows, dim_t columns )
        {
            if constexpr ( kIsFp4 )
            {
                if ( blob.getMetadata().dtype != kWeightDtype )
                {
                    const shape_t expected{ config_.getNumExperts(), rows, columns };

                    if ( blob.getMetadata().shape != expected )
                    {
                        throw std::invalid_argument( std::format(
                            "MixtureOfExperts '{}': '{}' must be {} to quantize, got {}",
                            this->getName(), name, shapeToString( expected ), shapeToString( blob.getMetadata().shape ) ) );
                    }

                    operation_->quantize( blob, target, *scales, config_.getNumExperts() * rows, columns );

                    return;
                }
            }

            this->loadParameterFromBlob( name, blob, target, target.shape() );
        }

        static dim_t storedColumns( dim_t columns ) noexcept
        {
            if constexpr ( kIsQuantized )
            {
                return columns * TWeightQuantization::kStorageBitsPerElement / 8;
            }
            else
            {
                return columns;
            }
        }

        static dim_t groupsPerRow( dim_t columns ) noexcept
        {
            if constexpr ( kIsFp4 )
            {
                return columns / TWeightQuantization::kQuantizationGroupSize;
            }
            else
            {
                return 0;
            }
        }

        // The stacked weight and its scales are two allocations.
        std::size_t projectionBytes( dim_t rows, dim_t columns, std::size_t granularity ) const
        {
            const dim_t experts = config_.getNumExperts();

            return occupiedDeviceBytes( storageBytes<kWeightDtype>( experts * rows * storedColumns( columns ) ), granularity )
                + occupiedDeviceBytes(
                    storageBytes<TensorDataType::FP32>( experts * rows * groupsPerRow( columns ) ), granularity );
        }

        // A token reads top_k of the experts' equal rows; the rest are resident and untouched.
        std::size_t inactiveBytes( std::size_t parameter_bytes ) const noexcept
        {
            const auto experts = static_cast<std::size_t>( config_.getNumExperts() );

            return parameter_bytes / experts * ( experts - static_cast<std::size_t>( config_.getTopK() ) );
        }

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
