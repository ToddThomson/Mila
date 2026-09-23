/**
 * @file Router.ixx
 * @brief Mixture-of-experts router: which experts each token runs, and with what combine weights.
 *
 * Gemma 4's routing chain (Specifications/Gemma4MoE.md Phase 1):
 *   logits           = proj( rms_norm_without_scale( x ) * scale * hidden_size^-0.5 )
 *   weights, indices = top_k( softmax( logits ) ), renormalized, times per_expert_scale[ index ]
 */

module;
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

export module Dnn.Components.Router;

export import Dnn.Components.RouterConfig;

import Dnn.Component;
import Dnn.ComponentType;
import Dnn.CompositeComponent;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Dnn.Components.RmsNorm;
import Dnn.Components.Linear;
import Compute.DeviceAllocation;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.IExecutionContext;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.OperationTraits;
import Compute.CpuMemoryResource;
import Compute.Observation;
import Serialization.ModelArchive;
import Serialization.Mode;
import Serialization.Tensor;
import Serialization.SafeTensors;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Routes each token to its top_k experts.
     *
     * The unscaled RMS norm and the learned `scale` are expressed as one `RmsNorm` child whose weight
     * is `scale * hidden_size^-0.5`, derived whenever `scale` is initialized or loaded. That weight is
     * therefore not a parameter of its own: the flat vocabulary is the checkpoint's -- `proj.weight`,
     * `scale`, `per_expert_scale` -- and the derived weight is never written to it.
     *
     * Inference-only. The projection stays unquantized: the router is a precision holdout
     * (MixtureOfExperts.md section 7.5).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Router : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using IndexTensorType = Tensor<TensorDataType::INT32, MR>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using LinearType = Linear<TDeviceType, TPrecision>;

        /// One call's result. Both views are router-owned and valid until the next forward().
        struct Routing
        {
            TensorType& weights;
            IndexTensorType& indices;
        };

        explicit Router( const std::string& name, const RouterConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : CompositeComponentBase( name ), config_( config )
        {
            config_.validate();

            createGraph();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "Router: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~Router() override = default;

        /**
         * @brief Select experts for every token of @p input.
         *
         * @param input [..., hidden_size], the residual stream the router reads unnormalized.
         * @return Combine weights [..., top_k] and expert indices [..., top_k]. Order within a row
         *         carries no meaning: the combine is a sum over the selected experts.
         */
        Routing forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Router::forward: must be built before use." );
            }

            auto& normalized = norm_->forward( input );
            auto& logits = proj_->forward( normalized );

            shape_t routing_shape = input.shape();
            routing_shape.back() = config_.getTopK();

            if ( weights_view_->shape() != routing_shape )
            {
                weights_view_.emplace( weights_->view( routing_shape ) );
                indices_view_.emplace( indices_->view( routing_shape ) );
            }

            operation_->forward( logits, *weights_view_, *indices_view_ );

            this->publish( ComputePass::Forward, "weights", *weights_view_ );

            return { *weights_view_, *indices_view_ };
        }

        std::vector<std::string> getParameterNames() const override
        {
            return { "scale", "per_expert_scale" };
        }

        /// The projection and the two owned vectors; the norm's derived weight is not counted.
        dim_t parameterCount() const override
        {
            dim_t count = this->template getComponentAs<LinearType>( this->getName() + ".proj" )->parameterCount();

            if ( scale_ )
            {
                count += scale_->size();
            }

            if ( per_expert_scale_ )
            {
                count += per_expert_scale_->size();
            }

            return count;
        }

        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
        {
            if ( name == "scale" )
            {
                this->loadParameterFromBlob( "scale", blob, *scale_, scale_->shape() );
                deriveNormWeight();
            }
            else if ( name == "per_expert_scale" )
            {
                this->loadParameterFromBlob( "per_expert_scale", blob, *per_expert_scale_, per_expert_scale_->shape() );
            }
            else
            {
                CompositeComponentBase::loadParameter( name, blob );
            }
        }

        /**
         * @brief The checkpoint's vocabulary: the projection, then the two owned vectors.
         *
         * Deliberately not the base recursion, which would also write the norm child's derived
         * weight -- a tensor no checkpoint carries and every load would have to ignore.
         */
        void saveFlatTensors(
            Serialization::SafeTensorsWriter& writer,
            const std::string& prefix,
            Serialization::TensorSavePass pass ) const override
        {
            const std::string owned_prefix = prefix.empty() ? std::string{} : prefix + ".";

            this->template getComponentAs<LinearType>( this->getName() + ".proj" )
                ->saveFlatTensors( writer, owned_prefix + "proj", pass );

            this->saveParameterToWriter( writer, owned_prefix + "scale", *scale_, pass );
            this->saveParameterToWriter( writer, owned_prefix + "per_expert_scale", *per_expert_scale_, pass );
        }

        const ComponentType getType() const override
        {
            return ComponentType::Router;
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "weights", ComputePassMask{ ComputePass::Forward } } };
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
            {
                stats += child->getMemoryStats();
            }

            for ( const ITensor* tensor : { static_cast<const ITensor*>( scale_.get() ),
                                            static_cast<const ITensor*>( per_expert_scale_.get() ) } )
            {
                if ( tensor )
                {
                    stats.device_parameter_bytes += occupiedTensorBytes( *tensor );
                }
            }

            if ( weights_ )
            {
                stats.device_state_bytes += occupiedTensorBytes( *weights_ );
            }

            if ( indices_ )
            {
                stats.device_state_bytes += occupiedTensorBytes( *indices_ );
            }

            return stats;
        }

        /**
         * @brief What onBuilding() would allocate for this context, without allocating.
         *
         * Children by name, sized as onBuilding() builds them: both read the input shape.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            const auto& input_shape = context.inputShape();

            validateInputShape( input_shape );

            const std::string n = this->getName();

            MemoryStats stats;

            stats += this->template getComponentAs<RmsNormType>( n + ".norm" )->getRequiredMemory( context );
            stats += this->template getComponentAs<LinearType>( n + ".proj" )->getRequiredMemory( context );

            const std::size_t granularity = context.getAllocationGranularity();

            // scale and per_expert_scale are two allocations.
            if ( !scale_ )
            {
                stats.device_parameter_bytes +=
                    occupiedDeviceBytes( storageBytes<TPrecision>( config_.getHiddenSize() ), granularity )
                    + occupiedDeviceBytes( storageBytes<TPrecision>( config_.getNumExperts() ), granularity );
            }

            const dim_t routing_elements = elementCount( input_shape ) / config_.getHiddenSize() * config_.getTopK();

            stats.device_state_bytes += occupiedDeviceBytes( storageBytes<TPrecision>( routing_elements ), granularity );
            stats.device_state_bytes +=
                occupiedDeviceBytes( storageBytes<TensorDataType::INT32>( routing_elements ), granularity );

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Router: " << this->getName() << std::endl;
            oss << "Hidden size: " << config_.getHiddenSize() << std::endl;
            oss << "Experts: top " << config_.getTopK() << " of " << config_.getNumExperts() << std::endl;

            return oss.str();
        }

    protected:

        void onExecutionContextSet() override
        {
            CompositeComponentBase::onExecutionContextSet();

            operation_ = std::make_shared<OpType>( this->getExecutionContext(), config_ );
        }

        void onBuilding( const BuildContext& context ) override
        {
            const auto& input_shape = context.inputShape();

            validateInputShape( input_shape );

            const std::string n = this->getName();
            auto device = this->getExecutionContext()->getDeviceId();

            norm_ = this->template getComponentAs<RmsNormType>( n + ".norm" );
            norm_->build( context );

            proj_ = this->template getComponentAs<LinearType>( n + ".proj" );
            proj_->build( context );

            if ( !scale_ )
            {
                scale_ = std::make_shared<TensorType>( device, shape_t{ config_.getHiddenSize() }, n + ".scale" );
                per_expert_scale_ = std::make_shared<TensorType>( device, shape_t{ config_.getNumExperts() }, n + ".per_expert_scale" );
            }

            if ( context.shouldInitializeParameters() )
            {
                fill( *scale_, 1.0f, this->getExecutionContext() );
                fill( *per_expert_scale_, 1.0f, this->getExecutionContext() );
            }

            deriveNormWeight();

            shape_t logits_shape = input_shape;
            logits_shape.back() = config_.getNumExperts();

            operation_->setParameters( per_expert_scale_.get(), nullptr );
            operation_->build( context.withShape( logits_shape ) );

            shape_t routing_shape = input_shape;
            routing_shape.back() = config_.getTopK();

            weights_ = std::make_shared<TensorType>( device, routing_shape, n + ".weights" );
            indices_ = std::make_shared<IndexTensorType>( device, routing_shape, n + ".indices" );

            weights_view_.emplace( weights_->view( routing_shape ) );
            indices_view_.emplace( indices_->view( routing_shape ) );
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            for ( const auto& child : this->getComponents() )
            {
                child->setTrainingMode( training_mode );
            }
        }

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            CompositeComponentBase::save_( archive, mode );

            this->saveParameterToArchive( archive, "scale", *scale_ );
            this->saveParameterToArchive( archive, "per_expert_scale", *per_expert_scale_ );
        }

        void load_( ModelArchive& archive, SerializationMode mode ) override
        {
            CompositeComponentBase::load_( archive, mode );

            for ( const auto& parameter_name : getParameterNames() )
            {
                const std::string prefix = "tensors/" + parameter_name;

                if ( !archive.hasFile( prefix + "/data.bin" ) )
                {
                    throw std::runtime_error( std::format(
                        "Router '{}': archive has no blob for '{}'", this->getName(), parameter_name ) );
                }

                auto blob = readTensorBlob<CpuMemoryResource>( archive, prefix );

                loadParameter( parameter_name, blob );
            }
        }

    private:
        using OpType = typename OperationTraits<OperationType::RouterOp, TDeviceType, TPrecision>::type;

        RouterConfig config_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };

        std::shared_ptr<RmsNormType> norm_{ nullptr };
        std::shared_ptr<LinearType> proj_{ nullptr };

        std::shared_ptr<TensorType> scale_{ nullptr };
        std::shared_ptr<TensorType> per_expert_scale_{ nullptr };

        std::shared_ptr<TensorType> weights_{ nullptr };
        std::shared_ptr<IndexTensorType> indices_{ nullptr };
        std::optional<TensorType> weights_view_;
        std::optional<IndexTensorType> indices_view_;

        void createGraph()
        {
            const std::string n = this->getName();
            const dim_t hidden_size = config_.getHiddenSize();

            this->addComponent( std::make_shared<RmsNormType>( n + ".norm",
                RmsNormConfig( shape_t{ hidden_size } ).withEpsilon( config_.getEpsilon() ).withBias( false ) ) );

            this->addComponent( std::make_shared<LinearType>( n + ".proj",
                LinearConfig( hidden_size, config_.getNumExperts() ).withBias( false ) ) );
        }

        // norm weight = scale * hidden_size^-0.5 -- the unscaled norm, the learned scale and the
        // root-size factor folded into the one multiply RmsNorm applies.
        void deriveNormWeight()
        {
            if ( !norm_ || !scale_ )
            {
                throw std::logic_error( std::format(
                    "Router '{}': the norm weight is derived from scale after build", this->getName() ) );
            }

            auto* weight = static_cast<TensorType*>( norm_->getParameters()[ 0 ] );
            const float root_size = static_cast<float>( 1.0 / std::sqrt( static_cast<double>( config_.getHiddenSize() ) ) );

            copy( *scale_, *weight, this->getExecutionContext() );
            scale( *weight, root_size, *weight, this->getExecutionContext() );
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.empty() || input_shape.back() != config_.getHiddenSize() )
            {
                throw std::invalid_argument( std::format(
                    "Router '{}': input must be [..., {}]", this->getName(), config_.getHiddenSize() ) );
            }
        }
    };
}
