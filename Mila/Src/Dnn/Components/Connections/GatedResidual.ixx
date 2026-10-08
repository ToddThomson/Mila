/**
 * @file GatedResidual.ixx
 * @brief Gated residual (hyper-connections): n residual streams read into one sublayer input, and written back gated.
 *
 * Qwen 4's residual (Specifications/Qwen4.md section 3.1). Each sublayer reads its H-wide input from the n * H-wide
 * stream state through one of these and writes its output back through the same one.
 */

module;
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

export module Dnn.Components.GatedResidual;

export import Dnn.Components.GatedResidualConfig;

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
import Compute.Observation;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Reads a sublayer's input from n residual streams and writes its output back into them.
     *
     *   N  = GroupedRmsNorm( S )                          stream norm, groups of H, unit-offset weight
     *   r  = sigmoid( up( silu( down( N ) / n ) ) )       per-element read gate
     *   x  = mean_over_streams( r * N )                   read(): the sublayer's input, H wide
     *   g  = 2 * sigmoid( inject( N ) / n )               one write gate per stream
     *   S' = S + concat_i( g_i * y )                      inject(): y is the sublayer's output
     *
     * The write adds to S, not to N. A read-only residual (GatedResidualConfig::withInjection( false )) ends the
     * network: it has no inject projection and inject() is refused.
     *
     * Children, by name: `.norm`, `.fc_down`, `.fc_up`, and `.fc_inject` when injecting -- the converter's names.
     * Inference-only.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GatedResidual : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using LinearType = Linear<TDeviceType, TPrecision>;

        explicit GatedResidual( const std::string& name, const GatedResidualConfig& config,
            std::optional<DeviceId> device_id = std::nullopt )
            : CompositeComponentBase( name ), config_( config )
        {
            config_.validate();

            createGraph();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "GatedResidual: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~GatedResidual() override = default;

        /**
         * @brief The sublayer's input, and the write gates inject() will use.
         *
         * @param stream [..., n * H], the residual state.
         * @return [..., H], valid until the next read().
         */
        TensorType& read( const TensorType& stream )
        {
            requireBuilt( "read" );

            const auto& stream_shape = stream.shape();

            validateStreamShape( stream_shape );
            resizeViews( stream_shape );

            auto& normed = norm_->forward( stream );
            auto& down = down_->forward( normed );

            operation_->activateDown( down, *activated_view_ );

            auto& up = up_->forward( *activated_view_ );

            operation_->mix( normed, up, *mixed_view_ );

            if ( inject_ )
            {
                auto& raw_weights = inject_->forward( normed );

                operation_->injectionWeights( raw_weights, *weights_view_ );
            }

            this->publish( ComputePass::Forward, "mixed", *mixed_view_ );

            return *mixed_view_;
        }

        /**
         * @brief The stream after the sublayer: S + g * y, with g from the last read() of the same @p stream.
         *
         * @param stream          [..., n * H], the state read() was given.
         * @param sublayer_output [..., H].
         * @return [..., n * H], valid until the next inject().
         */
        TensorType& inject( const TensorType& stream, const TensorType& sublayer_output )
        {
            requireBuilt( "inject" );

            if ( !inject_ )
            {
                throw std::logic_error( std::format(
                    "GatedResidual '{}': inject() on a read-only residual", this->getName() ) );
            }

            if ( stream.shape() != mixed_stream_shape_ )
            {
                throw std::invalid_argument( std::format(
                    "GatedResidual '{}': inject() must follow read() of the same stream shape", this->getName() ) );
            }

            operation_->inject( stream, *weights_view_, sublayer_output, *output_view_ );

            this->publish( ComputePass::Forward, "output", *output_view_ );

            return *output_view_;
        }

        /// The write gates of the last read(), [..., n].
        const TensorType& injectionWeights() const
        {
            requireBuilt( "injectionWeights" );

            if ( !inject_ )
            {
                throw std::logic_error( std::format(
                    "GatedResidual '{}': a read-only residual has no write gates", this->getName() ) );
            }

            return *weights_view_;
        }

        const ComponentType getType() const override
        {
            return ComponentType::GatedResidual;
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "mixed", ComputePassMask{ ComputePass::Forward } }, { "output", ComputePassMask{ ComputePass::Forward } } };
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
            {
                stats += child->getMemoryStats();
            }

            for ( const ITensor* tensor : { static_cast<const ITensor*>( activated_.get() ), static_cast<const ITensor*>( mixed_.get() ),
                                            static_cast<const ITensor*>( weights_.get() ), static_cast<const ITensor*>( output_.get() ) } )
            {
                if ( tensor )
                {
                    stats.device_state_bytes += occupiedTensorBytes( *tensor );
                }
            }

            return stats;
        }

        /// What onBuilding() would allocate for this context, without allocating: the children and four owned buffers.
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            const auto& stream_shape = context.inputShape();

            validateStreamShape( stream_shape );

            const std::string n = this->getName();

            MemoryStats stats;

            stats += this->template getComponentAs<RmsNormType>( n + ".norm" )->getRequiredMemory( context );
            stats += this->template getComponentAs<LinearType>( n + ".fc_down" )->getRequiredMemory( context );
            stats += this->template getComponentAs<LinearType>( n + ".fc_up" )->getRequiredMemory(
                context.withShape( widthShape( stream_shape, config_.getRank() ) ) );

            if ( config_.hasInjection() )
            {
                stats += this->template getComponentAs<LinearType>( n + ".fc_inject" )->getRequiredMemory( context );
            }

            const std::size_t granularity = context.getAllocationGranularity();

            for ( const dim_t width : ownedWidths() )
            {
                stats.device_state_bytes += occupiedDeviceBytes(
                    storageBytes<TPrecision>( elementCount( widthShape( stream_shape, width ) ) ), granularity );
            }

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "GatedResidual: " << this->getName() << std::endl;
            oss << "Streams: " << config_.getStreams() << " x " << config_.getModelDim()
                << ", rank " << config_.getRank() << ( config_.hasInjection() ? "" : ", read-only" ) << std::endl;

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
            const auto& stream_shape = context.inputShape();

            validateStreamShape( stream_shape );

            const std::string n = this->getName();
            auto device = this->getExecutionContext()->getDeviceId();

            norm_ = this->template getComponentAs<RmsNormType>( n + ".norm" );
            norm_->build( context );

            down_ = this->template getComponentAs<LinearType>( n + ".fc_down" );
            down_->build( context );

            up_ = this->template getComponentAs<LinearType>( n + ".fc_up" );
            up_->build( context.withShape( widthShape( stream_shape, config_.getRank() ) ) );

            if ( config_.hasInjection() )
            {
                inject_ = this->template getComponentAs<LinearType>( n + ".fc_inject" );
                inject_->build( context );
            }

            operation_->build( context );

            activated_ = std::make_shared<TensorType>( device, widthShape( stream_shape, config_.getRank() ), n + ".activated" );
            mixed_ = std::make_shared<TensorType>( device, widthShape( stream_shape, config_.getModelDim() ), n + ".mixed" );

            if ( config_.hasInjection() )
            {
                weights_ = std::make_shared<TensorType>( device, widthShape( stream_shape, config_.getStreams() ), n + ".weights" );
                output_ = std::make_shared<TensorType>( device, stream_shape, n + ".output" );
            }

            built_stream_shape_ = stream_shape;
            mixed_stream_shape_ = shape_t{};
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            for ( const auto& child : this->getComponents() )
            {
                child->setTrainingMode( training_mode );
            }
        }

    private:
        using OpType = typename OperationTraits<OperationType::GatedResidualOp, TDeviceType, TPrecision>::type;

        GatedResidualConfig config_;

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
        std::shared_ptr<OpType> operation_{ nullptr };

        std::shared_ptr<RmsNormType> norm_{ nullptr };
        std::shared_ptr<LinearType> down_{ nullptr };
        std::shared_ptr<LinearType> up_{ nullptr };
        std::shared_ptr<LinearType> inject_{ nullptr };

        std::shared_ptr<TensorType> activated_{ nullptr };
        std::shared_ptr<TensorType> mixed_{ nullptr };
        std::shared_ptr<TensorType> weights_{ nullptr };
        std::shared_ptr<TensorType> output_{ nullptr };

        std::optional<TensorType> activated_view_;
        std::optional<TensorType> mixed_view_;
        std::optional<TensorType> weights_view_;
        std::optional<TensorType> output_view_;

        shape_t built_stream_shape_{};
        shape_t mixed_stream_shape_{};

        void createGraph()
        {
            const std::string n = this->getName();
            const dim_t stream_width = config_.getStreamWidth();

            this->addComponent( std::make_shared<RmsNormType>( n + ".norm",
                RmsNormConfig( shape_t{ stream_width } )
                    .withEpsilon( config_.getEpsilon() )
                    .withBias( false )
                    .withUnitOffset( 1.0f )
                    .withGroupSize( config_.getModelDim() ) ) );

            this->addComponent( std::make_shared<LinearType>( n + ".fc_down",
                LinearConfig( stream_width, config_.getRank() ).withBias( false ) ) );

            this->addComponent( std::make_shared<LinearType>( n + ".fc_up",
                LinearConfig( config_.getRank(), stream_width ).withBias( false ) ) );

            if ( config_.hasInjection() )
            {
                this->addComponent( std::make_shared<LinearType>( n + ".fc_inject",
                    LinearConfig( stream_width, config_.getStreams() ).withBias( false ) ) );
            }
        }

        // The widths of the owned buffers onBuilding() allocates, in its order.
        std::vector<dim_t> ownedWidths() const
        {
            if ( config_.hasInjection() )
            {
                return { config_.getRank(), config_.getModelDim(), config_.getStreams(), config_.getStreamWidth() };
            }

            return { config_.getRank(), config_.getModelDim() };
        }

        static shape_t widthShape( const shape_t& stream_shape, dim_t width )
        {
            shape_t shape = stream_shape;
            shape.back() = width;

            return shape;
        }

        void resizeViews( const shape_t& stream_shape )
        {
            if ( stream_shape == mixed_stream_shape_ )
            {
                return;
            }

            if ( elementCount( stream_shape ) > elementCount( built_stream_shape_ ) )
            {
                throw std::invalid_argument( std::format(
                    "GatedResidual '{}': stream is larger than the built shape", this->getName() ) );
            }

            activated_view_.emplace( activated_->view( widthShape( stream_shape, config_.getRank() ) ) );
            mixed_view_.emplace( mixed_->view( widthShape( stream_shape, config_.getModelDim() ) ) );

            if ( config_.hasInjection() )
            {
                weights_view_.emplace( weights_->view( widthShape( stream_shape, config_.getStreams() ) ) );
                output_view_.emplace( output_->view( stream_shape ) );
            }

            mixed_stream_shape_ = stream_shape;
        }

        void requireBuilt( const char* caller ) const
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( std::format( "GatedResidual '{}': {} before build", this->getName(), caller ) );
            }
        }

        void validateStreamShape( const shape_t& stream_shape ) const
        {
            if ( stream_shape.empty() || stream_shape.back() != config_.getStreamWidth() )
            {
                throw std::invalid_argument( std::format(
                    "GatedResidual '{}': stream must be [..., {}]", this->getName(), config_.getStreamWidth() ) );
            }
        }
    };
}
