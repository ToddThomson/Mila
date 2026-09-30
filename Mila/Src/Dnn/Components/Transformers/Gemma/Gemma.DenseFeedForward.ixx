/**
 * @file Gemma.DenseFeedForward.ixx
 * @brief Gemma 4's dense feed-forward sublayer: pre_norm -> GeGLU mlp -> post_norm, as one component named ffn.
 */

module;
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

export module Dnn.Components.GemmaDenseFeedForward;

import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ActivationType;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.CompositeComponent;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Dnn.Components.RmsNorm;
import Dnn.Components.GatedMLP;
import Dnn.Components.GemmaConfig;
import Dnn.Components.GemmaBlockWorkspace;
import Dnn.Quantization.Weight.Policies;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief The feed-forward sublayer of a dense Gemma 4 block (the 12B).
     *
     * Reads the post-attention residual and returns post_norm( mlp( pre_norm( residual ) ) ), which the block adds
     * back to the residual. Children: `pre_norm`, `mlp` (GatedMLP with a GELU gate), `post_norm`.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision, WeightQuantPolicy TWeightQuantization = NoWeightQuant>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GemmaDenseFeedForward : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using MlpType = GatedMLP<TDeviceType, TPrecision, ActivationType::Gelu, TWeightQuantization>;
        using WorkspaceType = GemmaBlockWorkspace<TDeviceType, TPrecision>;

        GemmaDenseFeedForward( const std::string& name, const GemmaConfig& config )
            : CompositeComponentBase( name ), config_( config )
        {
            createGraph();
        }

        TensorType& forward( const TensorType& residual )
        {
            auto& input = pre_norm_->forward( residual );

            // decode() is GatedMLP's capture-free inference path, so it serves prefill too.
            auto& dense = mlp_->decode( input );

            return post_norm_->forward( dense );
        }

        /**
         * @brief Route the block workspace's feed-forward slots into the children; before build().
         */
        void installSharedWorkspace( const WorkspaceType& workspace )
        {
            if ( this->isBuilt() )
                throw std::logic_error( "GemmaDenseFeedForward::installSharedWorkspace: must be called before build()" );

            norm( "pre_norm" )->installSharedOutput( workspace.ffn_in );
            mlp()->installSharedOutputs( workspace.gate_up, workspace.ffn_act, workspace.ffn_down );
            norm( "post_norm" )->installSharedOutput( workspace.ffn_normed );
        }

        const ComponentType getType() const override
        {
            return ComponentType::GatedMlp;
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
                stats += child->getMemoryStats();

            return stats;
        }

        /**
         * @brief What onBuilding() would allocate for this stream-width context, without allocating.
         *
         * The context carries the caller's pooling declaration, which every child honours.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            MemoryStats stats;

            stats += norm( "pre_norm" )->getRequiredMemory( context );
            stats += mlp()->getRequiredMemory( context );
            stats += norm( "post_norm" )->getRequiredMemory( context );

            return stats;
        }

    protected:

        void onBuilding( const BuildContext& context ) override
        {
            pre_norm_ = norm( "pre_norm" );
            pre_norm_->build( context );

            mlp_ = mlp();
            mlp_->build( context );

            post_norm_ = norm( "post_norm" );
            post_norm_->build( context );
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            for ( const auto& child : this->getComponents() )
                child->setTrainingMode( training_mode );
        }

    private:
        GemmaConfig config_;

        std::shared_ptr<RmsNormType> pre_norm_{ nullptr };
        std::shared_ptr<MlpType> mlp_{ nullptr };
        std::shared_ptr<RmsNormType> post_norm_{ nullptr };

        // Children by name: the members are assigned in onBuilding() and are null before it.
        std::shared_ptr<RmsNormType> norm( const std::string& suffix ) const
        {
            return this->template getComponentAs<RmsNormType>( this->getName() + "." + suffix );
        }

        std::shared_ptr<MlpType> mlp() const
        {
            return this->template getComponentAs<MlpType>( this->getName() + ".mlp" );
        }

        void createGraph()
        {
            const std::string n = this->getName();
            const dim_t model_dim = config_.getModelDim();
            const auto norm_config = RmsNormConfig( shape_t{ model_dim } )
                .withEpsilon( config_.getRMSNormEpsilon() ).withBias( false );

            this->addComponent( std::make_shared<RmsNormType>( n + ".pre_norm", norm_config ) );
            this->addComponent( std::make_shared<MlpType>( n + ".mlp",
                GatedMLPConfig( model_dim, config_.getHiddenDimension() ).withGateActivation( ActivationType::Gelu ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".post_norm", norm_config ) );
        }
    };
}
