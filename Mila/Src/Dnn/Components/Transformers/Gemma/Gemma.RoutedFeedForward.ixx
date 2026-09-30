/**
 * @file Gemma.RoutedFeedForward.ixx
 * @brief Gemma 4's mixture-of-experts feed-forward sublayer: a dense GeGLU branch beside a routed expert bank, as
 *        one component named ffn.
 */

module;
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

export module Dnn.Components.GemmaRoutedFeedForward;

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
import Dnn.Components.Residual;
import Dnn.Components.GatedMLP;
import Dnn.Components.Router;
import Dnn.Components.MixtureOfExperts;
import Dnn.Components.GemmaConfig;
import Dnn.Components.GemmaBlockWorkspace;
import Dnn.Quantization.Weight.Policies;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief The feed-forward sublayer of a mixture-of-experts Gemma 4 block (the 26B-A4B).
     *
     * The dense branch normalizes the residual and runs the GeGLU `mlp`; the routed branch routes on the residual
     * itself and runs the selected experts on `experts_pre_norm` of it (Gemma4MoE.md Phase 1). Each branch has its
     * own post-norm, the two are summed, and `post_norm` closes the sublayer. Constrained to the weight policies
     * the expert bank implements, so a policy it would refuse at construction cannot be instantiated.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision, WeightQuantPolicy TWeightQuantization = NoWeightQuant>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType> && expertBankImplements<TWeightQuantization>
    class GemmaRoutedFeedForward : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using ResidualType = Residual<TDeviceType, TPrecision>;
        using MlpType = GatedMLP<TDeviceType, TPrecision, ActivationType::Gelu, TWeightQuantization>;
        using RouterType = Router<TDeviceType, TPrecision>;
        using ExpertsType = MixtureOfExperts<TDeviceType, TPrecision, ActivationType::Gelu, TWeightQuantization>;
        using WorkspaceType = GemmaBlockWorkspace<TDeviceType, TPrecision>;

        GemmaRoutedFeedForward( const std::string& name, const GemmaConfig& config )
            : CompositeComponentBase( name ), config_( config )
        {
            createGraph();
        }

        TensorType& forward( const TensorType& residual )
        {
            auto& dense_input = pre_norm_->forward( residual );

            // decode() is GatedMLP's capture-free inference path, so it serves prefill too.
            auto& dense = mlp_->decode( dense_input );
            auto& dense_normed = dense_post_norm_->forward( dense );

            auto routing = router_->forward( residual );
            auto& expert_input = experts_pre_norm_->forward( residual );
            auto& routed = experts_->forward( expert_input, routing.weights, routing.indices );
            auto& routed_normed = experts_post_norm_->forward( routed );

            auto& combined = sum_->forward( dense_normed, routed_normed );

            return post_norm_->forward( combined );
        }

        /**
         * @brief Route the block workspace's feed-forward slots into the children; before build().
         *
         * The router and the expert bank allocate their own outputs.
         */
        void installSharedWorkspace( const WorkspaceType& workspace )
        {
            if ( this->isBuilt() )
                throw std::logic_error( "GemmaRoutedFeedForward::installSharedWorkspace: must be called before build()" );

            norm( "pre_norm" )->installSharedOutput( workspace.ffn_in );
            mlp()->installSharedOutputs( workspace.gate_up, workspace.ffn_act, workspace.ffn_down );
            norm( "dense_post_norm" )->installSharedOutput( workspace.ffn_dense_normed );
            norm( "experts_pre_norm" )->installSharedOutput( workspace.ffn_expert_in );
            norm( "experts_post_norm" )->installSharedOutput( workspace.ffn_expert_normed );
            sum()->installSharedOutput( workspace.ffn_sum );
            norm( "post_norm" )->installSharedOutput( workspace.ffn_normed );
        }

        const ComponentType getType() const override
        {
            return ComponentType::MixtureOfExperts;
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
         * The router and the expert bank always allocate their own outputs, so they are asked without the pooled
         * declaration that the norms, the mlp and the sum honour.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            const BuildContext unpooled = context.withInstalledOutput( false );

            MemoryStats stats;

            stats += norm( "pre_norm" )->getRequiredMemory( context );
            stats += mlp()->getRequiredMemory( context );
            stats += norm( "dense_post_norm" )->getRequiredMemory( context );
            stats += router()->getRequiredMemory( unpooled );
            stats += norm( "experts_pre_norm" )->getRequiredMemory( context );
            stats += experts()->getRequiredMemory( unpooled );
            stats += norm( "experts_post_norm" )->getRequiredMemory( context );
            stats += sum()->getRequiredMemory( context );
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

            dense_post_norm_ = norm( "dense_post_norm" );
            dense_post_norm_->build( context );

            router_ = router();
            router_->build( context );

            experts_pre_norm_ = norm( "experts_pre_norm" );
            experts_pre_norm_->build( context );

            experts_ = experts();
            experts_->build( context );

            experts_post_norm_ = norm( "experts_post_norm" );
            experts_post_norm_->build( context );

            sum_ = sum();
            sum_->build( context );

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
        std::shared_ptr<RmsNormType> dense_post_norm_{ nullptr };
        std::shared_ptr<RouterType> router_{ nullptr };
        std::shared_ptr<RmsNormType> experts_pre_norm_{ nullptr };
        std::shared_ptr<ExpertsType> experts_{ nullptr };
        std::shared_ptr<RmsNormType> experts_post_norm_{ nullptr };
        std::shared_ptr<ResidualType> sum_{ nullptr };
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

        std::shared_ptr<RouterType> router() const
        {
            return this->template getComponentAs<RouterType>( this->getName() + ".router" );
        }

        std::shared_ptr<ExpertsType> experts() const
        {
            return this->template getComponentAs<ExpertsType>( this->getName() + ".experts" );
        }

        std::shared_ptr<ResidualType> sum() const
        {
            return this->template getComponentAs<ResidualType>( this->getName() + ".sum" );
        }

        void createGraph()
        {
            const std::string n = this->getName();
            const dim_t model_dim = config_.getModelDim();
            const float epsilon = config_.getRMSNormEpsilon();
            const auto norm_config = RmsNormConfig( shape_t{ model_dim } ).withEpsilon( epsilon ).withBias( false );

            this->addComponent( std::make_shared<RmsNormType>( n + ".pre_norm", norm_config ) );
            this->addComponent( std::make_shared<MlpType>( n + ".mlp",
                GatedMLPConfig( model_dim, config_.getHiddenDimension() ).withGateActivation( ActivationType::Gelu ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".dense_post_norm", norm_config ) );
            this->addComponent( std::make_shared<RouterType>( n + ".router",
                RouterConfig( model_dim, config_.getNumExperts(), config_.getTopKExperts() ).withEpsilon( epsilon ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".experts_pre_norm", norm_config ) );
            this->addComponent( std::make_shared<ExpertsType>( n + ".experts", MixtureOfExpertsConfig(
                model_dim, config_.getExpertHiddenDimension(), config_.getNumExperts(), config_.getTopKExperts() ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".experts_post_norm", norm_config ) );
            this->addComponent( std::make_shared<ResidualType>( n + ".sum", ResidualConfig{} ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".post_norm", norm_config ) );
        }
    };
}
