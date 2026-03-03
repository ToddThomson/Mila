/**
 * @file Llama.Block.ixx
 * @brief LLaMA transformer block partition for LlamaTransformer.
 *
 * Device-templated composite block using RMSNorm, attention, residuals and MLP.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <optional>

export module Dnn.Components.LlamaTransformer:Block;
import :Config;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorOps;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.CompositeComponent;
import Dnn.ActivationType;
import Compute.Precision;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ExecutionContextFactory;
import Dnn.Components.RmsNorm;
import Dnn.Components.MultiHeadAttention;
import Dnn.Components.Residual;
import Dnn.Components.Linear;
import Dnn.Components.MLP;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LlamaBlock : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using AttentionType = MultiHeadAttention<TDeviceType, TPrecision>;
        using ResidualType = Residual<TDeviceType, TPrecision>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using MLPType = MLP<TDeviceType, TPrecision>;

        explicit LlamaBlock( const std::string& name, const LlamaConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : CompositeComponentBase( name ), config_( config )
        {
            config_.validate();

            createGraph();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "LlamaBlock: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~LlamaBlock() override = default;

        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "LlamaBlock must be built before forward()." );
            }

            auto& rms1_out = rms1_->forward( input );
            this->getExecutionContext()->synchronize();

            auto& qkv_out = qkv_proj_->forward( rms1_out );
            this->getExecutionContext()->synchronize();

            auto& attn_out = attn_->forward( qkv_out );
            this->getExecutionContext()->synchronize();

            auto& out_proj = out_proj_->forward( attn_out );
            this->getExecutionContext()->synchronize();

            auto& res1_out = res1_->forward( input, out_proj );
            this->getExecutionContext()->synchronize();

            auto& rms2_out = rms2_->forward( res1_out );
            this->getExecutionContext()->synchronize();

            auto& ffn_out = ffn_->forward( rms2_out );
            this->getExecutionContext()->synchronize();

            auto& res2_out = res2_->forward( res1_out, ffn_out );
            this->getExecutionContext()->synchronize();

            last_rms1_out_ = &rms1_out;
            last_qkv_out_ = &qkv_out;
            last_attn_out_ = &attn_out;
            last_out_proj_out_ = &out_proj;
            last_res1_out_ = &res1_out;
            last_rms2_out_ = &rms2_out;
            last_ffn_out_ = &ffn_out;
            last_res2_out_ = &res2_out;

            forward_executed_ = this->isTraining();

            return res2_out;
        }

        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "LlamaBlock must be built before backward()." );

            if ( !this->isTraining() )
                throw std::runtime_error( "LlamaBlock must be in training mode to call backward(). Call setTraining(true) first." );

            if ( !forward_executed_ )
                throw std::runtime_error( "LlamaBlock::backward: a training-mode forward() must be executed before backward()." );

            auto [d_res1_from_res2, d_ffn_from_res2] =
                res2_->backward( *last_res1_out_, *last_ffn_out_, output_grad );

            auto& d_rms2_from_ffn = ffn_->backward( *last_rms2_out_, d_ffn_from_res2 );
            auto& d_res1_from_rms2 = rms2_->backward( *last_res1_out_, d_rms2_from_ffn );

            zero( *d_res1_accum_ );
            add( d_res1_from_res2, d_res1_from_rms2, *d_res1_accum_ );

            auto [d_input_from_res1, d_out_proj_from_res1] =
                res1_->backward( input, *last_out_proj_out_, *d_res1_accum_ );

            auto& d_attn_from_out_proj = out_proj_->backward( *last_attn_out_, d_out_proj_from_res1 );
            this->getExecutionContext()->synchronize();

            auto& d_qkv = attn_->backward( *last_qkv_out_, d_attn_from_out_proj );
            auto& d_rms1 = qkv_proj_->backward( *last_rms1_out_, d_qkv );
            auto& d_input_from_rms1 = rms1_->backward( input, d_rms1 );

            zero( *d_input_ );
            add( d_input_from_res1, d_input_from_rms1, *d_input_ );

            forward_executed_ = false;

            return *d_input_;
        }

        TensorType& decode( const TensorType& input, int position )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "LlamaBlock must be built before decode()." );

            auto& rms1_out = rms1_->forward( input );
            this->getExecutionContext()->synchronize();

            auto& qkv_out = qkv_proj_->decode( rms1_out );
            this->getExecutionContext()->synchronize();

            auto& attn_out = attn_->decode( qkv_out, position );
            this->getExecutionContext()->synchronize();

            auto& out_proj = out_proj_->decode( attn_out );
            this->getExecutionContext()->synchronize();

            auto& res1_out = res1_->forward( input, out_proj );
            this->getExecutionContext()->synchronize();

            auto& rms2_out = rms2_->forward( res1_out );
            this->getExecutionContext()->synchronize();

            auto& ffn_out = ffn_->decode( rms2_out );
            this->getExecutionContext()->synchronize();

            auto& res2_out = res2_->forward( res1_out, ffn_out );
            this->getExecutionContext()->synchronize();

            return res2_out;
        }

        bool supportsKVCache() const noexcept
        {
            return attn_ && attn_->supportsKVCache();
        }

        void initializeKVCache( int64_t max_seq_len )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "LlamaBlock must be built before initializeKVCache()." );

            attn_->initializeKVCache( max_seq_len );
        }

        void resetKVCache()
        {
            attn_->resetKVCache();
        }

        void zeroGradients() override
        {
            if ( d_res1_accum_ ) zero( *d_res1_accum_ );
            if ( d_input_ )      zero( *d_input_ );

            attn_->zeroGradients();
            qkv_proj_->zeroGradients();
            out_proj_->zeroGradients();
            rms1_->zeroGradients();
            rms2_->zeroGradients();
            res1_->zeroGradients();
            res2_->zeroGradients();
            ffn_->zeroGradients();
        }

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            attn_->save_( archive, mode );
            rms1_->save_( archive, mode );
            rms2_->save_( archive, mode );
            qkv_proj_->save_( archive, mode );
            out_proj_->save_( archive, mode );
            res1_->save_( archive, mode );
            res2_->save_( archive, mode );
            ffn_->save_( archive, mode );
        }

        void load_( ModelArchive& archive, SerializationMode mode )
        {
            attn_->load_( archive, mode );
            rms1_->load_( archive, mode );
            rms2_->load_( archive, mode );
            qkv_proj_->load_( archive, mode );
            out_proj_->load_( archive, mode );
            res1_->load_( archive, mode );
            res2_->load_( archive, mode );
            ffn_->load_( archive, mode );
        }

        const ComponentType getType() const override
        {
            return ComponentType::Transformer;
        }

    protected:
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            cached_input_shape_ = input_shape;

            rms1_ = this->template getComponentAs<RmsNormType>( this->getName() + ".rms_1" );
            rms1_->build( input_shape );

            qkv_proj_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_qkv_proj" );
            qkv_proj_->build( input_shape );

            out_proj_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_out_proj" );
            out_proj_->build( input_shape );

            shape_t qkv_shape = input_shape;
            qkv_shape.back() = static_cast<int64_t>(config_.getModelDim() * 3);

            attn_ = this->template getComponentAs<AttentionType>( this->getName() + ".attn" );
            attn_->build( qkv_shape );

            rms2_ = this->template getComponentAs<RmsNormType>( this->getName() + ".rms_2" );
            rms2_->build( input_shape );

            res1_ = this->template getComponentAs<ResidualType>( this->getName() + ".res_1" );
            res1_->build( input_shape );

            res2_ = this->template getComponentAs<ResidualType>( this->getName() + ".res_2" );
            res2_->build( input_shape );

            ffn_ = this->template getComponentAs<MLPType>( this->getName() + ".mlp" );
            ffn_->build( input_shape );

            auto device = this->getDeviceId();

            d_res1_accum_ = std::make_shared<TensorType>( device, input_shape );
            d_res1_accum_->setName( this->getName() + ".d_res1_accum" );
            zero( *d_res1_accum_ );

            d_input_ = std::make_shared<TensorType>( device, input_shape );
            d_input_->setName( this->getName() + ".d_input" );
            zero( *d_input_ );
        }

        void onTrainingChanging( bool is_training ) override
        {
            if ( attn_ )     attn_->setTraining( is_training );
            if ( rms1_ )     rms1_->setTraining( is_training );
            if ( rms2_ )     rms2_->setTraining( is_training );
            if ( qkv_proj_ ) qkv_proj_->setTraining( is_training );
            if ( out_proj_ ) out_proj_->setTraining( is_training );
            if ( res1_ )     res1_->setTraining( is_training );
            if ( res2_ )     res2_->setTraining( is_training );
            if ( ffn_ )      ffn_->setTraining( is_training );

            forward_executed_ = false;
        }

    private:
        LlamaConfig config_;
        shape_t cached_input_shape_;
        bool forward_executed_{ false };

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        std::shared_ptr<AttentionType> attn_{ nullptr };
        std::shared_ptr<RmsNormType> rms1_{ nullptr };
        std::shared_ptr<RmsNormType> rms2_{ nullptr };
        std::shared_ptr<LinearType> qkv_proj_{ nullptr };
        std::shared_ptr<LinearType> out_proj_{ nullptr };
        std::shared_ptr<ResidualType> res1_{ nullptr };
        std::shared_ptr<ResidualType> res2_{ nullptr };
        std::shared_ptr<MLPType> ffn_{ nullptr };

        std::shared_ptr<TensorType> d_res1_accum_{ nullptr };
        std::shared_ptr<TensorType> d_input_{ nullptr };

        TensorType* last_rms1_out_{ nullptr };
        TensorType* last_qkv_out_{ nullptr };
        TensorType* last_attn_out_{ nullptr };
        TensorType* last_out_proj_out_{ nullptr };
        TensorType* last_res1_out_{ nullptr };
        TensorType* last_rms2_out_{ nullptr };
        TensorType* last_ffn_out_{ nullptr };
        TensorType* last_res2_out_{ nullptr };

        void createGraph()
        {
            auto attn_cfg = MultiHeadAttentionConfig( config_.getModelDim(), config_.getNumHeads() );
            this->addComponent( std::make_shared<AttentionType>( this->getName() + ".attn", attn_cfg, std::nullopt ) );

            auto rms1_cfg = RmsNormConfig().withNormalizedShape( shape_t{ static_cast<int64_t>(config_.getModelDim()) } );
            this->addComponent( std::make_shared<RmsNormType>( this->getName() + ".rms_1", rms1_cfg, std::nullopt ) );

            auto rms2_cfg = RmsNormConfig().withNormalizedShape( shape_t{ static_cast<int64_t>(config_.getModelDim()) } );
            this->addComponent( std::make_shared<RmsNormType>( this->getName() + ".rms_2", rms2_cfg, std::nullopt ) );

            auto qkv_cfg = LinearConfig( static_cast<dim_t>(config_.getModelDim()), static_cast<dim_t>(config_.getModelDim() * 3) );
            qkv_cfg.withBias( config_.useBias() );
            this->addComponent( std::make_shared<LinearType>( this->getName() + ".fc_qkv_proj", qkv_cfg, std::nullopt ) );

            auto out_proj_cfg = LinearConfig( static_cast<dim_t>(config_.getModelDim()), static_cast<dim_t>(config_.getModelDim()) );
            out_proj_cfg.withBias( config_.useBias() );
            this->addComponent( std::make_shared<LinearType>( this->getName() + ".fc_out_proj", out_proj_cfg, std::nullopt ) );

            ResidualConfig res_cfg1;
            res_cfg1.withScalingFactor( config_.getResidualScale() );
            this->addComponent( std::make_shared<ResidualType>( this->getName() + ".res_1", res_cfg1, std::nullopt ) );

            ResidualConfig res_cfg2;
            res_cfg2.withScalingFactor( config_.getResidualScale() );
            this->addComponent( std::make_shared<ResidualType>( this->getName() + ".res_2", res_cfg2, std::nullopt ) );

            dim_t hidden_dim = static_cast<dim_t>(config_.getHiddenSize());
            if ( hidden_dim == 0 )
            {
                hidden_dim = static_cast<dim_t>(config_.getModelDim() * 4);
            }

            auto mlp_cfg = MLPConfig( static_cast<dim_t>(config_.getModelDim()), hidden_dim );
            mlp_cfg.withBias( config_.useBias() )
                .withActivation( config_.getActivationType() );
            this->addComponent( std::make_shared<MLPType>( this->getName() + ".mlp", mlp_cfg, std::nullopt ) );
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 3 )
            {
                throw std::invalid_argument( "LlamaBlock: input must have model-layout shape [B, T, embedding_dim]" );
            }

            int64_t trailing = input_shape.back();
            if ( trailing != static_cast<int64_t>(config_.getModelDim()) )
            {
                std::ostringstream oss;
                oss << "LlamaBlock: model dimension mismatch. Config says "
                    << config_.getModelDim() << " got " << trailing;
                throw std::invalid_argument( oss.str() );
            }
        }
    };
}