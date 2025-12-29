/**
 * @file Transformer.ixx
 * @brief Transformer encoder block implementation.
 *
 * Provides a device-templated composite Transformer encoder block that
 * composes attention, layer-norm, residual connections and MLP sub-components.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <optional>

export module Dnn.Blocks.Transformer;
export import :Config;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Component;
import Dnn.CompositeComponent;
import Dnn.ActivationType;
import Compute.Precision;
import Compute.MemoryResource;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.IExecutionContext;
import Compute.ExecutionContextFactory;
import Dnn.Components.LayerNorm;
import Dnn.Components.Attention;
import Dnn.Components.Residual;
import Dnn.Components.Linear;
import Dnn.Blocks.MLP;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Transformer encoder block as a composite component.
     *
     * Device-templated composite component that composes:
     *   LayerNorm -> QKV projection -> MultiHeadSelfAttention -> Residual ->
     *   LayerNorm -> MLP -> Residual
     *
     * Construction follows the same patterns used by the `MLP` block:
     * - Graph created in constructor (context-independent)
     * - Optional owned ExecutionContext when a DeviceId is provided
     * - Shape-dependent build occurs in onBuilding()
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Transformer : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using ComponentPtr = typename CompositeComponentBase::ComponentPtr;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;
        using LayerNormType = LayerNorm<TDeviceType, TPrecision>;
        using AttentionType = Attention<TDeviceType, TPrecision>;
        using ResidualType = Residual<TDeviceType, TPrecision>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using MLPType = MLP<TDeviceType, TPrecision>;

        /**
         * @brief Construct Transformer in shared or standalone mode.
         *
         * - name: component name (used as prefix for sub-components)
         * - config: validated configuration for the Transformer
         * - device_id (optional): when provided and matching TDeviceType, an owned
         *   ExecutionContext is created and bound (standalone mode). Otherwise the
         *   component expects a parent to set an ExecutionContext (shared mode).
         */
        explicit Transformer( const std::string& name, const TransformerConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : CompositeComponentBase( name ), config_( config )
        {
            config_.validate();

            createGraph();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "Transformer: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );

                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~Transformer() override = default;

        // ====================================================================
        // Forward and backward dispatch (hot path)
        // ====================================================================

        void forward( const ITensor& input, ITensor& output )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Transformer must be built before forward()." );
            }

            // If input is a concrete Tensor on this device/precision, print detailed representation.
            /*if ( const auto* in_t = dynamic_cast<const TensorType*>(&input) )
            {
                std::clog << this->getName() << ": input tensor:\n" << in_t->toString( true ) << std::endl;
            }*/

            ln1_->forward( input, *ln1_output_ );
            /*if ( ln1_output_ )
            {
                std::clog << this->getName() << ": ln1_output:\n" << ln1_output_->toString( true ) << std::endl;
            }*/
            
            qkv_proj_->forward( *ln1_output_, *qkv_output_ );
            /*if ( qkv_output_ )
            {
                std::clog << this->getName() << ": qkv_output:\n" << qkv_output_->toString( true ) << std::endl;
            }*/
            
            attn_->forward( *qkv_output_, *attn_output_ );
            /*if ( attn_output_ )
            {
                std::clog << this->getName() << ": attn_output:\n" << attn_output_->toString( true ) << std::endl;
            }*/

            res1_->forward( input, *attn_output_, *res1_output_ );
            /*if ( res1_output_ )
            {
                std::clog << this->getName() << ": res1_output:\n" << res1_output_->toString( true ) << std::endl;
            }*/

            ln2_->forward( *res1_output_, *ln2_output_ );
            /*if ( ln2_output_ )
            {
                std::clog << this->getName() << ": ln2_output:\n" << ln2_output_->toString( true ) << std::endl;
            }*/

            ffn_->forward( *ln2_output_, *ffn_output_ );

            res2_->forward( *res1_output_, *ffn_output_, output );
            // If output is a concrete Tensor on this device/precision, print detailed representation.
            /*if ( auto* out_t = dynamic_cast<TensorType*>(&output) )
            {
                std::clog << this->getName() << ": output:\n" << out_t->toString( true ) << std::endl;
            }*/

            forward_executed_ = this->isTraining();
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Transformer must be built before backward()." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "Transformer must be in training mode to call backward(). Call setTraining(true) first." );
            }

            if ( !forward_executed_ )
            {
                throw std::runtime_error( "Transformer::backward: a training-mode forward() must be executed before backward()" );
            }

            // Gradient buffer usage (5 pre-allocated buffers):
            //   d_act_1_: reused for d_ffn -> d_ln2 -> d_attn -> d_ln1
            //   d_act_2_: reused for d_res1_from_ln2 -> d_in_from_ln1
            //   d_act_3_: holds d_res1 / d_res1_total / d_in_from_res (residual path)
            //   d_act_4_: scratch for gradient accumulation
            //   d_qkv_:   QKV projection gradients (3x embedding dim)

            copy( static_cast<const TensorType&>(output_grad), *d_act_3_ );
            copy( static_cast<const TensorType&>(output_grad), *d_act_1_ );

            ffn_->backward( *ln2_output_, *d_act_1_, *d_act_2_ );

            ln2_->backward( *res1_output_, *d_act_2_, *d_act_4_ );

            res1_->backward( *d_act_3_, *d_act_4_, *d_act_3_ );

            copy( *d_act_3_, *d_act_1_ );

            attn_->backward( *qkv_output_, *d_act_1_, *d_qkv_ );

            qkv_proj_->backward( *ln1_output_, *d_qkv_, *d_act_1_ );

            ln1_->backward( input, *d_act_1_, *d_act_2_ );

            res1_->backward( *d_act_3_, *d_act_2_, input_grad );

            forward_executed_ = false;
        }

        // ====================================================================
        // Serialization (follow MLP style)
        // ====================================================================

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            attn_->save_( archive, mode );
            ln1_->save_( archive, mode );
            ln2_->save_( archive, mode );
            qkv_proj_->save_( archive, mode );
            res1_->save_( archive, mode );
            res2_->save_( archive, mode );
            ffn_->save_( archive, mode );
        }

        void load_( ModelArchive& archive, SerializationMode mode )
        {
            attn_->load_( archive, mode );
            ln1_->load_( archive, mode );
            ln2_->load_( archive, mode );
            qkv_proj_->load_( archive, mode );
            res1_->load_( archive, mode );
            res2_->load_( archive, mode );
            ffn_->load_( archive, mode );
        }

        // ====================================================================
        // Diagnostics
        // ====================================================================

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "====================" << std::endl;
            oss << "Transformer: " << this->getName() << std::endl;

            if ( !cached_input_shape_.empty() )
            {
                oss << "Input shape: (";
                for ( size_t i = 0; i < cached_input_shape_.size(); ++i )
                {
                    oss << cached_input_shape_[ i ];
                    if ( i != cached_input_shape_.size() - 1 )
                        oss << ", ";
                }
                oss << ")" << std::endl;
            }

            oss << "Number of heads: " << config_.getNumHeads() << std::endl;
            oss << "MLP hidden dimension: " << config_.getHiddenDimension() << std::endl;
            oss << "Architecture: Pre-LN" << std::endl;

            if ( this->hasExecutionContext() )
            {
                oss << "Device: " << this->getDeviceId().toString() << std::endl;
            }
            else
            {
                oss << "Device: (context not set)" << std::endl;
            }

            if ( this->isBuilt() )
            {
                oss << "Parameter count: " << this->parameterCount() << std::endl;
            }

            return oss.str();
        }

    protected:

        /**
         * @brief Build hook: validate shape, bind children and allocate buffers.
         *
         * Mirrors the approach used by `MLP::onBuilding`.
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            cached_input_shape_ = input_shape;

            ln1_ = this->template getComponentAs<LayerNormType>( this->getName() + ".lnorm_1" );
            ln1_->build( input_shape );

            qkv_proj_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_qkv_proj" );
            qkv_proj_->build( input_shape );

            shape_t qkv_shape = input_shape;
            qkv_shape.back() = static_cast<int64_t>(config_.getEmbeddingDim() * 3);

            attn_ = this->template getComponentAs<AttentionType>( this->getName() + ".attn" );
            attn_->build( qkv_shape );

            ln2_ = this->template getComponentAs<LayerNormType>( this->getName() + ".lnorm_2" );
            ln2_->build( input_shape );

            res1_ = this->template getComponentAs<ResidualType>( this->getName() + ".res_1" );
            res1_->build( input_shape );

            res2_ = this->template getComponentAs<ResidualType>( this->getName() + ".res_2" );
            res2_->build( input_shape );

            ffn_ = this->template getComponentAs<MLPType>( this->getName() + ".mlp" );
            ffn_->build( input_shape );

            auto device = this->getDeviceId();

            // Forward activation buffers
            ln1_output_ = std::make_shared<TensorType>( device, input_shape );
            ln1_output_->setName( this->getName() + ".lnorm_1_output" );

            qkv_output_ = std::make_shared<TensorType>( device, qkv_shape );
            qkv_output_->setName( this->getName() + ".qkv_output" );

            attn_output_ = std::make_shared<TensorType>( device, input_shape );
            attn_output_->setName( this->getName() + ".attn_output" );

            res1_output_ = std::make_shared<TensorType>( device, input_shape );
            res1_output_->setName( this->getName() + ".res_1_output" );

            ln2_output_ = std::make_shared<TensorType>( device, input_shape );
            ln2_output_->setName( this->getName() + ".lnorm_2_output" );

            ffn_output_ = std::make_shared<TensorType>( device, input_shape );
            ffn_output_->setName( this->getName() + ".ffn_output" );

            res2_output_ = std::make_shared<TensorType>( device, input_shape );
            res2_output_->setName( this->getName() + ".res_2_output" );

            // Backward gradient buffers (pre-allocated for reuse)
            d_act_1_ = std::make_shared<TensorType>( device, input_shape );
            d_act_1_->setName( this->getName() + ".d_act_1" );

            d_act_2_ = std::make_shared<TensorType>( device, input_shape );
            d_act_2_->setName( this->getName() + ".d_act_2" );

            d_act_3_ = std::make_shared<TensorType>( device, input_shape );
            d_act_3_->setName( this->getName() + ".d_act_3" );

            d_act_4_ = std::make_shared<TensorType>( device, input_shape );
            d_act_4_->setName( this->getName() + ".d_act_4" );

            d_qkv_ = std::make_shared<TensorType>( device, qkv_shape );
            d_qkv_->setName( this->getName() + ".d_qkv" );
        }

        void onTrainingChanging( bool is_training ) override
        {
            if ( attn_ )     attn_->setTraining( is_training );
            if ( ln1_ )      ln1_->setTraining( is_training );
            if ( ln2_ )      ln2_->setTraining( is_training );
            if ( qkv_proj_ ) qkv_proj_->setTraining( is_training );
            if ( res1_ )     res1_->setTraining( is_training );
            if ( res2_ )     res2_->setTraining( is_training );
            if ( ffn_ )      ffn_->setTraining( is_training );

            forward_executed_ = false;
        }

    private:
        TransformerConfig config_;

        shape_t cached_input_shape_;

        bool forward_executed_{ false };

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        std::shared_ptr<AttentionType> attn_{ nullptr };
        std::shared_ptr<LayerNormType> ln1_{ nullptr };
        std::shared_ptr<LayerNormType> ln2_{ nullptr };
        std::shared_ptr<LinearType> qkv_proj_{ nullptr };
        std::shared_ptr<ResidualType> res1_{ nullptr };
        std::shared_ptr<ResidualType> res2_{ nullptr };
        std::shared_ptr<MLPType> ffn_{ nullptr };

        // Forward activation buffers
        std::shared_ptr<TensorType> ln1_output_{ nullptr };
        std::shared_ptr<TensorType> qkv_output_{ nullptr };
        std::shared_ptr<TensorType> attn_output_{ nullptr };
        std::shared_ptr<TensorType> res1_output_{ nullptr };
        std::shared_ptr<TensorType> ln2_output_{ nullptr };
        std::shared_ptr<TensorType> ffn_output_{ nullptr };
        std::shared_ptr<TensorType> res2_output_{ nullptr };

        // Backward gradient buffers (pre-allocated)
        std::shared_ptr<TensorType> d_act_1_{ nullptr };
        std::shared_ptr<TensorType> d_act_2_{ nullptr };
        std::shared_ptr<TensorType> d_act_3_{ nullptr };
        std::shared_ptr<TensorType> d_act_4_{ nullptr };
        std::shared_ptr<TensorType> d_qkv_{ nullptr };

        void createGraph()
        {
            auto attn_cfg = AttentionConfig( config_.getEmbeddingDim(), config_.getNumHeads() );

            auto attn_component = std::make_shared<AttentionType>( this->getName() + ".attn", attn_cfg, std::nullopt );
            this->addComponent( attn_component );

            auto ln1_cfg = LayerNormConfig().withNormalizedShape( shape_t{ static_cast<int64_t>(config_.getEmbeddingDim()) } );
            auto ln1_component = std::make_shared<LayerNormType>( this->getName() + ".lnorm_1", ln1_cfg, std::nullopt );
            this->addComponent( ln1_component );

            auto ln2_cfg = LayerNormConfig().withNormalizedShape( shape_t{ static_cast<int64_t>(config_.getEmbeddingDim()) } );
            auto ln2_component = std::make_shared<LayerNormType>( this->getName() + ".lnorm_2", ln2_cfg, std::nullopt );
            this->addComponent( ln2_component );

            auto qkv_cfg = LinearConfig( static_cast<dim_t>(config_.getEmbeddingDim()), static_cast<dim_t>(config_.getEmbeddingDim() * 3) );
            qkv_cfg.withBias( config_.useBias() );
            auto qkv_component = std::make_shared<LinearType>( this->getName() + ".fc_qkv_proj", qkv_cfg, std::nullopt );
            this->addComponent( qkv_component );

            ResidualConfig res_cfg1;
            res_cfg1.withScalingFactor( 1.0f );
            auto res1_component = std::make_shared<ResidualType>( this->getName() + ".res_1", res_cfg1, std::nullopt );
            this->addComponent( res1_component );

            ResidualConfig res_cfg2;
            res_cfg2.withScalingFactor( 1.0f );
            auto res2_component = std::make_shared<ResidualType>( this->getName() + ".res_2", res_cfg2, std::nullopt );
            this->addComponent( res2_component );

            dim_t hidden_dim = static_cast<dim_t>(config_.getHiddenDimension());
            if ( hidden_dim == 0 )
            {
                hidden_dim = static_cast<dim_t>(config_.getEmbeddingDim() * 4);
            }

            auto mlp_cfg = MLPConfig( static_cast<dim_t>(config_.getEmbeddingDim()), static_cast<dim_t>(hidden_dim) );
            mlp_cfg.withBias( config_.useBias() )
                   .withActivation( config_.getActivationType() );

            auto mlp_component = std::make_shared<MLPType>( this->getName() + ".mlp", mlp_cfg, std::nullopt );
            this->addComponent( mlp_component );
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 3 )
            {
                throw std::invalid_argument( "Transformer: input must have model-layout shape [B, T, embedding_dim]" );
            }

            int64_t trailing = input_shape.back();
            if ( trailing != static_cast<int64_t>(config_.getEmbeddingDim()) )
            {
                std::ostringstream oss;
                oss << "Transformer: embedding dimension mismatch. Config says "
                    << config_.getEmbeddingDim() << " got " << trailing;
                throw std::invalid_argument( oss.str() );
            }
        }
    };
}