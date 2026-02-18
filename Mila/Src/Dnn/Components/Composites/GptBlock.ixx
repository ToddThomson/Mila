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
#include <iomanip>
#include <format>
#include <utility>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <optional>
#include <algorithm>

export module Dnn.Components.GptBlock;
export import :Config;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Dnn.TensorInitializers;
import Dnn.Component;
import Dnn.ComponentType;
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
import Dnn.Components.MLP;
import Serialization.ModelArchive;
import Serialization.Mode;

// DEBUG:
import Dnn.TensorHelpers;
import Utils.Logger;

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
    class GptBlock : public CompositeComponent<TDeviceType, TPrecision>
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
        explicit GptBlock( const std::string& name, const GptBlockConfig& config, std::optional<DeviceId> device_id = std::nullopt )
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

        ~GptBlock() override = default;

        // ====================================================================
        // Forward and backward dispatch (hot path)
        // ====================================================================

        /**
         * @brief Forward pass.
         *
         * Produces the block output into an owned buffer and returns a non-owning
         * pointer to that buffer. The returned pointer is mutable and non-owning.
         * It remains valid until the block is destroyed or the block mutates the
         * buffer on a subsequent call.
         *
         * Preconditions:
         *  - Component must be built prior to calling forward().
         *
         * Postconditions:
         *  - forward_executed_ is set to the current training mode state.
         *
         * @param input Forward input tensor
         * @return Non-owning pointer to ITensor containing the block output
         */
        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "Transformer must be built before forward()." );
            }

            // DEBUG: Check input range
            auto host_input = toHost<TensorDataType::FP32>( input );
            auto host_input_ptr = host_input.data();
            const size_t n = host_input.size();
            auto [min_in, max_in] = std::minmax_element( host_input_ptr, host_input_ptr + n );
            Utils::Logger::debug( std::format( "GptBlock {} input in:[{:.3f}, {:.3f}] with shape:{}]",
                this->getName(), *min_in, *max_in, shapeToString( input.shape() ) ) );
            // END DEBUG:

            auto& ln1_out = ln1_->forward( input );
            this->getExecutionContext()->synchronize();

            // DEBUG: Check output range
            auto host_output = toHost<TensorDataType::FP32>( ln1_out );
            auto host_output_ptr = host_output.data();
            size_t output_n = host_output.size();
            auto [min_out, max_out] = std::minmax_element( host_output_ptr, host_output_ptr + output_n );

            Utils::Logger::debug( std::format( "GptBlock {} ln_1 in:[{:.3f}, {:.3f}] out:[{:.3f}, {:.3f}]",
                this->getName(), *min_in, *max_in, *min_out, *max_out ) );
            // DEBUG END

            auto& qkv_out = qkv_proj_->forward( ln1_out );
            this->getExecutionContext()->synchronize();

            // DEBUG: Check output range
            host_output = toHost<TensorDataType::FP32>( qkv_out );
            host_output_ptr = host_output.data();
            output_n = host_output.size();
            auto [min_qkv_out, max_qkv_out] = std::minmax_element( host_output_ptr, host_output_ptr + output_n );

            Utils::Logger::debug( std::format( "GptBlock {} qkv_proj out:[{:.3f}, {:.3f}]",
                this->getName(), *min_qkv_out, *max_qkv_out ) );
            // DEBUG END

            auto& attn_out = attn_->forward( qkv_out );
            this->getExecutionContext()->synchronize();

            // DEBUG: Check output range
            host_output = toHost<TensorDataType::FP32>( attn_out );
            host_output_ptr = host_output.data();
            output_n = host_output.size();
            auto [min_attn_out, max_attn_out] = std::minmax_element( host_output_ptr, host_output_ptr + output_n );

            Utils::Logger::debug( std::format( "GptBlock {} attn out:[{:.3f}, {:.3f}]",
                this->getName(), *min_attn_out, *max_attn_out ) );
            // DEBUG END

            // BUG: FIXED: The Gpt2 based transformer was missing the output projection fc layer here!!!
            auto& out_proj = out_proj_->forward( attn_out );
            this->getExecutionContext()->synchronize();

            // DEBUG: Check output range
            host_output = toHost<TensorDataType::FP32>( out_proj );
            host_output_ptr = host_output.data();
            output_n = host_output.size();
            auto [min_out_proj, max_out_proj] = std::minmax_element( host_output_ptr, host_output_ptr + output_n );

            Utils::Logger::debug( std::format( "GptBlock {} out_proj out:[{:.3f}, {:.3f}]",
                this->getName(), *min_out_proj, *max_out_proj ) );
            // DEBUG END

            auto& res1_out = res1_->forward( input, out_proj );
            this->getExecutionContext()->synchronize();

            // DEBUG: Check output range
            host_output = toHost<TensorDataType::FP32>( res1_out );
            host_output_ptr = host_output.data();
            output_n = host_output.size();
            auto [min_res1_out, max_res1_out] = std::minmax_element( host_output_ptr, host_output_ptr + output_n );

            Utils::Logger::debug( std::format( "GptBlock {} res_1_out out:[{:.3f}, {:.3f}]",
                this->getName(), *min_res1_out, *max_res1_out ) );
            // DEBUG END

            auto& ln2_out = ln2_->forward( res1_out );
            this->getExecutionContext()->synchronize();

            // DEBUG: Check output range
            host_output = toHost<TensorDataType::FP32>( ln2_out );
            host_output_ptr = host_output.data();
            output_n = host_output.size();
            auto [min_ln2_out, max_ln2_out] = std::minmax_element( host_output_ptr, host_output_ptr + output_n );

            Utils::Logger::debug( std::format( "GptBlock {} ln_2_out out:[{:.3f}, {:.3f}]",
                this->getName(), *min_ln2_out, *max_ln2_out ) );
            // DEBUG END

            auto& ffn_out = ffn_->forward( ln2_out );
            this->getExecutionContext()->synchronize();

            // DEBUG: Check output range
            host_output = toHost<TensorDataType::FP32>( ffn_out );
            host_output_ptr = host_output.data();
            output_n = host_output.size();
            auto [min_ffn_out, max_ffn_out] = std::minmax_element( host_output_ptr, host_output_ptr + output_n );

            Utils::Logger::debug( std::format( "GptBlock {} ffn_out out:[{:.3f}, {:.3f}]",
                this->getName(), *min_ffn_out, *max_ffn_out ) );
            // DEBUG END

            auto& res2_out = res2_->forward( res1_out, ffn_out );
            this->getExecutionContext()->synchronize();

            // DEBUG: Check output range
            host_output = toHost<TensorDataType::FP32>( res2_out );
            host_output_ptr = host_output.data();
            output_n = host_output.size();
            auto [min_res2_out, max_res2_out] = std::minmax_element( host_output_ptr, host_output_ptr + output_n );

            Utils::Logger::debug( std::format( "GptBlock {} res2_out out:[{:.3f}, {:.3f}]",
                this->getName(), *min_res2_out, *max_res2_out ) );
            // DEBUG END

            // Cache non-owning pointers produced by the last forward for use by backward().
            // REVIEW: If you need these for backward, just call the components' accessors:
            last_ln1_out_  = &ln1_out;
            last_qkv_out_  = &qkv_out;
            last_attn_out_ = &attn_out;
            last_out_proj_out_ = &out_proj;
            last_res1_out_ = &res1_out;
            last_ln2_out_  = &ln2_out;
            last_ffn_out_  = &ffn_out;
            last_res2_out_ = &res2_out;

            // Copy final child output into the transformer's encapsulated output buffer
            // to preserve the component contract (forward returns pointer to transformer's buffer).
            //copy( *res2_out, *output_buffer_ );
            this->getExecutionContext()->synchronize();

            forward_executed_ = this->isTraining();

            return res2_out;
        }

        /**
         * @brief Backward pass that returns a non-owning pointer to the input-gradient tensor.
         *
         * The transformer block owns a preallocated input-gradient buffer allocated
         * during `onBuilding()`. This method computes gradients w.r.t. the block's
         * input and stores them in that owned buffer. A non-owning pointer to the
         * buffer is returned for caller chaining. The returned pointer remains valid
         * until the block is destroyed or the block mutates the buffer on a subsequent call.
         *
         * Preconditions:
         *  - The component must be built.
         *  - Training mode must be active.
         *  - A training-mode forward() must have been executed before calling backward().
         *
         * @param input Forward input previously passed to `forward()`
         * @param output_grad Gradient w.r.t. this block's output
         * @return Non-owning pointer to ITensor containing dLoss/d(input)
         */
        TensorType& backward( const TensorType& input, const TensorType& output_grad )
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

            // Backward through res2
            auto [d_res1_from_res2, d_ffn_from_res2] =
                res2_->backward( *last_res1_out_, *last_ffn_out_, output_grad );

            // FFN branch
            auto& d_ln2_from_ffn = ffn_->backward( *last_ln2_out_, d_ffn_from_res2 );
            auto& d_res1_from_ln2 = ln2_->backward( *last_res1_out_, d_ln2_from_ffn );

            // ACCUMULATE at res1_out junction using d_res1_accum_
            zero( *d_res1_accum_ );
            add( d_res1_from_res2, d_res1_from_ln2, *d_res1_accum_ );

            // Backward through res1
            auto [d_input_from_res1, d_out_proj_from_res1 /* was d_attn_from_res1 */] =
                res1_->backward( input, *last_out_proj_out_, /* *last_attn_out_, */ *d_res1_accum_ );

            // BUG FIX: Added backward through out_proj layer
            auto& d_attn_from_out_proj = out_proj_->backward( *last_attn_out_, d_out_proj_from_res1 );
            this->getExecutionContext()->synchronize();

            // Attention branch
            auto& d_qkv = attn_->backward( *last_qkv_out_, d_attn_from_out_proj /* was d_attn_from_res1 */);
            auto& d_ln1 = qkv_proj_->backward( *last_ln1_out_, d_qkv );
            auto& d_input_from_ln1 = ln1_->backward( input, d_ln1 );

            // ACCUMULATE at input junction using d_input_
            zero( *d_input_ );
            add( d_input_from_res1, d_input_from_ln1, *d_input_ );

            forward_executed_ = false;
            return *d_input_;
        }

        void initializeKVCache( int64_t max_seq_len )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptBlock must be built before initializeKVCache()." );
            }

            attn_->initializeKVCache( max_seq_len );
        }

        void resetKVCache()
        {
            attn_->resetKVCache();
        }

        TensorType& forwardPrefill( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptBlock must be built before forwardPrefill()." );
            }

            auto& ln1_out = ln1_->forward( input );
            this->getExecutionContext()->synchronize();

            auto& qkv_out = qkv_proj_->forward( ln1_out );
            this->getExecutionContext()->synchronize();

            auto& attn_out = attn_->forwardPrefill( qkv_out );
            this->getExecutionContext()->synchronize();

            auto& out_proj = out_proj_->forward( attn_out );
            this->getExecutionContext()->synchronize();

            auto& res1_out = res1_->forward( input, out_proj );
            this->getExecutionContext()->synchronize();

            auto& ln2_out = ln2_->forward( res1_out );
            this->getExecutionContext()->synchronize();

            auto& ffn_out = ffn_->forward( ln2_out );
            this->getExecutionContext()->synchronize();

            auto& res2_out = res2_->forward( res1_out, ffn_out );
            this->getExecutionContext()->synchronize();

            return res2_out;
        }

        TensorType& forwardDecode( const TensorType& input, int position )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptBlock must be built before forwardDecode()." );
            }

            auto& ln1_out = ln1_->forward( input );
            this->getExecutionContext()->synchronize();

            auto& qkv_out = qkv_proj_->forward( ln1_out );
            this->getExecutionContext()->synchronize();

            auto& attn_out = attn_->forwardDecode( qkv_out, position );
            this->getExecutionContext()->synchronize();

            auto& out_proj = out_proj_->forward( attn_out );
            this->getExecutionContext()->synchronize();

            auto& res1_out = res1_->forward( input, out_proj );
            this->getExecutionContext()->synchronize();

            auto& ln2_out = ln2_->forward( res1_out );
            this->getExecutionContext()->synchronize();

            auto& ffn_out = ffn_->forward( ln2_out );
            this->getExecutionContext()->synchronize();

            auto& res2_out = res2_->forward( res1_out, ffn_out );
            this->getExecutionContext()->synchronize();

            return res2_out;
        }

        void zeroGradients() override
        {
            if ( d_res1_accum_ )
                zero( *d_res1_accum_ );

            if ( d_input_ )
                zero( *d_input_ );

            attn_->zeroGradients();
            qkv_proj_->zeroGradients();
            out_proj_->zeroGradients();
            ln1_->zeroGradients();
            ln2_->zeroGradients();
            res1_->zeroGradients();
            res2_->zeroGradients();
            ffn_->zeroGradients();
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            attn_->save_( archive, mode );
            ln1_->save_( archive, mode );
            ln2_->save_( archive, mode );
            qkv_proj_->save_( archive, mode );
            out_proj_->save_( archive, mode );
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
            out_proj_->load_( archive, mode );
            res1_->load_( archive, mode );
            res2_->load_( archive, mode );
            ffn_->load_( archive, mode );
        }

        // ====================================================================
        // Identification and Description
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::Transformer;
        }

        // ====================================================================
        // Diagnostics
        // ====================================================================

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "====================" << std::endl;
            oss << "GptBlock: " << this->getName() << std::endl;

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
            oss << "MLP hidden dimension: " << config_.getHiddenSize() << std::endl;
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

            ln1_ = this->template getComponentAs<LayerNormType>( this->getName() + ".ln_1" );
            ln1_->build( input_shape );

            qkv_proj_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_qkv_proj" );
            qkv_proj_->build( input_shape );

            out_proj_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_out_proj" );
            out_proj_->build( input_shape );

            shape_t qkv_shape = input_shape;
            qkv_shape.back() = static_cast<int64_t>(config_.getModelDim() * 3);

            attn_ = this->template getComponentAs<AttentionType>( this->getName() + ".attn" );
            attn_->build( qkv_shape );

            ln2_ = this->template getComponentAs<LayerNormType>( this->getName() + ".ln_2" );
            ln2_->build( input_shape );

            res1_ = this->template getComponentAs<ResidualType>( this->getName() + ".res_1" );
            res1_->build( input_shape );

            res2_ = this->template getComponentAs<ResidualType>( this->getName() + ".res_2" );
            res2_->build( input_shape );

            ffn_ = this->template getComponentAs<MLPType>( this->getName() + ".mlp" );
            ffn_->build( input_shape );

            auto device = this->getDeviceId();

            //// Forward activation buffers (kept for compatibility / optional use)
            //ln1_output_ = std::make_shared<TensorType>( device, input_shape );
            //ln1_output_->setName( this->getName() + ".lnorm_1_output" );

            //qkv_output_ = std::make_shared<TensorType>( device, qkv_shape );
            //qkv_output_->setName( this->getName() + ".qkv_output" );

            //attn_output_ = std::make_shared<TensorType>( device, input_shape );
            //attn_output_->setName( this->getName() + ".attn_output" );

            //res1_output_ = std::make_shared<TensorType>( device, input_shape );
            //res1_output_->setName( this->getName() + ".res_1_output" );

            //ln2_output_ = std::make_shared<TensorType>( device, input_shape );
            //ln2_output_->setName( this->getName() + ".lnorm_2_output" );

            //ffn_output_ = std::make_shared<TensorType>( device, input_shape );
            //ffn_output_->setName( this->getName() + ".ffn_output" );

            //res2_output_ = std::make_shared<TensorType>( device, input_shape );
            //res2_output_->setName( this->getName() + ".res_2_output" );
            
            // Backward gradient buffers for residual paths
            d_res1_accum_ = std::make_shared<TensorType>( device, input_shape );
            d_res1_accum_->setName( this->getName() + ".d_res1_accum" );
            zero( *d_res1_accum_ );

            // Owned input-gradient buffer (strict encapsulation)
            d_input_ = std::make_shared<TensorType>( device, input_shape );
            d_input_->setName( this->getName() + ".d_input" );
            zero( *d_input_ );
        }

        void onTrainingChanging( bool is_training ) override
        {
            if ( attn_ )     attn_->setTraining( is_training );
            if ( ln1_ )      ln1_->setTraining( is_training );
            if ( ln2_ )      ln2_->setTraining( is_training );
            if ( qkv_proj_ ) qkv_proj_->setTraining( is_training );
            if ( out_proj_ ) out_proj_->setTraining( is_training );
            if ( res1_ )     res1_->setTraining( is_training );
            if ( res2_ )     res2_->setTraining( is_training );
            if ( ffn_ )      ffn_->setTraining( is_training );

            forward_executed_ = false;
        }

    private:
        GptBlockConfig config_;

        shape_t cached_input_shape_;

        bool forward_executed_{ false };

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        std::shared_ptr<AttentionType> attn_{ nullptr };
        std::shared_ptr<LayerNormType> ln1_{ nullptr };
        std::shared_ptr<LayerNormType> ln2_{ nullptr };
        std::shared_ptr<LinearType> qkv_proj_{ nullptr };
        std::shared_ptr<LinearType> out_proj_{ nullptr };
        std::shared_ptr<ResidualType> res1_{ nullptr };
        std::shared_ptr<ResidualType> res2_{ nullptr };
        std::shared_ptr<MLPType> ffn_{ nullptr };

        // Backward gradient buffers for residual paths
        std::shared_ptr<TensorType> d_res1_accum_{ nullptr };

        // Owned input-gradient buffer (encapsulated)
        std::shared_ptr<TensorType> d_input_{ nullptr };

        // Non-owning pointers to most-recent forward outputs (used by backward)
        TensorType* last_ln1_out_{ nullptr };
        TensorType* last_qkv_out_{ nullptr };
        TensorType* last_attn_out_{ nullptr };
        TensorType* last_out_proj_out_{ nullptr };
        TensorType* last_res1_out_{ nullptr };
        TensorType* last_ln2_out_{ nullptr };
        TensorType* last_ffn_out_{ nullptr };
        TensorType* last_res2_out_{ nullptr };

        void createGraph()
        {
            auto attn_cfg = AttentionConfig( config_.getModelDim(), config_.getNumHeads() );

            auto attn_component = std::make_shared<AttentionType>( this->getName() + ".attn", attn_cfg, std::nullopt );
            this->addComponent( attn_component );

            auto ln1_cfg = LayerNormConfig().withNormalizedShape( shape_t{ static_cast<int64_t>(config_.getModelDim()) } );
            auto ln1_component = std::make_shared<LayerNormType>( this->getName() + ".ln_1", ln1_cfg, std::nullopt );
            this->addComponent( ln1_component );

            auto ln2_cfg = LayerNormConfig().withNormalizedShape( shape_t{ static_cast<int64_t>(config_.getModelDim()) } );
            auto ln2_component = std::make_shared<LayerNormType>( this->getName() + ".ln_2", ln2_cfg, std::nullopt );
            this->addComponent( ln2_component );

            auto qkv_cfg = LinearConfig( static_cast<dim_t>(config_.getModelDim()), static_cast<dim_t>(config_.getModelDim() * 3) );
            qkv_cfg.withBias( config_.useBias() );
            auto qkv_component = std::make_shared<LinearType>( this->getName() + ".fc_qkv_proj", qkv_cfg, std::nullopt );
            this->addComponent( qkv_component );

            auto out_proj_cfg = LinearConfig( static_cast<dim_t>(config_.getModelDim()), static_cast<dim_t>(config_.getModelDim()) );
            out_proj_cfg.withBias( config_.useBias() );
            auto out_proj_component = std::make_shared<LinearType>( this->getName() + ".fc_out_proj", out_proj_cfg, std::nullopt );
            this->addComponent( out_proj_component );

            ResidualConfig res_cfg1;
            res_cfg1.withScalingFactor( config_.getResidualScale() );
            auto res1_component = std::make_shared<ResidualType>( this->getName() + ".res_1", res_cfg1, std::nullopt );
            this->addComponent( res1_component );

            ResidualConfig res_cfg2;
            res_cfg2.withScalingFactor( config_.getResidualScale() );
            auto res2_component = std::make_shared<ResidualType>( this->getName() + ".res_2", res_cfg2, std::nullopt );
            this->addComponent( res2_component );

            dim_t hidden_dim = static_cast<dim_t>(config_.getHiddenSize());
            if ( hidden_dim == 0 )
            {
                hidden_dim = static_cast<dim_t>(config_.getModelDim() * 4);
            }

            auto mlp_cfg = MLPConfig( static_cast<dim_t>(config_.getModelDim()), static_cast<dim_t>(hidden_dim) );
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
            if ( trailing != static_cast<int64_t>(config_.getModelDim()) )
            {
                std::ostringstream oss;
                oss << "GptBlock: model dimension mismatch. Config says "
                    << config_.getModelDim() << " got " << trailing;
                throw std::invalid_argument( oss.str() );
            }
        }
    };
}