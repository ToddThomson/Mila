/**
 * @file Transformer.ixx
 * @brief Transformer encoder block implementation.
 *
 * Provides a device-templated composite Transformer encoder block that
 * composes attention, layer-norm, residual connections and MLP sub-modules.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <type_traits>

export module Dnn.Blocks.Transformer;
export import :Config;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Module;
import Dnn.CompositeModule;
import Dnn.ActivationType;
import Compute.Precision;
import Compute.MemoryResource;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Dnn.Modules.LayerNorm;
import Dnn.Modules.Attention;
import Dnn.Modules.Residual;
import Dnn.Modules.Linear;
import Dnn.Blocks.MLP;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Transformer encoder block as a composite module.
     *
     * Device-templated composite module that composes:
     *   LayerNorm -> QKV projection -> MultiHeadSelfAttention -> Residual ->
     *   LayerNorm -> MLP -> Residual
     *
     * Two-phase initialization: createModules() constructs child modules,
     * build() finalizes shapes and allocates intermediate tensors.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Transformer : public CompositeModule<TDeviceType>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using CompositeModuleBase = CompositeModule<TDeviceType>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;
        using LayerNormType = LayerNorm<TDeviceType, TPrecision>;
        using AttentionType = Attention<TDeviceType, TPrecision>;
        using ResidualType = Residual<TDeviceType, TPrecision>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using MLPType = MLP<TDeviceType, TPrecision>;

        /**
         * @brief Construct with an execution context and config.
         *
         * Execution context must be non-null. Modules are created here;
         * concrete shape-dependent setup happens in build().
         */
        explicit Transformer( std::shared_ptr<ExecutionContextType> exec_context,
            const TransformerConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            createModules();
        }

        ~Transformer() override = default;

        // ====================================================================
        // Build lifecycle
        // ====================================================================

        bool isBuilt() const override
        {
            return built_ && attn_ && ln1_ && ln2_ && ffn_ && res1_ && res2_ && qkv_proj_;
        }

        void build( const shape_t& input_shape ) override
        {
            if (built_)
                return;

            // Validate runtime input shape and consistency with config
            validateInputShape( input_shape );

            cached_input_shape_ = input_shape;

            // Build LayerNorm for [B,T,embedding_dim]
            ln1_->build( input_shape );

            // QKV projection is a linear from embedding_dim -> 3 * embedding_dim
            qkv_proj_->build( input_shape );

            // Attention expects concatenated QKV: trailing dimension = 3 * embedding_dim
            shape_t qkv_shape = input_shape;
            qkv_shape.back() = static_cast<int64_t>( config_.getEmbeddingDim() * 3 );

            attn_->build( qkv_shape );

            // LayerNorm2 keeps same trailing dim as res1_output (embedding_dim)
            ln2_->build( input_shape );

            // Residual modules may allocate projection parameters if needed
            res1_->build( input_shape );
            res2_->build( input_shape );

            // MLP may change hidden dim internally; let MLP handle it
            ffn_->build( input_shape );

            auto device = exec_context_->getDevice();

            ln1_output_ = std::make_shared<TensorType>( device, input_shape );
            ln1_output_->setName( this->getName() + ".ln1_output" );

            // qkv_output has trailing dim 3 * embedding_dim
            qkv_output_ = std::make_shared<TensorType>( device, qkv_shape );
            qkv_output_->setName( this->getName() + ".qkv_output" );

            attn_output_ = std::make_shared<TensorType>( device, input_shape );
            attn_output_->setName( this->getName() + ".attn_output" );

            res1_output_ = std::make_shared<TensorType>( device, input_shape );
            res1_output_->setName( this->getName() + ".res1_output" );

            ln2_output_ = std::make_shared<TensorType>( device, input_shape );
            ln2_output_->setName( this->getName() + ".ln2_output" );

            ffn_output_ = std::make_shared<TensorType>( device, input_shape );
            ffn_output_->setName( this->getName() + ".ffn_output" );

            res2_output_ = std::make_shared<TensorType>( device, input_shape );
            res2_output_->setName( this->getName() + ".res2_output" );

            built_ = true;
        }

        // ====================================================================
        // Compute dispatch
        // ====================================================================

        void forward( const ITensor& input, ITensor& output )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Transformer must be built before forward()." );
            }

            // Work with concrete TensorType references for buffers
            TensorType const& in_t = static_cast<TensorType const&>(input);
            TensorType& out_t = static_cast<TensorType&>(output);

            // Pre-LN: normalize first
            ln1_->forward( in_t, *ln1_output_ );

            // Project normalized embeddings to concatenated QKV: [B,T,3*embedding_dim]
            qkv_proj_->forward( *ln1_output_, *qkv_output_ );

            // Attention consumes concatenated QKV and produces [B,T,embedding_dim]
            attn_->forward( *qkv_output_, *attn_output_ );

            // res1 = in + attn_output (use Residual module)
            res1_->forward( in_t, *attn_output_, *res1_output_ );

            ln2_->forward( *res1_output_, *ln2_output_ );
            ffn_->forward( *ln2_output_, *ffn_output_ );

            // out = res1 + ffn_output (use Residual module)
            res2_->forward( *res1_output_, *ffn_output_, out_t );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save( ModelArchive& archive ) const override
        {
            attn_->save( archive );
            ln1_->save( archive );
            ln2_->save( archive );
            qkv_proj_->save( archive );
            res1_->save( archive );
            res2_->save( archive );
            ffn_->save( archive );
        }

        void load( ModelArchive& archive ) override
        {
            attn_->load( archive );
            ln1_->load( archive );
            ln2_->load( archive );
            qkv_proj_->load( archive );
            res1_->load( archive );
            res2_->load( archive );
            ffn_->load( archive );
        }

        // ====================================================================
        // Module interface
        // ====================================================================

        std::string getName() const override
        {
            return config_.getName();
        }

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return exec_context_->getDevice();
        }

        void synchronize() override
        {
            if (exec_context_)
            {
                exec_context_->synchronize();
            }

            attn_->synchronize();
            ln1_->synchronize();
            ln2_->synchronize();
            qkv_proj_->synchronize();
            res1_->synchronize();
            res2_->synchronize();
            ffn_->synchronize();
        }

        void setTraining( bool is_training ) override
        {
            CompositeModuleBase::setTraining( is_training );

            attn_->setTraining( is_training );
            ln1_->setTraining( is_training );
            ln2_->setTraining( is_training );
            qkv_proj_->setTraining( is_training );
            res1_->setTraining( is_training );
            res2_->setTraining( is_training );
            ffn_->setTraining( is_training );
        }

        bool isTraining() const override
        {
            return CompositeModuleBase::isTraining();
        }

        size_t parameterCount() const override
        {
            size_t total = 0;

            total += attn_->parameterCount();
            total += ln1_->parameterCount();
            total += ln2_->parameterCount();
            total += qkv_proj_->parameterCount();
            total += res1_->parameterCount();
            total += res2_->parameterCount();
            total += ffn_->parameterCount();

            return total;
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;

            auto p = attn_->getParameters();
            params.insert( params.end(), p.begin(), p.end() );

            p = ln1_->getParameters();
            params.insert( params.end(), p.begin(), p.end() );

            p = ln2_->getParameters();
            params.insert( params.end(), p.begin(), p.end() );

            p = qkv_proj_->getParameters();
            params.insert( params.end(), p.begin(), p.end() );

            p = res1_->getParameters();
            params.insert( params.end(), p.begin(), p.end() );

            p = res2_->getParameters();
            params.insert( params.end(), p.begin(), p.end() );

            p = ffn_->getParameters();
            params.insert( params.end(), p.begin(), p.end() );

            return params;
        }

        std::vector<ITensor*> getParameterGradients() const override
        {
            if (!isTraining())
            {
                throw std::runtime_error( "Cannot get parameter gradients when not in training mode" );
            }

            std::vector<ITensor*> grads;

            auto g = attn_->getParameterGradients();
            grads.insert( grads.end(), g.begin(), g.end() );

            g = ln1_->getParameterGradients();
            grads.insert( grads.end(), g.begin(), g.end() );

            g = ln2_->getParameterGradients();
            grads.insert( grads.end(), g.begin(), g.end() );

            g = qkv_proj_->getParameterGradients();
            grads.insert( grads.end(), g.begin(), g.end() );

            g = res1_->getParameterGradients();
            grads.insert( grads.end(), g.begin(), g.end() );

            g = res2_->getParameterGradients();
            grads.insert( grads.end(), g.begin(), g.end() );

            g = ffn_->getParameterGradients();
            grads.insert( grads.end(), g.begin(), g.end() );

            return grads;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "====================" << std::endl;
            oss << "Transformer: " << getName() << std::endl;

            if (!cached_input_shape_.empty())
            {
                oss << "Input shape: (";
                for (size_t i = 0; i < cached_input_shape_.size(); ++i)
                {
                    oss << cached_input_shape_[i];
                    if (i != cached_input_shape_.size() - 1) oss << ", ";
                }
                oss << ")" << std::endl;
            }

            oss << "Number of heads: " << config_.getNumHeads() << std::endl;
            oss << "MLP hidden dimension: " << config_.getHiddenDimension() << std::endl;
            oss << "Architecture: Pre-LN" << std::endl;

            if (exec_context_ && exec_context_->getDevice())
            {
                oss << "Device: " << deviceTypeToString( exec_context_->getDevice()->getDeviceType() ) << std::endl;
            }

            oss << "Parameter count: " << parameterCount() << std::endl;

            // blank line before return per style
            return oss.str();
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        const TransformerConfig& getConfig() const noexcept
        {
            return config_;
        }

        std::shared_ptr<AttentionType> getAttention() const noexcept
        {
            return attn_;
        }
        std::shared_ptr<LayerNormType> getLn1() const noexcept
        {
            return ln1_;
        }
        std::shared_ptr<LayerNormType> getLn2() const noexcept
        {
            return ln2_;
        }
        std::shared_ptr<ResidualType> getRes1() const noexcept
        {
            return res1_;
        }
        std::shared_ptr<ResidualType> getRes2() const noexcept
        {
            return res2_;
        }
        std::shared_ptr<MLPType> getFFN() const noexcept
        {
            return ffn_;
        }

    protected:
        void buildImpl( const shape_t& ) override
        {
            // build() handles shape-dependent initialization for this composite.
        }

    private:
        TransformerConfig config_;
        bool built_{ false };
        std::shared_ptr<ExecutionContextType> exec_context_;

        shape_t cached_input_shape_;

        std::shared_ptr<AttentionType> attn_{ nullptr };
        std::shared_ptr<LayerNormType> ln1_{ nullptr };
        std::shared_ptr<LayerNormType> ln2_{ nullptr };
        std::shared_ptr<LinearType> qkv_proj_{ nullptr };
        std::shared_ptr<ResidualType> res1_{ nullptr };
        std::shared_ptr<ResidualType> res2_{ nullptr };
        std::shared_ptr<MLPType> ffn_{ nullptr };

        std::shared_ptr<TensorType> ln1_output_{ nullptr };
        std::shared_ptr<TensorType> qkv_output_{ nullptr };
        std::shared_ptr<TensorType> attn_output_{ nullptr };
        std::shared_ptr<TensorType> res1_output_{ nullptr };
        std::shared_ptr<TensorType> ln2_output_{ nullptr };
        std::shared_ptr<TensorType> ffn_output_{ nullptr };
        std::shared_ptr<TensorType> res2_output_{ nullptr };

        void createModules()
        {
            dim_t embedding_dim = config_.getEmbeddingDim();
            dim_t num_heads = config_.getNumHeads();
            dim_t hidden_dim = static_cast<dim_t>(config_.getHiddenDimension());

            // If hidden dim not specified, follow default: 4x embedding dim
            if (hidden_dim == 0)
            {
                hidden_dim = embedding_dim * 4;
            }

            // Attention module is configured with base embedding_dim and num_heads.
            auto attn_cfg = AttentionConfig( embedding_dim, num_heads );
            attn_cfg.withName( config_.getName() + ".attn" );

            attn_ = std::make_shared<AttentionType>( exec_context_, attn_cfg );

            // LayerNorms configured to normalize over the trailing embedding_dim
            auto ln1_cfg = LayerNormConfig();
            ln1_cfg.withNormalizedShape( shape_t{ static_cast<int64_t>( embedding_dim ) } )
                .withName( config_.getName() + ".ln1" );

            ln1_ = std::make_shared<LayerNormType>( exec_context_, ln1_cfg );

            auto ln2_cfg = LayerNormConfig();
            ln2_cfg.withNormalizedShape( shape_t{ static_cast<int64_t>( embedding_dim ) } )
                .withName( config_.getName() + ".ln2" );

            ln2_ = std::make_shared<LayerNormType>( exec_context_, ln2_cfg );

            // QKV projection: Linear(in=embedding_dim, out=3*embedding_dim)
            auto qkv_cfg = LinearConfig( static_cast<dim_t>( embedding_dim ), static_cast<dim_t>( embedding_dim * 3 ) );
            qkv_cfg.withName( config_.getName() + ".qkv_proj" )
                .withBias( config_.useBias() );

            qkv_proj_ = std::make_shared<LinearType>( exec_context_, qkv_cfg );

            // Residual configurations (simple addition, default scaling)
            ResidualConfig res_cfg1;
            res_cfg1.withName( config_.getName() + ".res1" )
                .withScalingFactor( 1.0f );

            ResidualConfig res_cfg2;
            res_cfg2.withName( config_.getName() + ".res2" )
                .withScalingFactor( 1.0f );

            res1_ = std::make_shared<ResidualType>( exec_context_, res_cfg1 );
            res2_ = std::make_shared<ResidualType>( exec_context_, res_cfg2 );

            auto mlp_cfg = MLPConfig( static_cast<dim_t>(embedding_dim), static_cast<dim_t>(hidden_dim) );
            mlp_cfg.withName( config_.getName() + ".mlp" )
                .withBias( config_.useBias() )
                .withActivation( config_.getActivationType() );

            ffn_ = std::make_shared<MLPType>( exec_context_, mlp_cfg );
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            // Expect model layout [B, T, embedding_dim]
            if (input_shape.size() != 3)
            {
                throw std::invalid_argument( "Transformer: input must have model-layout shape [B, T, embedding_dim]" );
            }

            int64_t trailing = input_shape.back();
            if (trailing != static_cast<int64_t>( config_.getEmbeddingDim() ))
            {
                std::ostringstream oss;
                oss << "Transformer: embedding dimension mismatch. Config says "
                    << config_.getEmbeddingDim() << " got " << trailing;
                throw std::invalid_argument( oss.str() );
            }
        }
    };
}