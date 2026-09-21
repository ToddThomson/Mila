/**
 * @file GemmaModel.MixtureOfExperts.Parity.Cuda.cpp
 * @brief Layer-streamed HuggingFace parity for the Gemma 4 26B-A4B stack (Specifications/Gemma4MoE.md Phase 8).
 *
 * The BF16 weights are 47 GiB against a 16 GiB card, so one decoder block is resident at a time, against the
 * references `Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_layer_stream.py` writes. Requires the weights and
 * both references and skips without them, so it never runs in CI.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    namespace fs = std::filesystem;

    namespace
    {
        constexpr auto kPrecision = TensorDataType::BF16;

        using NoKvCompression = Mila::Dnn::Quant::KvCache::NoKvCompression;
        using NoWeightQuant = Mila::Dnn::Quant::Weight::NoWeightQuant;
        using MR = typename DeviceTypeTraits<DeviceType::Cuda>::memory_resource;

        using GemmaBf16 = GemmaModel<DeviceType::Cuda, kPrecision>;
        // The routed chassis a load of these weights builds: the dense branch delegated to `mlp`.
        using LocalBlock = GemmaBlock<DeviceType::Cuda, kPrecision, false, NoWeightQuant, NoKvCompression, true, true>;
        using GlobalBlock = GemmaBlock<DeviceType::Cuda, kPrecision, true, NoWeightQuant, NoKvCompression, true, true>;
        using EmbeddingType = TokenEmbedding<DeviceType::Cuda, TensorDataType::INT32, kPrecision, NoWeightQuant>;
        using RmsNormType = RmsNorm<DeviceType::Cuda, kPrecision>;
        using HeadType = Linear<DeviceType::Cuda, kPrecision>;
        using DecoderLayer = ITransformerBlock<DeviceType::Cuda, kPrecision>;
        using LoadableComponent = Component<DeviceType::Cuda, kPrecision>;
        using CompositeType = CompositeComponent<DeviceType::Cuda, kPrecision>;

        using DeviceTensor = Tensor<kPrecision, MR>;
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using TokenTensor = Tensor<TensorDataType::INT32, MR>;
        using HostTokenTensor = Tensor<TensorDataType::INT32, CpuMemoryResource>;

        // Against HuggingFace's FP32 run, the reference the BF16 wiring gate was decided against (Gemma4MoE.md
        // Phase 8, "Decided 2026-09-12"): HuggingFace's own BF16 run sits 1.4e-2 to 4.1e-2 from its FP32 run on
        // this chassis, so a tighter bound would fail a correct BF16 implementation.
        constexpr double kRelativeTolerance = 1.0e-1;

        fs::path modelDirectory()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "gemma";
        }

        void synchronizeDevice()
        {
            ASSERT_EQ( cudaDeviceSynchronize(), cudaSuccess );
        }

        std::vector<float> readReferenceVector( WeightsReader& reader, const std::string& name )
        {
            auto blob = reader.readTensorBlob<CpuMemoryResource>( name );

            std::vector<float> values( blob.sizeBytes() / sizeof( float ) );
            std::memcpy( values.data(), blob.data(), blob.sizeBytes() );

            return values;
        }

        std::vector<int32_t> readPromptIds( WeightsReader& reader )
        {
            auto blob = reader.readTensorBlob<CpuMemoryResource>( "prompt_ids" );

            std::vector<int32_t> ids( blob.sizeBytes() / sizeof( int32_t ) );
            std::memcpy( ids.data(), blob.data(), blob.sizeBytes() );

            return ids;
        }

        double relativeError( const std::vector<float>& got, const std::vector<float>& want )
        {
            double diff = 0.0;
            double norm = 0.0;

            for ( size_t i = 0; i < want.size(); ++i )
            {
                const double d = static_cast<double>( got[ i ] ) - static_cast<double>( want[ i ] );
                diff += d * d;
                norm += static_cast<double>( want[ i ] ) * static_cast<double>( want[ i ] );
            }

            return norm > 0.0 ? std::sqrt( diff ) / std::sqrt( norm ) : std::sqrt( diff );
        }

        std::vector<float> toHost( const DeviceTensor& tensor )
        {
            HostTensor host( Device::Cpu(), tensor.shape() );
            copy( tensor, host );

            const auto* data = static_cast<const float*>( host.data() );

            return std::vector<float>( data, data + static_cast<size_t>( host.size() ) );
        }

        void fromHost( const std::vector<float>& values, size_t offset, DeviceTensor& tensor )
        {
            HostTensor host( Device::Cpu(), tensor.shape() );
            std::memcpy( host.data(), values.data() + offset, static_cast<size_t>( host.size() ) * sizeof( float ) );

            copy( host, tensor );
        }

        std::vector<float> tail( const std::vector<float>& values, dim_t width )
        {
            return std::vector<float>( values.end() - width, values.end() );
        }

        int64_t argmax( const std::vector<float>& values )
        {
            return std::distance( values.begin(), std::ranges::max_element( values ) );
        }

        /// Load every weight under `prefix` into the component named for it.
        void loadComponentParameters(
            WeightsReader& reader, LoadableComponent& component, const std::string& prefix )
        {
            size_t loaded = 0;

            for ( const auto& name : reader.getTensorNames() )
            {
                if ( !name.starts_with( prefix ) )
                    continue;

                // A leaf's tensor has no path after the prefix ("temb.wte" -> "wte"); a composite's does.
                const std::string relative = name.substr( prefix.size() );
                const auto last_dot = relative.rfind( '.' );

                const std::string component_path =
                    last_dot == std::string::npos ? std::string{} : relative.substr( 0, last_dot );
                const std::string parameter_name =
                    last_dot == std::string::npos ? relative : relative.substr( last_dot + 1 );

                auto blob = reader.readTensorBlob<CpuMemoryResource>( name );

                if ( component_path.empty() )
                {
                    component.loadParameter( parameter_name, blob );
                }
                else
                {
                    auto* composite = dynamic_cast<CompositeType*>( &component );
                    ASSERT_NE( composite, nullptr ) << "Nested path on a leaf component: " << name;

                    composite->findComponent( component_path )->loadParameter( parameter_name, blob );
                }

                ++loaded;
            }

            // A prefix that matched nothing leaves uninitialized memory and a plausible hidden state.
            ASSERT_GT( loaded, 0u ) << "No weights matched prefix '" << prefix << "'";
        }
    }

    class GemmaMixtureOfExpertsParityCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
                GTEST_SKIP() << "No CUDA device available";

            weights_ = modelDirectory() / "gemma4_26b_a4b_it_bf16.bin";
            reference_ = modelDirectory() / "gemma4_26b_a4b_ref.bin";
            truth_ = modelDirectory() / "gemma4_26b_a4b_ref_fp32.bin";

            for ( const auto& path : { weights_, reference_, truth_ } )
            {
                if ( !fs::exists( path ) )
                    GTEST_SKIP() << "Not present: " << path.string()
                                 << " -- see Tools/Converters/Gemma/gemma_4_26b_moe/hf_gemma_layer_stream.py";
            }
        }

        fs::path weights_;
        fs::path reference_;
        fs::path truth_;
    };

    // Each block is constructed, loaded, run and destroyed in turn, which leaves three differences from a real
    // load, all recorded against the Qwen harness in Qwen3.8.md: every component owns its own stream, so stages
    // are separated by a device-wide synchronize; the hidden state crosses the host between blocks, since a
    // block's output dies with it; and prefill is one chunk with flash off, so the GQA score width is the
    // prompt. The workspaces come from the factories GemmaTransformer itself calls.
    TEST_F( GemmaMixtureOfExpertsParityCudaTests, LayerStream_MatchesHuggingFaceReference )
    {
        WeightsReader weights( weights_ );
        WeightsReader reference( reference_ );
        WeightsReader truth( truth_ );

        const GemmaConfig config = GemmaBf16::configFromMetadata( weights.getWeightsMetadata() );
        ASSERT_TRUE( config.hasMixtureOfExperts() ) << "these weights are not the routed chassis";
        ASSERT_TRUE( config.getTieWordEmbeddings() );

        const std::vector<int32_t> prompt_ids = readPromptIds( reference );
        ASSERT_FALSE( prompt_ids.empty() );
        ASSERT_EQ( prompt_ids, readPromptIds( truth ) ) << "the two references ran different prompts";

        const dim_t B = 1;
        const dim_t T = static_cast<dim_t>( prompt_ids.size() );
        const dim_t model_dim = config.getModelDim();
        const DeviceId device{ DeviceType::Cuda, 0 };

        const BuildContext block_context =
            BuildContext( shape_t{ B, T, model_dim }, RuntimeMode::Inference, false ).withPrefillSize( T );

        auto block_workspace = makeGemmaBlockWorkspace<DeviceType::Cuda, kPrecision>(
            config, device, B, T, "parity.block_ws." );

        // score_width = T because flash prefill is off below; the two must agree.
        auto gqa_workspace = makeGqaWorkspace<DeviceType::Cuda, kPrecision>(
            device, B, config.getNumHeads(), std::max( config.getHeadDim(), config.getGlobalHeadDim() ),
            T, T, T, "parity.gqa_ws." );

        std::vector<float> hidden;

        // ---- Embedding ------------------------------------------------------
        {
            TokenEmbeddingConfig embedding_config;
            embedding_config.withVocabSize( config.getVocabSize() )
                .withEmbeddingDim( static_cast<size_t>( model_dim ) )
                .withEmbeddingScale( static_cast<float>( std::sqrt( static_cast<double>( model_dim ) ) ) );

            EmbeddingType embedding( "temb", embedding_config, device );
            embedding.build( BuildContext( shape_t{ B, T }, RuntimeMode::Inference, false ) );
            loadComponentParameters( weights, embedding, "temb." );

            HostTokenTensor host_tokens( Device::Cpu(), shape_t{ B, T } );
            std::memcpy( host_tokens.data(), prompt_ids.data(), prompt_ids.size() * sizeof( int32_t ) );

            TokenTensor tokens( device, shape_t{ B, T } );
            copy( host_tokens, tokens );

            auto& embedded = embedding.forward( tokens );
            synchronizeDevice();

            hidden = toHost( embedded );
        }

        ASSERT_EQ( hidden.size(), static_cast<size_t>( B * T * model_dim ) );

        // ---- Layers, one resident at a time ---------------------------------
        std::vector<double> layer_errors;

        for ( dim_t i = 0; i < config.getNumLayers(); ++i )
        {
            const std::string layer_name = std::format( "tf_layer_{}", i );
            const bool is_global = config.isGlobalLayer( i );

            DeviceTensor input( device, shape_t{ B, T, model_dim } );
            fromHost( hidden, 0, input );

            std::shared_ptr<LocalBlock> local_block;
            std::shared_ptr<GlobalBlock> global_block;
            DecoderLayer* layer = nullptr;
            LoadableComponent* loadable = nullptr;

            if ( is_global )
            {
                global_block = std::make_shared<GlobalBlock>( layer_name, config, device );
                global_block->installSharedWorkspace( block_workspace );
                global_block->build( block_context );
                global_block->setState( gqa_workspace.state() );
                global_block->setUseFlashPrefill( false );
                global_block->setUseFlashDecode( true );

                layer = global_block.get();
                loadable = global_block.get();
            }
            else
            {
                local_block = std::make_shared<LocalBlock>( layer_name, config, device );
                local_block->installSharedWorkspace( block_workspace );
                local_block->build( block_context );
                local_block->setState( gqa_workspace.state() );
                local_block->setUseFlashPrefill( false );
                local_block->setUseFlashDecode( true );

                layer = local_block.get();
                loadable = local_block.get();
            }

            loadComponentParameters( weights, *loadable, layer_name + "." );

            auto& output = layer->prefill( input, 0 );
            synchronizeDevice();

            hidden = toHost( output );

            const std::vector<float> got = tail( hidden, model_dim );
            const std::vector<float> want = readReferenceVector( reference, std::format( "hidden_layer_{}", i ) );
            const std::vector<float> exact = readReferenceVector( truth, std::format( "hidden_layer_{}", i ) );

            ASSERT_EQ( got.size(), exact.size() ) << "width mismatch at " << layer_name;

            const double mila_error = relativeError( got, exact );
            layer_errors.push_back( mila_error );

            std::cout << std::format( "[MILA] layer_{:02d} {:<7} mila_vs_fp32={:.3e}  hf_vs_fp32={:.3e}  mila_vs_hf={:.3e}\n",
                i, is_global ? "global" : "local", mila_error, relativeError( want, exact ),
                relativeError( got, want ) ) << std::flush;
        }

        // ---- Final norm and the tied head -----------------------------------
        std::vector<float> logits;
        double final_error = 0.0;
        {
            auto rms_config = RmsNormConfig( shape_t{ model_dim } )
                .withEpsilon( config.getRMSNormEpsilon() )
                .withBias( false );

            RmsNormType final_norm( "rmsn_final", rms_config, device );
            final_norm.build( BuildContext( shape_t{ B, 1, model_dim }, RuntimeMode::Inference, false ) );
            loadComponentParameters( weights, final_norm, "rmsn_final." );

            DeviceTensor last_position( device, shape_t{ B, 1, model_dim } );
            fromHost( hidden, static_cast<size_t>( ( T - 1 ) * model_dim ), last_position );

            auto& normed = final_norm.forward( last_position );
            synchronizeDevice();

            const std::vector<float> normed_row = toHost( normed );
            const std::vector<float> exact = readReferenceVector( truth, "hidden_final" );
            final_error = relativeError( normed_row, exact );

            std::cout << std::format( "[MILA] final_norm     mila_vs_fp32={:.3e}  hf_vs_fp32={:.3e}\n",
                final_error, relativeError( readReferenceVector( reference, "hidden_final" ), exact ) ) << std::flush;

            // Tied: the head reads the embedding table, which is the only copy the weights carry.
            HeadType lm_head( "lm_head", LinearConfig( model_dim, config.getVocabSize() ).withBias( false ), device );
            lm_head.build( BuildContext( shape_t{ B, 1, model_dim }, RuntimeMode::Inference, false ) );
            lm_head.loadParameter( "weight", weights.readTensorBlob<CpuMemoryResource>( "temb.wte" ) );

            auto& logit_tensor = lm_head.forward( normed );
            synchronizeDevice();

            logits = toHost( logit_tensor );
        }

        // ---- The gate -------------------------------------------------------
        for ( size_t i = 0; i < layer_errors.size(); ++i )
        {
            EXPECT_LT( layer_errors[ i ], kRelativeTolerance ) << "layer " << i << " diverges from HuggingFace FP32";
        }

        EXPECT_LT( final_error, kRelativeTolerance ) << "final norm diverges from HuggingFace FP32";

        const std::vector<float> want_logits = readReferenceVector( reference, "logits_last" );
        const std::vector<float> exact_logits = readReferenceVector( truth, "logits_last" );
        ASSERT_EQ( logits.size(), exact_logits.size() );

        const double logit_error = relativeError( logits, exact_logits );

        std::cout << std::format( "[MILA] logits  argmax={} (HF BF16 {}, HF FP32 {})  mila_vs_fp32={:.3e}  hf_vs_fp32={:.3e}\n",
            argmax( logits ), argmax( want_logits ), argmax( exact_logits ), logit_error,
            relativeError( want_logits, exact_logits ) ) << std::flush;

        EXPECT_LT( logit_error, kRelativeTolerance ) << "logits diverge from HuggingFace FP32";
        EXPECT_EQ( argmax( logits ), argmax( exact_logits ) ) << "next token differs from HuggingFace FP32";
        EXPECT_EQ( argmax( logits ), argmax( want_logits ) ) << "next token differs from HuggingFace BF16";
    }
}
