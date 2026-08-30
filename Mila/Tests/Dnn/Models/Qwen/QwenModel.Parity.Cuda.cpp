/**
 * @file QwenModel.Parity.Cuda.cpp
 * @brief Layer-streamed HuggingFace parity for the Qwen 3.8 27B stack (Phase 4).
 *
 * WHY THIS IS NOT A `QwenModel::fromPretrained` TEST. The BF16 artifact is 50 GiB against a
 * 12 GiB card, so the whole stack cannot be resident and `GemmaModel.Parity.Cuda.cpp`'s method
 * -- load the model, generate, compare tokens -- is unavailable at this scale. This test holds
 * ONE decoder block at a time: it constructs block i, loads only that block's tensors from the
 * artifact, prefills the prompt through it, carries the hidden state out to the host, and
 * destroys it. Peak residency is one block plus the embedding and head tables.
 *
 * It mirrors, step for step, what the Python reference driver does on the HuggingFace side:
 *
 *   Tools/Converters/Qwen/qwen38_BF16/hf_qwen_layer_stream.py
 *
 * Produce the two inputs with:
 *   python Qwen/convert_weights.py --model Qwen/Qwen3.8-27B \
 *       --output <repo>/Data/Models/Qwen/qwen38_27b_bf16.bin
 *   python Qwen/qwen38_BF16/hf_qwen_layer_stream.py --model Qwen/Qwen3.8-27B \
 *       --output <repo>/Data/Models/Qwen/qwen38_ref.bin
 *
 * The reference is itself a MILA container, so it is read with the same reader the weights are.
 *
 * FOUR THINGS THIS DOES DIFFERENTLY FROM A REAL LOAD, each deliberate:
 *
 *  - Every component owns its own ExecutionContext, because `setExecutionContext` is protected
 *    and only a parent composite may share one. Each stage therefore runs on its own stream and
 *    the harness synchronizes between stages. Correct, and slower than the model -- which does
 *    not matter here and would matter in `ProfileModel`.
 *  - The hidden state travels between layers THROUGH THE HOST. A block's output buffer is owned
 *    by the block and dies with it, which is the price of holding one at a time.
 *  - Prefill runs in ONE chunk. The transformer picks a chunk from an activation budget; a
 *    short prompt needs no ladder, and chunk-boundary equivalence is already pinned by the
 *    Phase 3 block tests rather than being this test's subject.
 *  - Flash prefill is off, and the GQA score width is the full context to match. The two MUST
 *    agree or the cuBLASLt path overflows a narrow score buffer.
 *
 * Requires the full 50 GiB artifact and is skipped without it, so it never runs in CI.
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

// The consumer-side CUDA block instantiation trap: Mila.ixx exports IExecutionContext but not
// ExecutionContext, and building a CUDA attention block reaches CudaGqaOp::build, which needs
// the complete type. Filed in BACKLOG against the export surface; the same line appears in
// Qwen.Block.Cuda.cpp for the same reason.

namespace Mila::Tests::Dnn::Models
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;

    namespace fs = std::filesystem;

    namespace
    {
        constexpr auto kPrecision = TensorDataType::BF16;

        using MR = typename DeviceTypeTraits<DeviceType::Cuda>::memory_resource;

        using QwenBf16 = QwenModel<DeviceType::Cuda, kPrecision>;
        using AttentionBlock = QwenAttentionBlock<DeviceType::Cuda, kPrecision>;
        using DeltaNetBlock = QwenDeltaNetBlock<DeviceType::Cuda, kPrecision>;
        using EmbeddingType =
            TokenEmbedding<DeviceType::Cuda, TensorDataType::INT32, kPrecision, NoWeightQuant>;
        using RmsNormType = RmsNorm<DeviceType::Cuda, kPrecision>;
        using HeadType = Linear<DeviceType::Cuda, kPrecision>;
        using DecoderLayer = ITransformerBlock<DeviceType::Cuda, kPrecision>;
        using ComponentType = Component<DeviceType::Cuda, kPrecision>;
        using CompositeType = CompositeComponent<DeviceType::Cuda, kPrecision>;

        using DeviceTensor = Tensor<kPrecision, MR>;
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using TokenTensor = Tensor<TensorDataType::INT32, MR>;
        using HostTokenTensor = Tensor<TensorDataType::INT32, CpuMemoryResource>;

        /**
         * @brief Wait for every stream on the device.
         *
         * Blunt on purpose. `Component::getExecutionContext()` is protected, so a harness that
         * constructs components independently cannot reach the stream each one owns -- and
         * because they are independent, each stage IS on a different stream. A device-wide
         * synchronize is the only barrier available here and costs nothing at this cadence.
         */
        void synchronizeDevice()
        {
            ASSERT_EQ( cudaDeviceSynchronize(), cudaSuccess );
        }

        // Relative L2 against the reference, per layer. BF16 storage carries ~3 decimal digits
        // and the two sides differ in kernel order and rounding, so exact agreement is not the
        // bar -- but this is now a real ratchet rather than a recorded baseline.
        //
        // Measured 2026-08-19 after the Qwen rotary-frequency fix: the per-layer error is flat
        // across the stack with a peak of 1.380e-2, and Mila sits CLOSER to an FP32 reference
        // than HuggingFace's own BF16 run at essentially every layer (logits 9.677e-3 against
        // 1.387e-2), because its cos/sin cache and rotation are FP32 where the reference's are
        // BF16. Bound set ~45% above the measured peak.
        //
        // It previously stood at 1.0e-1 against a 7.5e-2 peak, while the rotary frequencies
        // were spread over head_dim instead of rotary_dim. A bound loose enough to pass a real
        // defect is what let that sit; treat any regression toward this one as a defect rather
        // than as headroom. Specifications/Qwen3.8.md carries the profile.
        constexpr double kHiddenStateTolerance = 2.0e-2;

        fs::path artifactPath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_bf16.bin";
        }

        fs::path referencePath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_ref.bin";
        }

        /**
         * @brief Optional second reference, produced by the driver at --dtype float32.
         *
         * Present or not, the test asserts against the BF16 reference. What this adds is the
         * only comparison that separates ACCURACY from AGREEMENT: the BF16 reference is not
         * ground truth, so anywhere Mila is more careful than HuggingFace -- its RoPE keeps the
         * cos/sin cache and the rotation in FP32 where the reference does both in BF16 -- the
         * measured divergence is the reference's rounding, not Mila's. Same weights on both
         * sides; only the arithmetic width differs.
         */
        fs::path fp32ReferencePath()
        {
            return fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_ref_fp32.bin";
        }

        std::vector<float> readReferenceVector( PretrainedModelReader& reader, const std::string& name )
        {
            auto blob = reader.readTensorBlob<CpuMemoryResource>( name );

            std::vector<float> values( blob.sizeBytes() / sizeof( float ) );
            std::memcpy( values.data(), blob.data(), blob.sizeBytes() );

            return values;
        }

        std::vector<int32_t> readPromptIds( PretrainedModelReader& reader )
        {
            auto blob = reader.readTensorBlob<CpuMemoryResource>( "prompt_ids" );

            std::vector<int32_t> ids( blob.sizeBytes() / sizeof( int32_t ) );
            std::memcpy( ids.data(), blob.data(), blob.sizeBytes() );

            return ids;
        }

        /// Relative L2 error -- scale-free, so one bound covers every layer even though the
        /// residual stream grows by an order of magnitude down the stack.
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
            const size_t count = static_cast<size_t>( host.size() );

            std::memcpy( host.data(), values.data() + offset, count * sizeof( float ) );

            copy( host, tensor );
        }

        std::vector<float> tail( const std::vector<float>& values, dim_t width )
        {
            return std::vector<float>( values.end() - width, values.end() );
        }

        /// Load every artifact tensor under `prefix` into the component named for it.
        void loadComponentParameters(
            PretrainedModelReader& reader, ComponentType& component, const std::string& prefix )
        {
            size_t loaded = 0;

            for ( const auto& name : reader.getTensorNames() )
            {
                if ( !name.starts_with( prefix ) )
                    continue;

                // A leaf component's tensor has no path left after the prefix ("temb.wte" ->
                // "wte"); a composite's still carries one ("tf_layer_0.input_norm.weight" ->
                // "input_norm.weight").
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

            // A prefix that matched nothing leaves the component on uninitialized memory and
            // produces a plausible hidden state rather than an error.
            ASSERT_GT( loaded, 0u ) << "No artifact tensors matched prefix '" << prefix << "'";
        }
    }

    namespace
    {
        /**
         * @brief Which layer the reference instrumented, derived rather than declared.
         *
         * `stage_block_input` is that layer's input, so its last row is by construction the
         * previous layer's recorded last-token hidden state. Searching for the match avoids a
         * second source of truth for the index -- a constant here could silently disagree with
         * the `--dump-layer` the reference was produced with.
         */
        dim_t findDumpedLayer( PretrainedModelReader& reference, dim_t num_layers, dim_t model_dim )
        {
            const std::vector<float> block_input =
                readReferenceVector( reference, "stage_block_input" );
            const std::vector<float> last = tail( block_input, model_dim );

            for ( dim_t i = 1; i < num_layers; ++i )
            {
                const std::string name = std::format( "hidden_layer_{}", i - 1 );

                if ( !reference.hasTensor( name ) )
                    continue;

                if ( relativeError( last, readReferenceVector( reference, name ) ) == 0.0 )
                    return i;
            }

            return -1;
        }
    }

    class QwenParityCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
                GTEST_SKIP() << "No CUDA device available";

            artifact_ = artifactPath();
            reference_ = referencePath();

            if ( !fs::exists( artifact_ ) )
                GTEST_SKIP() << "Qwen artifact not present at: " << artifact_.string();

            if ( !fs::exists( reference_ ) )
                GTEST_SKIP() << "HF reference not present at: " << reference_.string()
                             << " -- produce it with Qwen/qwen38_BF16/hf_qwen_layer_stream.py";
        }

        fs::path artifact_;
        fs::path reference_;
    };

    // The Phase 4 gate: every layer's last-token hidden state against the HuggingFace
    // reference, and the last-position argmax. A failure localizes -- the first layer whose
    // relative error jumps is where Mila and the reference part company.
    TEST_F( QwenParityCudaTests, LayerStream_MatchesHuggingFaceReference )
    {
        PretrainedModelReader weights( artifact_ );
        PretrainedModelReader reference( reference_ );

        // Optional FP32 truth. When present, every layer reports THREE numbers: Mila-BF16 and
        // HF-BF16 each against FP32, plus Mila against HF. The first two are what decide
        // whether a divergence is Mila's error or the reference's.
        const bool has_fp32 = fs::exists( fp32ReferencePath() );
        std::unique_ptr<PretrainedModelReader> truth;

        if ( has_fp32 )
            truth = std::make_unique<PretrainedModelReader>( fp32ReferencePath() );

        const QwenConfig config = QwenBf16::configFromMetadata( weights.getPretrainedMetadata() );

        const std::vector<int32_t> prompt_ids = readPromptIds( reference );
        ASSERT_FALSE( prompt_ids.empty() );

        const dim_t B = 1;
        const dim_t T = static_cast<dim_t>( prompt_ids.size() );
        const dim_t model_dim = config.getModelDim();
        const DeviceId device{ DeviceType::Cuda, 0 };

        const BuildContext block_context =
            BuildContext( shape_t{ B, T, model_dim }, RuntimeMode::Inference, false )
                .withPrefillSize( T );

        auto block_workspace = makeQwenAttentionBlockWorkspace<DeviceType::Cuda, kPrecision>(
            config, device, B, T, "parity.block_ws." );

        // score_width = T because flash prefill is switched off below; the two must agree.
        auto gqa_workspace = makeQwenGqaWorkspace<DeviceType::Cuda, kPrecision>(
            config, device, B, T, T, T, "parity.gqa_ws." );

        // The full [B, T, model_dim] state, carried on the host across block lifetimes.
        std::vector<float> hidden;

        // ---- Embedding ------------------------------------------------------
        {
            TokenEmbeddingConfig embedding_config;
            embedding_config.withVocabSize( config.getVocabSize() )
                .withEmbeddingDim( static_cast<size_t>( model_dim ) );

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
        layer_errors.reserve( static_cast<size_t>( config.getNumLayers() ) );

        for ( dim_t i = 0; i < config.getNumLayers(); ++i )
        {
            const std::string layer_name = std::format( "tf_layer_{}", i );
            const bool is_attention = config.isFullAttentionLayer( i );

            DeviceTensor input( device, shape_t{ B, T, model_dim } );
            fromHost( hidden, 0, input );

            std::shared_ptr<AttentionBlock> attention_block;
            std::shared_ptr<DeltaNetBlock> deltanet_block;
            DecoderLayer* layer = nullptr;
            ComponentType* loadable = nullptr;

            if ( is_attention )
            {
                attention_block = std::make_shared<AttentionBlock>( layer_name, config, device );
                attention_block->installSharedWorkspace( block_workspace );
                attention_block->build( block_context );
                attention_block->setState( gqa_workspace.state() );
                attention_block->setUseFlashPrefill( false );
                attention_block->setUseFlashDecode( true );

                layer = attention_block.get();
                loadable = attention_block.get();
            }
            else
            {
                deltanet_block = std::make_shared<DeltaNetBlock>( layer_name, config, device );
                deltanet_block->build( block_context );

                layer = deltanet_block.get();
                loadable = deltanet_block.get();
            }

            loadComponentParameters( weights, *loadable, layer_name + "." );

            auto& output = layer->prefill( input, 0 );
            synchronizeDevice();

            hidden = toHost( output );

            const std::vector<float> got = tail( hidden, model_dim );
            const std::vector<float> want =
                readReferenceVector( reference, std::format( "hidden_layer_{}", i ) );

            ASSERT_EQ( got.size(), want.size() ) << "width mismatch at " << layer_name;

            const double error = relativeError( got, want );
            layer_errors.push_back( error );

            std::string line = std::format( "[MILA] layer_{:02d} {:<17} vs_hf={:.3e}",
                i, is_attention ? "full_attention" : "linear_attention", error );

            if ( truth )
            {
                const std::vector<float> exact =
                    readReferenceVector( *truth, std::format( "hidden_layer_{}", i ) );

                const double mila_error = relativeError( got, exact );
                const double reference_error = relativeError( want, exact );

                line += std::format( "  mila_vs_fp32={:.3e}  hf_vs_fp32={:.3e}  {}",
                    mila_error, reference_error,
                    mila_error <= reference_error ? "MILA_CLOSER" : "HF_CLOSER" );
            }

            std::cout << line << "\n" << std::flush;
        }

        // ---- Final norm and head --------------------------------------------
        std::vector<float> logits;
        {
            auto rms_config = RmsNormConfig( shape_t{ model_dim } )
                .withEpsilon( config.getRMSNormEpsilon() )
                .withBias( false )
                .withUnitOffset( 1.0f );

            RmsNormType final_norm( "rmsn_final", rms_config, device );
            final_norm.build(
                BuildContext( shape_t{ B, 1, model_dim }, RuntimeMode::Inference, false ) );
            loadComponentParameters( weights, final_norm, "rmsn_final." );

            DeviceTensor last_position( device, shape_t{ B, 1, model_dim } );
            fromHost( hidden, static_cast<size_t>( ( T - 1 ) * model_dim ), last_position );

            auto& normed = final_norm.forward( last_position );
            synchronizeDevice();

            const std::vector<float> normed_row = toHost( normed );
            const std::vector<float> want_final = readReferenceVector( reference, "hidden_final" );

            const double final_error = relativeError( normed_row, want_final );
            std::string line = std::format( "[MILA] final_norm{:<12} vs_hf={:.3e}", "", final_error );

            if ( truth )
            {
                const std::vector<float> exact = readReferenceVector( *truth, "hidden_final" );
                const double mila_error = relativeError( normed_row, exact );
                const double reference_error = relativeError( want_final, exact );

                line += std::format( "  mila_vs_fp32={:.3e}  hf_vs_fp32={:.3e}  {}",
                    mila_error, reference_error,
                    mila_error <= reference_error ? "MILA_CLOSER" : "HF_CLOSER" );
            }

            std::cout << line << "\n" << std::flush;

            EXPECT_LT( final_error, kHiddenStateTolerance ) << "final norm diverges from HF";

            auto head_config = LinearConfig( model_dim, config.getVocabSize() ).withBias( false );

            HeadType lm_head( "lm_head", head_config, device );
            lm_head.build( BuildContext( shape_t{ B, 1, model_dim }, RuntimeMode::Inference, false ) );
            loadComponentParameters( weights, lm_head, "lm_head." );

            auto& logit_tensor = lm_head.forward( normed );
            synchronizeDevice();

            logits = toHost( logit_tensor );
        }

        // ---- Stage attribution, on a COMMON input ---------------------------
        //
        // The per-layer numbers above cannot separate "this block adds error" from "this
        // block amplifies what arrived", because Mila's input to layer N is already its own
        // drifted state. Here the block is rebuilt and fed the REFERENCE's input for that
        // layer, so both sides start identical and every difference is that block's own.
        if ( truth && reference.hasTensor( "stage_block_input" ) )
        {
            const dim_t dumped = findDumpedLayer( reference, config.getNumLayers(), model_dim );
            ASSERT_GE( dumped, 0 ) << "stage_block_input matches no layer's recorded input";
            ASSERT_TRUE( config.isFullAttentionLayer( dumped ) )
                << "stage attribution reads the attention block's shared workspace; layer "
                << dumped << " is a DeltaNet block, which self-allocates its transients";

            std::cout << std::format( "\n[MILA] stage attribution, layer {} on a COMMON input\n",
                dumped ) << std::flush;

            // A workspace of its own: the one above still holds layer 63's values, and
            // reusing it would compare against whatever survived.
            auto stage_workspace = makeQwenAttentionBlockWorkspace<DeviceType::Cuda, kPrecision>(
                config, device, B, T, "parity.stage_ws." );
            auto stage_gqa = makeQwenGqaWorkspace<DeviceType::Cuda, kPrecision>(
                config, device, B, T, T, T, "parity.stage_gqa." );

            auto block = std::make_shared<AttentionBlock>(
                std::format( "tf_layer_{}", dumped ), config, device );
            block->installSharedWorkspace( stage_workspace );
            block->build( block_context );
            block->setState( stage_gqa.state() );
            block->setUseFlashPrefill( false );
            block->setUseFlashDecode( true );
            loadComponentParameters( weights, *block, std::format( "tf_layer_{}.", dumped ) );

            DeviceTensor stage_input( device, shape_t{ B, T, model_dim } );
            fromHost( readReferenceVector( reference, "stage_block_input" ), 0, stage_input );

            auto& stage_output = block->prefill( stage_input, 0 );
            synchronizeDevice();

            // Mila slot -> reference stage. Only slots with an addressable HF counterpart
            // appear; the fused qkv and the split q/k/v have no single reference tensor.
            const std::vector<std::pair<const char*, const DeviceTensor*>> pairs = {
                { "input_norm",     stage_workspace.normed.get() },
                // The q_norm/k_norm slots hold POST-RoPE values: the block takes a view over
                // them and rotates in place (Qwen.AttentionBlock.ixx:829). Comparing them against the
                // reference's q_norm hook reports ~36% divergence that is entirely the
                // rotation, so the reference captures q_roped/k_roped instead.
                { "q_roped",        stage_workspace.q_normed.get() },
                { "k_roped",        stage_workspace.k_normed.get() },
                { "gated",          stage_workspace.gated.get() },
                { "o_proj",         stage_workspace.o.get() },
                { "post_attn_norm", stage_workspace.ffn_in.get() },
                { "ffn_act",        stage_workspace.ffn_act.get() },
                { "ffn_down",       stage_workspace.ffn_down.get() },
            };

            for ( const auto& [name, slot] : pairs )
            {
                const std::string stage_name = std::string( "stage_" ) + name;

                if ( !reference.hasTensor( stage_name ) || !truth->hasTensor( stage_name ) )
                    continue;

                const std::vector<float> got = toHost( *slot );
                const std::vector<float> want = readReferenceVector( reference, stage_name );
                const std::vector<float> exact = readReferenceVector( *truth, stage_name );

                if ( got.size() != want.size() )
                {
                    std::cout << std::format( "[MILA]   {:<16} SHAPE {} vs {} -- not comparable\n",
                        name, got.size(), want.size() ) << std::flush;
                    continue;
                }

                std::cout << std::format(
                    "[MILA]   {:<16} mila_vs_fp32={:.3e}  hf_vs_fp32={:.3e}  {}\n",
                    name, relativeError( got, exact ), relativeError( want, exact ),
                    relativeError( got, exact ) <= relativeError( want, exact )
                        ? "MILA_CLOSER" : "HF_CLOSER" ) << std::flush;

                // A relative error this size is structural, not rounding, and the model
                // still predicts correctly -- which points at the COMPARISON's layout
                // rather than at the values. Per-head errors separate the two: a head
                // permutation leaves each head's values intact and misplaces them, so
                // some heads match well and others not at all, while a genuine value
                // defect is uniformly bad across heads.
                if ( relativeError( got, exact ) > 0.1 )
                {
                    const dim_t head_dim = config.getHeadDim();
                    const dim_t heads = static_cast<dim_t>( got.size() ) / ( T * head_dim );

                    std::string per_head;

                    for ( dim_t h = 0; h < std::min<dim_t>( heads, 8 ); ++h )
                    {
                        std::vector<float> a;
                        std::vector<float> b;

                        for ( dim_t t = 0; t < T; ++t )
                        {
                            const size_t base =
                                static_cast<size_t>( t ) * heads * head_dim + h * head_dim;

                            a.insert( a.end(), got.begin() + base, got.begin() + base + head_dim );
                            b.insert( b.end(), exact.begin() + base, exact.begin() + base + head_dim );
                        }

                        per_head += std::format( " h{}={:.2e}", h, relativeError( a, b ) );
                    }

                    std::cout << std::format( "[MILA]     per-head ({} heads):{}\n", heads, per_head )
                              << std::flush;
                }
            }

            // ---- The projection split, before any norm or rotation ----------
            //
            // The four slots the fused qkv is split into. This is the earliest point after
            // `input_norm` (which is bit-identical) at which the two sides can be compared,
            // so it separates the projection and the split from everything downstream.
            //
            // `q` and `gate` need the reference de-interleaved to match: the checkpoint
            // stores the query projection as [q_h0 | gate_h0 | q_h1 | gate_h1 | ...] and the
            // converter rearranges it into contiguous halves. That transform has no
            // counterpart in any other family and has been verified against the checkpoint
            // but never against a running block -- which is exactly what this row does.
            if ( reference.hasTensor( "stage_q_proj" ) && truth->hasTensor( "stage_q_proj" ) )
            {
                const dim_t heads = config.getNumHeads();
                const dim_t head_dim = config.getHeadDim();

                auto deinterleave = [&]( const std::vector<float>& source, bool want_gate )
                {
                    std::vector<float> out;
                    out.reserve( static_cast<size_t>( T ) * heads * head_dim );

                    for ( dim_t t = 0; t < T; ++t )
                    {
                        for ( dim_t h = 0; h < heads; ++h )
                        {
                            const size_t base =
                                ( static_cast<size_t>( t ) * heads + h ) * 2 * head_dim
                                + ( want_gate ? head_dim : 0 );

                            out.insert( out.end(), source.begin() + base,
                                source.begin() + base + head_dim );
                        }
                    }

                    return out;
                };

                const std::vector<float> q_proj_exact = readReferenceVector( *truth, "stage_q_proj" );
                const std::vector<float> q_proj_want = readReferenceVector( reference, "stage_q_proj" );

                const std::vector<std::tuple<const char*, const DeviceTensor*, std::vector<float>,
                    std::vector<float>>> splits = {
                    { "split_q",    stage_workspace.q.get(),
                      deinterleave( q_proj_exact, false ), deinterleave( q_proj_want, false ) },
                    { "split_gate", stage_workspace.gate.get(),
                      deinterleave( q_proj_exact, true ), deinterleave( q_proj_want, true ) },
                    { "split_k",    stage_workspace.k.get(),
                      readReferenceVector( *truth, "stage_k_proj" ),
                      readReferenceVector( reference, "stage_k_proj" ) },
                    { "split_v",    stage_workspace.v.get(),
                      readReferenceVector( *truth, "stage_v_proj" ),
                      readReferenceVector( reference, "stage_v_proj" ) },
                };

                for ( const auto& [name, slot, exact, want] : splits )
                {
                    const std::vector<float> got = toHost( *slot );

                    if ( got.size() != exact.size() )
                    {
                        std::cout << std::format( "[MILA]   {:<16} SHAPE {} vs {}\n",
                            name, got.size(), exact.size() ) << std::flush;
                        continue;
                    }

                    std::cout << std::format(
                        "[MILA]   {:<16} mila_vs_fp32={:.3e}  hf_vs_fp32={:.3e}  {}\n",
                        name, relativeError( got, exact ), relativeError( want, exact ),
                        relativeError( got, exact ) <= relativeError( want, exact )
                            ? "MILA_CLOSER" : "HF_CLOSER" ) << std::flush;
                }
            }

            const std::vector<float> block_out = toHost( stage_output );
            const std::vector<float> want_out = readReferenceVector( reference, "stage_block_output" );
            const std::vector<float> exact_out = readReferenceVector( *truth, "stage_block_output" );

            std::cout << std::format(
                "[MILA]   {:<16} mila_vs_fp32={:.3e}  hf_vs_fp32={:.3e}\n\n",
                "block_output", relativeError( block_out, exact_out ),
                relativeError( want_out, exact_out ) ) << std::flush;
        }

        // ---- The assertions -------------------------------------------------
        for ( size_t i = 0; i < layer_errors.size(); ++i )
        {
            EXPECT_LT( layer_errors[ i ], kHiddenStateTolerance )
                << "layer " << i << " diverges from the HF reference";
        }

        const std::vector<float> want_logits = readReferenceVector( reference, "logits_last" );
        ASSERT_EQ( logits.size(), want_logits.size() );

        const auto got_argmax = std::distance( logits.begin(), std::ranges::max_element( logits ) );
        const auto want_argmax =
            std::distance( want_logits.begin(), std::ranges::max_element( want_logits ) );

        std::cout << std::format( "[MILA] argmax={} (HF {}), logits vs_hf={:.3e}\n",
            got_argmax, want_argmax, relativeError( logits, want_logits ) ) << std::flush;

        if ( truth )
        {
            const std::vector<float> exact = readReferenceVector( *truth, "logits_last" );
            const auto exact_argmax =
                std::distance( exact.begin(), std::ranges::max_element( exact ) );

            const double mila_error = relativeError( logits, exact );
            const double reference_error = relativeError( want_logits, exact );

            std::cout << std::format(
                "[MILA] logits  mila_vs_fp32={:.3e}  hf_vs_fp32={:.3e}  {}  (fp32 argmax={})\n",
                mila_error, reference_error,
                mila_error <= reference_error ? "MILA_CLOSER" : "HF_CLOSER", exact_argmax )
                << std::flush;

            EXPECT_EQ( got_argmax, exact_argmax ) << "next-token prediction differs from FP32 truth";
        }

        // The gate Phase 4 exists for: the same next token as the reference, on real weights,
        // through all 64 layers at BF16.
        EXPECT_EQ( got_argmax, want_argmax ) << "next-token prediction differs from HuggingFace";
    }
}
