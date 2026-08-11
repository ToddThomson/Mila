/**
 * @file Gemma.Block.Cuda.cpp
 * @brief Structural tests for GemmaBlock<DeviceType::Cuda, FP32, kGlobal>.
 *
 * GemmaBlock is a Llama-family decoder block (GQA, RMSNorm, RoPE, gated FFN) and,
 * like LlamaBlock, is INFERENCE-ONLY: it implements IDecoderLayer (prefill/decode),
 * not Component forward/backward. There is no standalone Llama.Block test to mirror,
 * so this suite covers the block's structural contract directly:
 *   - construction / validation and the [B, T, model_dim] build contract,
 *   - the component graph and type identity,
 *   - and — the point of Gemma — the two instantiations' GEOMETRY: the decoupled
 *     head_dim and the K=V global packing that diverge from Llama.
 *
 * Numerics and the prefill/decode execution path are intentionally NOT exercised
 * here. prefill/decode need the shared GqaState workspace that the transformer owns,
 * so execution (logits shape) is covered in Gemma.Cuda.cpp at the network level —
 * the same way LlamaTransformer is validated rather than a standalone LlamaBlock.
 * Numeric correctness of the new attention/RoPE/QK-norm paths waits on the GQA
 * operation-level oracle + the HF parity oracle (BACKLOG: Correctness-oracle
 * dependency / Gemma Step 5f).
 *
 * CUDA device tests -- skipped when no CUDA device is present.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <stdexcept>
#include <system_error>

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Gemma
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    namespace
    {
        using LocalBlock  = Mila::Dnn::GemmaBlock<DeviceType::Cuda, TensorDataType::FP32, false>;
        using GlobalBlock = Mila::Dnn::GemmaBlock<DeviceType::Cuda, TensorDataType::FP32, true>;

        // Small but structurally-faithful Gemma geometry. head_dim (32) is decoupled
        // from embedding/num_heads (64/4 = 16), and the global layer widens head_dim
        // to 64 with a single shared KV head (K=V) -- the two Gemma deltas under test.
        constexpr int64_t kModelDim       = 64;
        constexpr int64_t kLayers         = 2;
        constexpr int64_t kHeads          = 4;
        constexpr int64_t kKVHeads        = 2;
        constexpr int64_t kHeadDim        = 32;
        constexpr int64_t kGlobalHeadDim  = 64;
        constexpr int64_t kGlobalKVHeads  = 1;
        constexpr int64_t kHidden         = 128;
        constexpr int64_t kVocab          = 128;
        constexpr int64_t kMaxSeq         = 32;
        constexpr int64_t kWindow         = 8;
        constexpr int64_t kGlobalRotary   = 32;

        // Derived widths the converter and block must agree on.
        constexpr int64_t kLocalQProj     = kHeads * kHeadDim;                       // 128
        constexpr int64_t kLocalKVProj    = kKVHeads * kHeadDim;                     //  64
        constexpr int64_t kLocalPackedQKV = (kHeads + 2 * kKVHeads) * kHeadDim;      // 256
        constexpr int64_t kGlobalQProj    = kHeads * kGlobalHeadDim;                 // 256
        constexpr int64_t kGlobalKVProj   = kGlobalKVHeads * kGlobalHeadDim;         //  64
        constexpr int64_t kGlobalPackedQKV = (kHeads + kGlobalKVHeads) * kGlobalHeadDim; // 320 (K=V: no V section)

        // One GemmaConfig drives both instantiations; kGlobal selects geometry from it.
        GemmaConfig smallConfig()
        {
            return GemmaConfig( kModelDim, kLayers )
                .withVocabularyLength( kVocab )
                .withNumHeads( kHeads )
                .withNumKVHeads( kKVHeads )
                .withHeadDim( kHeadDim )
                .withGlobalHeadDim( kGlobalHeadDim )
                .withNumGlobalKVHeads( kGlobalKVHeads )
                .withKeyEqualsValue( true )
                .withHiddenDimension( kHidden )
                .withMaxSequenceLength( kMaxSeq )
                .withRMSNormEpsilon( 1e-6f )
                .withWindow( kWindow )
                .withSlidingWindowPattern( 6 )
                .withGlobalRotaryDim( kGlobalRotary )
                .withRoPETheta( 10000.0f )
                .withGlobalRoPETheta( 1000000.0f )
                .withFinalLogitSoftcapping( 30.0f );
        }

        static_assert( LocalBlock::getDeviceType() == DeviceType::Cuda );
        static_assert( LocalBlock::getPrecision() == TensorDataType::FP32 );
        static_assert( LocalBlock::isGlobal() == false );
        static_assert( GlobalBlock::isGlobal() == true );
    }

    class GemmaBlockCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }
        }

        template<typename Block>
        std::unique_ptr<Block> builtBlock( RuntimeMode mode )
        {
            auto block = std::make_unique<Block>( "gemma_block", smallConfig(), Device::Cuda( 0 ) );

            // The transformer threads the prefill chunk to the block; a standalone
            // build must supply it (prefill_size defaults to 0, sizing scratch empty).
            block->build( BuildContext( shape_t{ batch_, seq_, kModelDim }, mode ).withPrefillSize( seq_ ) );

            return block;
        }

        static constexpr int64_t batch_ = 1;
        static constexpr int64_t seq_ = 4;
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( GemmaBlockCudaTests, ConstructLocal_Succeeds )
    {
        LocalBlock block( "gemma_local", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( block.getName(), "gemma_local" );
        EXPECT_EQ( block.getDeviceId().type, DeviceType::Cuda );
    }

    TEST_F( GemmaBlockCudaTests, ConstructGlobal_Succeeds )
    {
        GlobalBlock block( "gemma_global", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( block.getName(), "gemma_global" );
        EXPECT_EQ( block.getDeviceId().type, DeviceType::Cuda );
    }

    TEST_F( GemmaBlockCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( LocalBlock( "gemma_local", smallConfig(), Device::Cpu() ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( GemmaBlockCudaTests, BuildLocal_SetsIsBuilt )
    {
        auto block = builtBlock<LocalBlock>( RuntimeMode::Inference );

        EXPECT_TRUE( block->isBuilt() );
    }

    TEST_F( GemmaBlockCudaTests, BuildGlobal_SetsIsBuilt )
    {
        auto block = builtBlock<GlobalBlock>( RuntimeMode::Inference );

        EXPECT_TRUE( block->isBuilt() );
    }

    TEST_F( GemmaBlockCudaTests, BuildLocal_AllocatesParameters )
    {
        auto block = builtBlock<LocalBlock>( RuntimeMode::Inference );

        EXPECT_GT( block->parameterCount(), 0 );
    }

    TEST_F( GemmaBlockCudaTests, BuildGlobal_AllocatesParameters )
    {
        auto block = builtBlock<GlobalBlock>( RuntimeMode::Inference );

        EXPECT_GT( block->parameterCount(), 0 );
    }

    TEST_F( GemmaBlockCudaTests, Build_ThrowsOnNonRank3Input )
    {
        LocalBlock block( "gemma_local", smallConfig(), Device::Cuda( 0 ) );

        // Rank-2 input violates the [B, T, model_dim] contract.
        EXPECT_THROW( block.build( BuildContext( shape_t{ seq_, kModelDim }, RuntimeMode::Inference ).withPrefillSize( seq_ ) ),
            std::invalid_argument );
    }

    TEST_F( GemmaBlockCudaTests, Build_ThrowsOnModelDimMismatch )
    {
        LocalBlock block( "gemma_local", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_THROW( block.build( BuildContext( shape_t{ batch_, seq_, kModelDim + 1 }, RuntimeMode::Inference ).withPrefillSize( seq_ ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // C. Geometry — the Gemma deltas (decoupled head_dim + K=V global packing)
    // ====================================================================

    TEST_F( GemmaBlockCudaTests, LocalGeometry_SlidingWidthsAndWindow )
    {
        LocalBlock block( "gemma_local", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_FALSE( block.isGlobal() );
        EXPECT_EQ( block.headDim(), kHeadDim );
        EXPECT_EQ( block.numKVHeads(), kKVHeads );
        EXPECT_FALSE( block.keyEqualsValue() );
        EXPECT_EQ( block.window(), kWindow );
        EXPECT_EQ( block.qProjWidth(), kLocalQProj );
        EXPECT_EQ( block.kvProjWidth(), kLocalKVProj );
        EXPECT_EQ( block.packedQKVWidth(), kLocalPackedQKV );
    }

    TEST_F( GemmaBlockCudaTests, GlobalGeometry_WidenedHeadDimAndKEqualsV )
    {
        GlobalBlock block( "gemma_global", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_TRUE( block.isGlobal() );
        EXPECT_EQ( block.headDim(), kGlobalHeadDim );
        EXPECT_EQ( block.numKVHeads(), kGlobalKVHeads );
        EXPECT_TRUE( block.keyEqualsValue() );
        EXPECT_EQ( block.window(), 0 );                 // global layers are unbounded
        EXPECT_EQ( block.qProjWidth(), kGlobalQProj );
        EXPECT_EQ( block.kvProjWidth(), kGlobalKVProj );
        // K=V: the packed QKV drops the V section -> (num_heads + num_kv_heads) * head_dim.
        EXPECT_EQ( block.packedQKVWidth(), kGlobalPackedQKV );
    }

    TEST_F( GemmaBlockCudaTests, HeadDim_IsDecoupledFromResidualStream )
    {
        LocalBlock block( "gemma_local", smallConfig(), Device::Cuda( 0 ) );

        // The Gemma break: head_dim (32) is NOT model_dim / num_heads (64 / 4 = 16),
        // so the Q-projection width differs from the residual stream.
        EXPECT_NE( block.headDim(), kModelDim / kHeads );
        EXPECT_NE( block.qProjWidth(), kModelDim );
    }

    // ====================================================================
    // G. Components & Type identity
    // ====================================================================

    TEST_F( GemmaBlockCudaTests, GetComponents_ReturnsCorrectChildrenSize )
    {
        LocalBlock block( "gemma_local", smallConfig(), Device::Cuda( 0 ) );

        // 7 norms (input/q/k/v/post_attn/pre_ffn/post_ffn) + qkv_proj + rope + gqa +
        // o_proj + res_1 + fc_gate_up + geglu + fc_down + res_2.
        EXPECT_EQ( block.getComponents().size(), 16u );
    }

    TEST_F( GemmaBlockCudaTests, GlobalBlock_HasSameGraphShape )
    {
        GlobalBlock block( "gemma_global", smallConfig(), Device::Cuda( 0 ) );

        // The global block has the same component graph; only widths differ.
        EXPECT_EQ( block.getComponents().size(), 16u );
    }

    TEST_F( GemmaBlockCudaTests, GetType_IsTransformer )
    {
        LocalBlock block( "gemma_local", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( block.getType(), ComponentType::Transformer );
    }

    // ====================================================================
    // Serialization
    // ====================================================================
    //
    // GemmaBlock is the only composite in the tree that owns a parameter of its
    // own -- layer_scalar, the Gemma 4 Unified per-layer output scale. Every other
    // composite is a pure container whose state lives entirely in its children, so
    // this is the one place where "recurse into children" is not the whole job.
    // The hand-rolled save_ this replaces did neither half correctly: it recursed
    // without pushing a scope per child, and it never wrote layer_scalar at all --
    // so a Gemma archive silently dropped the per-layer scales, a numerics change
    // rather than a missing extra.

    namespace
    {
        std::filesystem::path makeTempArchivePath( const std::string& tag )
        {
            const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();

            return std::filesystem::temp_directory_path()
                / std::format( "mila_test_gemma_block_{}_{}.mila", tag, stamp );
        }

        // layer_scalar has no accessor, so it is set the only way the block exposes --
        // through loadParameter, with the [1] FP32 blob the load path expects.
        void setLayerScalar( LocalBlock& block, float value )
        {
            TensorMetadata meta;
            meta.dtype = TensorDataType::FP32;
            meta.shape = shape_t{ 1 };
            meta.total_bytes = sizeof( float );

            TensorBlobView blob( meta, &value, sizeof( float ) );

            block.loadParameter( "layer_scalar", blob );
        }

        // ...and read back only through an archive, which is the stronger oracle
        // anyway: it asserts what a checkpoint actually contains.
        float readLayerScalarFrom( const ModelArchive& archive )
        {
            float value = 0.0f;
            const size_t bytes = archive.readBlobInto( "tensors/layer_scalar/data.bin", &value, sizeof( float ) );

            EXPECT_EQ( bytes, sizeof( float ) );

            return value;
        }
    }

    TEST_F( GemmaBlockCudaTests, Save_WritesLayerScalarAndScopesEveryChild )
    {
        const auto path = makeTempArchivePath( "scalar" );
        std::error_code ec;
        std::filesystem::remove( path, ec );

        auto block = builtBlock<LocalBlock>( RuntimeMode::Inference );
        setLayerScalar( *block, 2.5f );

        {
            ModelArchive archive( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            block->save_( archive, SerializationMode::Checkpoint );
        }

        ModelArchive reader( path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

        // The block's own parameter, which the previous implementation never wrote.
        ASSERT_TRUE( reader.hasFile( "tensors/layer_scalar/data.bin" ) );
        EXPECT_FLOAT_EQ( readLayerScalarFrom( reader ), 2.5f );

        // And the children, each under its own scope. Sixteen children with several
        // parameter-owning norms and projections must not collapse onto one path.
        const auto files = reader.listFiles();
        const auto blob_count = std::count_if( files.begin(), files.end(),
            []( const std::string& name )
            {
                return name.ends_with( "/data.bin" );
            } );

        EXPECT_GT( blob_count, 5 ) << "children collapsed onto a single scope";

        std::filesystem::remove( path, ec );
    }

    TEST_F( GemmaBlockCudaTests, SaveThenLoad_RestoresLayerScalar )
    {
        const auto source_path = makeTempArchivePath( "src" );
        const auto target_path = makeTempArchivePath( "dst" );
        std::error_code ec;
        std::filesystem::remove( source_path, ec );
        std::filesystem::remove( target_path, ec );

        auto source = builtBlock<LocalBlock>( RuntimeMode::Inference );
        setLayerScalar( *source, 2.5f );

        {
            ModelArchive archive( source_path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            source->save_( archive, SerializationMode::Checkpoint );
        }

        // A fresh block: layer_scalar defaults to 1.0f, so a load_ that silently skips
        // the block's own parameter is distinguishable from one that restores it.
        auto target = builtBlock<LocalBlock>( RuntimeMode::Inference );

        {
            ModelArchive archive( source_path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );
            target->load_( archive, SerializationMode::Checkpoint );
        }

        // Re-save the target and inspect: 2.5 means restored, 1.0 means the default
        // survived and load_ did nothing.
        {
            ModelArchive archive( target_path.string(), std::make_unique<ZipSerializer>(), OpenMode::Write );
            target->save_( archive, SerializationMode::Checkpoint );
        }

        ModelArchive reader( target_path.string(), std::make_unique<ZipSerializer>(), OpenMode::Read );

        EXPECT_FLOAT_EQ( readLayerScalarFrom( reader ), 2.5f );

        std::filesystem::remove( source_path, ec );
        std::filesystem::remove( target_path, ec );
    }

    // NOTE: the "archive is missing a named parameter throws" contract is covered
    // generically in Core/CompositeComponent.cpp. Reproducing it here would mean
    // building a children-only archive, which requires calling CompositeComponent's
    // protected save_ from outside the class -- so the Gemma-specific value stays in
    // the two tests above.
}
