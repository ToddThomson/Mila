/**
 * @file Qwen.Block.Cuda.cpp
 * @brief Structural tests for QwenAttentionBlock<DeviceType::Cuda, ...>.
 *
 * Two things are under test and they are different in kind.
 *
 * The first is GEOMETRY: the gated query projection, the decoupled head_dim, and the two
 * packed widths that differ by exactly the gate half. These are runtime checks on a built
 * block, and they need a device.
 *
 * The second is PER-ROLE PRECISION DISPATCH -- that four projections in one block resolve to
 * four different Linear instantiations, and that a plan omitting a role the block builds is
 * rejected. That is entirely a compile-time property and is asserted as one; a runtime test
 * of it could only observe what the compiler had already decided.
 *
 * Numerics and the prefill/decode execution path are covered at the network level in
 * Qwen.Cuda.cpp, the same way LlamaTransformer is validated rather than a standalone block:
 * prefill/decode need the shared GqaState workspace that the transformer owns.
 *
 * CUDA device tests -- skipped when no CUDA device is present.
 */

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

import Mila;

// Compute.ExecutionContext is NOT part of the public umbrella (Mila.ixx exports only
// IExecutionContext and the factory), and instantiating a CUDA block reaches
// CudaGqaOp::build, which needs ExecutionContext<Cuda> COMPLETE -- not merely reachable
// through the block module's own imports. Without this line the instantiation fails with
// "use of undefined type ExecutionContext<Cuda>" pointing into CudaGqaOp.
//
// The equivalent Gemma suite needs no such import only because GemmaModel instantiates
// GemmaBlock inside the library, where the type is complete, and MSVC reuses that
// instantiation. Qwen has no model yet (QwenModel is Phase 4), so this suite is the first
// place in the tree where a CUDA block is instantiated by a consumer. A downstream
// consumer would hit exactly this, so it is a library defect rather than a test quirk --
// filed in BACKLOG against the public export surface, not worked around there.
import Compute.ExecutionContext;

namespace Mila::Tests::Dnn::Components::Transformers::Qwen
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;

    namespace
    {
        // Small but structurally faithful. head_dim (32) is decoupled from
        // embedding/num_heads (64 / 4 = 16), and every split width is a multiple of the
        // structural kernel's FP32 vector width of 4.
        constexpr dim_t kModelDim = 64;
        constexpr dim_t kLayers = 4;
        constexpr dim_t kHeads = 4;
        constexpr dim_t kKVHeads = 2;
        constexpr dim_t kHeadDim = 32;
        constexpr dim_t kHidden = 128;
        constexpr dim_t kVocab = 128;
        constexpr dim_t kMaxSeq = 32;

        constexpr dim_t kQProj = kHeads * kHeadDim;              // 128
        constexpr dim_t kGatedQProj = 2 * kQProj;                // 256
        constexpr dim_t kKVProj = kKVHeads * kHeadDim;           //  64
        constexpr dim_t kPackedQKV = kGatedQProj + 2 * kKVProj;  // 384
        constexpr dim_t kAttentionPackedQKV = kQProj + 2 * kKVProj; // 256

        QwenConfig smallConfig()
        {
            return QwenConfig( kModelDim, kLayers )
                .withVocabularyLength( kVocab )
                .withNumHeads( kHeads )
                .withNumKVHeads( kKVHeads )
                .withHeadDim( kHeadDim )
                .withAttentionOutputGate( true )
                .withHiddenDimension( kHidden )
                .withMaxSequenceLength( kMaxSeq )
                .withRMSNormEpsilon( 1e-6f )
                .withRoPETheta( 1e7f )
                .withPartialRotaryFactor( 0.25f )
                .withFullAttentionInterval( 1 );
        }

        using ReferenceBlock = QwenAttentionBlock<DeviceType::Cuda, TensorDataType::FP32>;

        // The declaration the whole per-role mechanism exists to make readable. It names one
        // plan, and the four projections underneath it carry four different formats.
        using PlannedBlock = QwenAttentionBlock<DeviceType::Cuda, TensorDataType::BF16, QwenPrecisionPlan>;

        // A bare policy is still a valid spelling and means "uniform".
        using UniformBlock = QwenAttentionBlock<DeviceType::Cuda, TensorDataType::BF16, PerGroupFp4<128>>;

        struct PlanMissingFeedForwardDown
        {
            using QkvProjection = PerGroupFp4<128>;
            using OutputProjection = PerGroupFp4<128>;
            using FeedForwardGateUp = PerGroupCodebook2<32>;
        };
    }

    // ====================================================================
    // Per-role precision dispatch -- entirely compile-time
    // ====================================================================

    // The four roles resolve to four DIFFERENT Linear instantiations. Without this the plan
    // would be documentation: it would compile, and every projection would still carry one
    // format.
    static_assert( std::is_same_v<PlannedBlock::QkvProjectionType,
        Linear<DeviceType::Cuda, TensorDataType::BF16, PerGroupFp4<128>>> );
    static_assert( std::is_same_v<PlannedBlock::FeedForwardGateUpType,
        Linear<DeviceType::Cuda, TensorDataType::BF16, PerGroupCodebook2<32>>> );
    static_assert( std::is_same_v<PlannedBlock::FeedForwardDownType,
        Linear<DeviceType::Cuda, TensorDataType::BF16, PerGroupCodebook3<64>>> );
    static_assert( !std::is_same_v<PlannedBlock::FeedForwardGateUpType,
        PlannedBlock::FeedForwardDownType> );
    static_assert( !std::is_same_v<PlannedBlock::QkvProjectionType,
        PlannedBlock::FeedForwardGateUpType> );

    // A bare policy lifts, so every projection in this block is the same type -- which is
    // what "uniform" has to mean for the older spellings to keep working.
    static_assert( std::is_same_v<UniformBlock::QkvProjectionType, UniformBlock::FeedForwardDownType> );
    static_assert( std::is_same_v<UniformBlock::FeedForwardGateUpType,
        Linear<DeviceType::Cuda, TensorDataType::BF16, PerGroupFp4<128>>> );

    // The reference block quantizes nothing.
    static_assert( !ReferenceBlock::PrecisionPlan::FeedForwardGateUp::kIsQuantized );

    // A plan omitting a role the block builds fails the block's own requires-clause. The
    // assertion is on the constraint the clause names, because a class that fails to compile
    // cannot be expressed as a passing test.
    static_assert( !DecoderPrecisionPlan<PrecisionPlanFor<PlanMissingFeedForwardDown>> );
    static_assert( DecoderPrecisionPlan<PrecisionPlanFor<QwenPrecisionPlan>> );

    static_assert( ReferenceBlock::getDeviceType() == DeviceType::Cuda );
    static_assert( ReferenceBlock::getPrecision() == TensorDataType::FP32 );

    class QwenBlockCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }
        }

        std::unique_ptr<ReferenceBlock> builtBlock( RuntimeMode mode )
        {
            auto block = std::make_unique<ReferenceBlock>( "qwen_block", smallConfig(), Device::Cuda( 0 ) );

            // The transformer threads the prefill chunk to the block; a standalone build
            // must supply it (prefill_size defaults to 0, sizing scratch empty).
            block->build(
                BuildContext( shape_t{ batch_, seq_, kModelDim }, mode ).withPrefillSize( seq_ ) );

            return block;
        }

        static constexpr dim_t batch_ = 1;
        static constexpr dim_t seq_ = 4;
    };

    // ====================================================================
    // A. Construction and validation
    // ====================================================================

    TEST_F( QwenBlockCudaTests, Construct_Succeeds )
    {
        ReferenceBlock block( "qwen_block", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( block.getName(), "qwen_block" );
        EXPECT_EQ( block.getDeviceId().type, DeviceType::Cuda );
    }

    TEST_F( QwenBlockCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( ReferenceBlock( "qwen_block", smallConfig(), Device::Cpu() ),
            std::invalid_argument );
    }

    TEST_F( QwenBlockCudaTests, Construct_WithoutTheOutputGateThrows )
    {
        // Not a defect in the config -- a different architecture. Accepting it would produce
        // a Qwen-named block that is plain grouped-query attention.
        QwenConfig ungated = smallConfig().withAttentionOutputGate( false );

        EXPECT_THROW( ReferenceBlock( "qwen_block", ungated, Device::Cuda( 0 ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // B. Build lifecycle
    // ====================================================================

    TEST_F( QwenBlockCudaTests, Build_SetsIsBuiltAndAllocatesParameters )
    {
        auto block = builtBlock( RuntimeMode::Inference );

        EXPECT_TRUE( block->isBuilt() );
        EXPECT_GT( block->parameterCount(), 0 );
    }

    TEST_F( QwenBlockCudaTests, Build_ThrowsOnNonRank3Input )
    {
        ReferenceBlock block( "qwen_block", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_THROW( block.build(
            BuildContext( shape_t{ seq_, kModelDim }, RuntimeMode::Inference ).withPrefillSize( seq_ ) ),
            std::invalid_argument );
    }

    TEST_F( QwenBlockCudaTests, Build_ThrowsOnModelDimMismatch )
    {
        ReferenceBlock block( "qwen_block", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_THROW( block.build(
            BuildContext( shape_t{ batch_, seq_, kModelDim + 1 }, RuntimeMode::Inference )
                .withPrefillSize( seq_ ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // C. Geometry -- the gate half and the decoupled head width
    // ====================================================================

    TEST_F( QwenBlockCudaTests, PackedQkvCarriesTheGateHalf )
    {
        ReferenceBlock block( "qwen_block", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( block.qProjWidth(), kQProj );
        EXPECT_EQ( block.kvProjWidth(), kKVProj );
        EXPECT_EQ( block.packedQKVWidth(), kPackedQKV );

        // What the projection emits exceeds what attention consumes by exactly one query
        // width -- that surplus is the gate.
        EXPECT_EQ( block.packedQKVWidth() - kAttentionPackedQKV, block.qProjWidth() );
    }

    TEST_F( QwenBlockCudaTests, HeadDimIsDecoupledFromResidualStream )
    {
        ReferenceBlock block( "qwen_block", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_NE( block.headDim(), kModelDim / kHeads );
        EXPECT_NE( block.qProjWidth(), kModelDim );
    }

    // ====================================================================
    // D. Component graph
    // ====================================================================

    TEST_F( QwenBlockCudaTests, GetComponents_ReturnsTheExpectedChildren )
    {
        ReferenceBlock block( "qwen_block", smallConfig(), Device::Cuda( 0 ) );

        // 2 norms (input, post_attn) + fc_qkv_proj + q_norm + k_norm + rope + gqa +
        // output_gate + fc_o_proj + res_1 + fc_gate_up + swiglu + fc_down + res_2.
        EXPECT_EQ( block.getComponents().size(), 14u );
    }

    TEST_F( QwenBlockCudaTests, QkNormIsPartOfTheGraph )
    {
        auto block = builtBlock( RuntimeMode::Inference );

        // The checkpoint carries self_attn.q_norm/k_norm [head_dim] on every full-attention
        // layer. Named for the same reason the output gate is: a rename must break the build
        // rather than silently drop the normalization.
        EXPECT_NO_THROW( (void)block->getComponent( "qwen_block.q_norm" ) );
        EXPECT_NO_THROW( (void)block->getComponent( "qwen_block.k_norm" ) );
    }

    TEST_F( QwenBlockCudaTests, OutputGateIsPartOfTheGraph )
    {
        auto block = builtBlock( RuntimeMode::Inference );

        // Named, not merely present: the block reaches it by name at build, so a rename
        // here is a build-time failure rather than a silent omission of the gate.
        EXPECT_NO_THROW( (void)block->getComponent( "qwen_block.output_gate" ) );
    }

    TEST_F( QwenBlockCudaTests, GetType_IsTransformer )
    {
        ReferenceBlock block( "qwen_block", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( block.getType(), ComponentType::Transformer );
    }

    // ====================================================================
    // E. Memory prediction agrees with the allocation
    // ====================================================================

    TEST_F( QwenBlockCudaTests, GetRequiredMemory_CoversWhatBuildAllocates )
    {
        ReferenceBlock block( "qwen_block", smallConfig(), Device::Cuda( 0 ) );

        const BuildContext context =
            BuildContext( shape_t{ batch_, seq_, kModelDim }, RuntimeMode::Inference )
                .withPrefillSize( seq_ );

        const MemoryStats predicted = block.getRequiredMemory( context );

        block.build( context );

        const MemoryStats actual = block.getMemoryStats();

        // Parameters are exact: a weight is a weight, and a prediction that misses one has
        // mis-derived a projection width.
        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes );

        // State is one-directional. A prediction must never undershoot, but it may exceed
        // what this block allocated: the RoPE cos/sin cache is process-wide and deduplicated
        // by RopeCacheRegistry, so a block built after another with the same
        // (theta, max_seq_len, head_dim) allocates none of its own.
        EXPECT_GE( predicted.device_state_bytes, actual.device_state_bytes );
    }
}
