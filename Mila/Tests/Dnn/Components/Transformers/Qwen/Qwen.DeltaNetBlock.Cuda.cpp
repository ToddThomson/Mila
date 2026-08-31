/**
 * @file Qwen.DeltaNetBlock.Cuda.cpp
 * @brief Structural and state tests for QwenDeltaNetBlock<DeviceType::Cuda, ...>.
 *
 * Three things are under test, and they differ in kind.
 *
 * GEOMETRY and GRAPH: that the block splits its projections the way the precision plan's
 * parameter counts require ([q|k] apart from v), and that every named child is present.
 *
 * PER-ROLE PRECISION: that the three DeltaNet roles resolve to three different Linear
 * instantiations, and that a plan omitting one is rejected. Entirely compile-time, asserted
 * as such.
 *
 * STATE: that a sequence fed in chunks equals the same sequence in one pass, through the
 * WHOLE block. This is the integration property the two pieces underneath were built for --
 * the convolution's rolling window and the mixer's recurrent state must both carry, and a
 * bug in either shows up here as a chunk-boundary discontinuity.
 *
 * CUDA device tests -- skipped when no CUDA device is present.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

import Mila;
import Compute.GqaState;

namespace Mila::Tests::Dnn::Components::Transformers::Qwen
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;

    namespace
    {
        // Small but structurally faithful: 3 value heads per key head, as the 27B has.
        constexpr dim_t kModelDim = 64;
        constexpr dim_t kLayers = 4;
        constexpr dim_t kHidden = 128;
        constexpr dim_t kVocab = 128;
        constexpr dim_t kMaxSeq = 32;

        constexpr dim_t kLinearKeyHeads = 2;
        constexpr dim_t kLinearValueHeads = 6;
        constexpr dim_t kLinearHeadDim = 8;
        constexpr dim_t kConvKernel = 4;

        constexpr dim_t kQueryKeyWidth = 2 * kLinearKeyHeads * kLinearHeadDim;   // 32
        constexpr dim_t kValueWidth = kLinearValueHeads * kLinearHeadDim;        // 48

        QwenConfig smallConfig()
        {
            return QwenConfig( kModelDim, kLayers )
                .withVocabularyLength( kVocab )
                .withNumHeads( 4 )
                .withNumKVHeads( 2 )
                .withHeadDim( 32 )
                .withAttentionOutputGate( true )
                .withHiddenDimension( kHidden )
                .withMaxSequenceLength( kMaxSeq )
                .withRMSNormEpsilon( 1e-6f )
                .withRoPETheta( 1e7f )
                .withPartialRotaryFactor( 0.25f )
                .withFullAttentionInterval( 4 )
                .withLinearNumKeyHeads( kLinearKeyHeads )
                .withLinearNumValueHeads( kLinearValueHeads )
                .withLinearHeadDim( kLinearHeadDim )
                .withLinearConvKernelDim( kConvKernel );
        }

        using ReferenceBlock = QwenDeltaNetBlock<DeviceType::Cuda, TensorDataType::FP32>;
        using Workspace = QwenDeltaNetBlockWorkspace<DeviceType::Cuda, TensorDataType::FP32>;

        // The declaration the per-role mechanism exists to make readable.
        using PlannedBlock = QwenDeltaNetBlock<DeviceType::Cuda, TensorDataType::BF16, QwenPrecisionPlan>;

        using UniformBlock =
            QwenDeltaNetBlock<DeviceType::Cuda, TensorDataType::BF16, QwenUniformPrecisionPlan<PerGroupFp4<128>>>;

        struct PlanMissingDeltaNetGating
        {
            using QkvProjection = PerGroupFp4<128>;
            using OutputProjection = PerGroupFp4<128>;
            using FeedForwardGateUp = PerGroupCodebook2<32>;
            using FeedForwardDown = PerGroupCodebook3<64>;
            using DeltaNetQueryKey = PerGroupCodebook3<64>;
            using DeltaNetValueGateOutput = PerGroupCodebook2<32>;
        };
    }

    // ====================================================================
    // Per-role precision dispatch -- entirely compile-time
    // ====================================================================

    // The three DeltaNet roles resolve to three DIFFERENT Linear instantiations. Without
    // this the split would be documentation: it would compile, and every projection would
    // still carry one format.
    static_assert( !std::is_same_v<PlannedBlock::QueryKeyProjectionType,
                                   PlannedBlock::ValueProjectionType> );
    static_assert( !std::is_same_v<PlannedBlock::ValueProjectionType,
                                   PlannedBlock::GatingProjectionType> );

    // The gating projection is NEVER quantized -- a and b drive the forget gate.
    static_assert( std::is_same_v<PlannedBlock::GatingProjectionType,
                                  Linear<DeviceType::Cuda, TensorDataType::BF16, NoWeightQuant>> );

    // v, z and out_proj share one role and therefore one instantiation.
    static_assert( std::is_same_v<PlannedBlock::ValueProjectionType,
                                  PlannedBlock::GateProjectionType> );
    static_assert( std::is_same_v<PlannedBlock::ValueProjectionType,
                                  PlannedBlock::OutputProjectionType> );

    // A plan missing a role this block builds is rejected at the BLOCK, naming it, rather
    // than falling back to some default precision.
    static_assert( !DeltaNetPrecisionRoles<PlanMissingDeltaNetGating> );

    // A uniform lift stays a valid spelling.
    static_assert( std::is_same_v<UniformBlock::QueryKeyProjectionType,
                                  UniformBlock::ValueProjectionType> );

    class QwenDeltaNetBlockCudaTests : public ::testing::Test
    {
    protected:
        using DeviceTensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        void SetUp() override
        {
            try
            {
                cuda_context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                cuda_context_ = nullptr;
            }

            if ( !cuda_context_ )
            {
                GTEST_SKIP() << "CUDA device not available";
            }
        }

        /**
         * @brief A built block with DETERMINISTIC NON-ZERO weights in every parameter.
         *
         * Inference-mode BuildContext leaves shouldInitializeParameters() false, so a block
         * built and used as-is carries whatever the allocator handed back -- in practice
         * zeros, which makes the mixer output zero and every state-carry test vacuous. The
         * fill below is what gives those tests something to distinguish, and
         * BlockTransformsItsInput is the control that says so.
         *
         * Values are a bounded repeating pattern rather than a ramp: the recurrence
         * multiplies a state by exp(g) every step, and a ramp over a large parameter count
         * puts the tail weights far enough from zero to saturate it.
         */
        std::unique_ptr<ReferenceBlock> builtBlock( dim_t chunk, const Workspace* workspace = nullptr )
        {
            auto block = std::make_unique<ReferenceBlock>( "delta_block", smallConfig(), Device::Cuda( 0 ) );

            if ( workspace )
                block->installSharedWorkspace( *workspace );

            block->build( BuildContext( shape_t{ batch_, chunk, kModelDim },
                RuntimeMode::Inference ).withPrefillSize( chunk ) );

            int index = 0;

            for ( auto* parameter : block->getParameters() )
            {
                auto* tensor = static_cast<DeviceTensor*>( parameter );
                HostFp32 host( Device::Cpu(), tensor->shape() );

                for ( dim_t i = 0; i < host.size(); ++i )
                {
                    host.data()[ i ] =
                        0.05f * static_cast<float>( ((i + index * 7) % 13) - 6 );
                }

                copy( host, *tensor, cuda_context_.get() );
                ++index;
            }

            cuda_context_->synchronize();

            return block;
        }

        HostFp32 rampHost( const shape_t& shape, float start, float step )
        {
            HostFp32 host( Device::Cpu(), shape );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = start + step * static_cast<float>( i );
            }

            return host;
        }

        DeviceTensor toDevice( const HostFp32& host )
        {
            DeviceTensor device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        HostFp32 toFloat( const DeviceTensor& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, cuda_context_.get() );
            cuda_context_->synchronize();

            return host;
        }

        static constexpr dim_t batch_ = 1;
        static constexpr dim_t seq_ = 8;

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    // ====================================================================
    // A. Construction and graph
    // ====================================================================

    TEST_F( QwenDeltaNetBlockCudaTests, Construct_StandaloneSucceeds )
    {
        ReferenceBlock block( "delta_block", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( block.getType(), ComponentType::Transformer );
    }

    TEST_F( QwenDeltaNetBlockCudaTests, Build_Succeeds )
    {
        auto block = builtBlock( seq_ );

        EXPECT_TRUE( block->isBuilt() );
        EXPECT_GT( block->parameterCount(), 0 );
    }

    TEST_F( QwenDeltaNetBlockCudaTests, TheMixerPiecesArePartOfTheGraph )
    {
        auto block = builtBlock( seq_ );

        // Named, not merely present: the block reaches each by name at build, so a rename
        // is a build failure rather than a silently missing piece of the mixer.
        EXPECT_NO_THROW( (void)block->getComponent( "delta_block.delta_rule" ) );
        EXPECT_NO_THROW( (void)block->getComponent( "delta_block.conv_qk" ) );
        EXPECT_NO_THROW( (void)block->getComponent( "delta_block.conv_v" ) );
        EXPECT_NO_THROW( (void)block->getComponent( "delta_block.norm_gate" ) );
        EXPECT_NO_THROW( (void)block->getComponent( "delta_block.fc_in_proj_qk" ) );
        EXPECT_NO_THROW( (void)block->getComponent( "delta_block.fc_in_proj_v" ) );
    }

    TEST_F( QwenDeltaNetBlockCudaTests, ProjectionsSplitQueryKeyFromValue )
    {
        ReferenceBlock block( "delta_block", smallConfig(), Device::Cuda( 0 ) );

        // The plan's parameter counts only add up when q/k are one projection and v is
        // another; a fused in_proj_qkv could not carry two storage policies.
        EXPECT_EQ( block.queryKeyWidth(), kQueryKeyWidth );
        EXPECT_EQ( block.valueWidth(), kValueWidth );
        EXPECT_EQ( block.gatingWidth(), kLinearValueHeads );
    }

    TEST_F( QwenDeltaNetBlockCudaTests, Build_ThrowsOnNonRank3Input )
    {
        ReferenceBlock block( "delta_block", smallConfig(), Device::Cuda( 0 ) );

        EXPECT_THROW( block.build( BuildContext( shape_t{ batch_, kModelDim },
            RuntimeMode::Inference ).withPrefillSize( seq_ ) ), std::invalid_argument );
    }

    // ====================================================================
    // B. The cache contract -- this block keeps a recurrence, not a cache
    // ====================================================================

    TEST_F( QwenDeltaNetBlockCudaTests, DoesNotSupportAKvCache )
    {
        auto block = builtBlock( seq_ );

        EXPECT_FALSE( block->supportsKvCache() );
    }

    TEST_F( QwenDeltaNetBlockCudaTests, RefusesToRewind )
    {
        auto block = builtBlock( seq_ );

        // A recurrent state is a lossy summary of every position it has seen: the
        // information needed to undo the last N steps is not in it. Accepting a rewind
        // would silently corrupt a prefix-reuse session rather than fail it.
        EXPECT_FALSE( block->rewindKvCache( 0 ) );
        EXPECT_FALSE( block->rewindKvCache( 4 ) );
    }

    TEST_F( QwenDeltaNetBlockCudaTests, SetStateIsAcceptedAndIgnored )
    {
        auto block = builtBlock( seq_ );

        // The interface names a concrete GqaState; this block has no attention transient.
        EXPECT_NO_THROW( block->setState( GqaState{} ) );
    }

    // ====================================================================
    // C. The integration property -- both states carry across a chunk boundary
    // ====================================================================

    /**
     * @brief Control for every equality test below: the block must actually DO something.
     *
     * Both residual paths pass the input straight through when the mixer and the
     * feed-forward contribute nothing, so a block with zero weights returns its input
     * unchanged -- and "chunked equals whole" then holds trivially, for two blocks that
     * compute nothing. This asserts the mixer moves the stream before anything else
     * compares two ways of running it.
     */
    TEST_F( QwenDeltaNetBlockCudaTests, BlockTransformsItsInput )
    {
        const shape_t shape{ batch_, seq_, kModelDim };

        auto block = builtBlock( seq_ );
        auto host_x = rampHost( shape, -0.4f, 0.013f );
        auto device_x = toDevice( host_x );

        auto& out_device = block->prefill( device_x, 0 );
        block->synchronize();
        auto out = toFloat( out_device );

        float max_change = 0.0f;

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            EXPECT_FALSE( std::isnan( out.data()[ i ] ) ) << "NaN at index " << i;
            max_change = std::max( max_change, std::fabs( out.data()[ i ] - host_x.data()[ i ] ) );
        }

        // Absolute, not a tolerance multiple: how far the block moves the stream is a
        // property of these weights, measured well above the 1e-4 the equality tests use.
        EXPECT_GT( max_change, 0.01f )
            << "the block returned its input essentially unchanged -- the mixer is "
               "contributing nothing and every equality test below is vacuous";
    }

    TEST_F( QwenDeltaNetBlockCudaTests, ChunkedPrefillEqualsWholeSequence )
    {
        constexpr dim_t kChunk = 4;
        const shape_t whole_shape{ batch_, seq_, kModelDim };

        auto host_x = rampHost( whole_shape, -0.4f, 0.013f );

        auto whole_block = builtBlock( seq_ );
        auto device_whole = toDevice( host_x );
        auto& whole_device = whole_block->prefill( device_whole, 0 );
        whole_block->synchronize();
        auto whole = toFloat( whole_device );

        // A second block, same weights (both zero-initialized deterministically), fed the
        // same sequence in two chunks. The only link between chunks is the carried state:
        // the convolution's rolling window AND the mixer's recurrence.
        auto chunked_block = builtBlock( kChunk );
        std::vector<float> chunked( static_cast<size_t>( batch_ * seq_ * kModelDim ), 0.0f );

        for ( dim_t start = 0; start < seq_; start += kChunk )
        {
            HostFp32 host_chunk( Device::Cpu(), shape_t{ batch_, kChunk, kModelDim } );

            for ( dim_t t = 0; t < kChunk; ++t )
            {
                for ( dim_t c = 0; c < kModelDim; ++c )
                {
                    host_chunk.data()[ t * kModelDim + c ] =
                        host_x.data()[ (start + t) * kModelDim + c ];
                }
            }

            auto device_chunk = toDevice( host_chunk );
            auto& chunk_device = chunked_block->prefill( device_chunk, start );
            chunked_block->synchronize();
            auto chunk_out = toFloat( chunk_device );

            for ( dim_t t = 0; t < kChunk; ++t )
            {
                for ( dim_t c = 0; c < kModelDim; ++c )
                {
                    chunked[ static_cast<size_t>( (start + t) * kModelDim + c ) ] =
                        chunk_out.data()[ t * kModelDim + c ];
                }
            }
        }

        for ( size_t i = 0; i < chunked.size(); ++i )
        {
            EXPECT_NEAR( chunked[ i ], whole.data()[ i ], 1e-4f ) << "at index " << i;
        }
    }

    TEST_F( QwenDeltaNetBlockCudaTests, DecodeProducesTheBlockShape )
    {
        auto block = builtBlock( 1 );

        auto host_step = rampHost( shape_t{ batch_, 1, kModelDim }, 0.1f, 0.01f );
        auto device_step = toDevice( host_step );

        auto& first = block->prefill( device_step, 0 );
        block->synchronize();

        EXPECT_EQ( first.shape()[ 0 ], batch_ );
        EXPECT_EQ( first.shape()[ 1 ], 1 );
        EXPECT_EQ( first.shape()[ 2 ], kModelDim );

        auto& second = block->decode( device_step, 1 );
        block->synchronize();

        EXPECT_EQ( second.shape()[ 2 ], kModelDim );
    }

    TEST_F( QwenDeltaNetBlockCudaTests, ResetKvCacheClearsBothCarriedStates )
    {
        const shape_t shape{ batch_, seq_, kModelDim };

        auto block = builtBlock( seq_ );
        auto host_x = rampHost( shape, -0.4f, 0.013f );
        auto device_x = toDevice( host_x );

        auto& first_device = block->prefill( device_x, 0 );
        block->synchronize();
        auto first = toFloat( first_device );

        block->resetKvCache();

        auto& second_device = block->prefill( device_x, 0 );
        block->synchronize();
        auto second = toFloat( second_device );

        for ( dim_t i = 0; i < first.size(); ++i )
        {
            EXPECT_NEAR( second.data()[ i ], first.data()[ i ], 1e-5f ) << "at index " << i;
        }
    }

    // ====================================================================
    // C2. Snapshot and restore -- Phase 3's remaining exit criterion
    //
    // `rewindKvCache` returns false here and always will: a recurrent state is a lossy
    // summary, so the information needed to undo N steps is not in it. Copying the state out
    // and putting it back is therefore the ONLY mechanism by which anything resembling
    // prefix reuse can work on this layer kind (PromptCaching.md section 2 is what an
    // attention layer does instead, for no memory at all).
    //
    // "Exactly" is meant literally below: same device, same weights, same inputs, so the
    // restored continuation must be BITWISE identical, not merely close. A tolerance here
    // would hide precisely the partial restore these tests exist to catch.
    // ====================================================================

    TEST_F( QwenDeltaNetBlockCudaTests, SnapshotRestoreRoundtripsExactly )
    {
        const shape_t shape{ batch_, seq_, kModelDim };

        auto block = builtBlock( seq_ );
        auto first_chunk = toDevice( rampHost( shape, -0.4f, 0.013f ) );
        auto second_chunk = toDevice( rampHost( shape, 0.2f, -0.009f ) );

        (void)block->prefill( first_chunk, 0 );
        block->synchronize();

        auto taken = block->makeStateSnapshot();
        block->snapshotState( taken );

        // Move the state on, then put it back and take a second reading of it.
        (void)block->prefill( second_chunk, seq_ );
        block->synchronize();

        block->restoreState( taken );

        auto retaken = block->makeStateSnapshot();
        block->snapshotState( retaken );

        auto expectIdentical = [] ( const auto& left, const auto& right, const char* piece )
            {
                ASSERT_EQ( left.size(), right.size() ) << piece;

                for ( dim_t i = 0; i < left.size(); ++i )
                {
                    ASSERT_EQ( left.data()[ i ], right.data()[ i ] )
                        << piece << " diverged at index " << i;
                }
            };

        expectIdentical( taken.query_key_window, retaken.query_key_window, "query/key window" );
        expectIdentical( taken.value_window, retaken.value_window, "value window" );
        expectIdentical( taken.recurrent, retaken.recurrent, "recurrent state" );
    }

    /**
     * @brief The property that matters: a restored block CONTINUES identically.
     *
     * Comparing snapshots to each other only proves the copy is a copy. What a caller needs
     * is that the block's future is the same -- so both arms run the same second chunk, and
     * the arm that was dirtied and restored must produce the same output as the arm that was
     * never disturbed. DirtiedWithoutRestore below is the control that says this can fail.
     */
    TEST_F( QwenDeltaNetBlockCudaTests, RestoredStateContinuesIdentically )
    {
        const shape_t shape{ batch_, seq_, kModelDim };

        auto block = builtBlock( seq_ );
        auto first_chunk = toDevice( rampHost( shape, -0.4f, 0.013f ) );
        auto second_chunk = toDevice( rampHost( shape, 0.2f, -0.009f ) );
        auto dirtying_chunk = toDevice( rampHost( shape, 0.7f, 0.021f ) );

        (void)block->prefill( first_chunk, 0 );
        block->synchronize();

        auto taken = block->makeStateSnapshot();
        block->snapshotState( taken );

        auto& undisturbed_device = block->prefill( second_chunk, seq_ );
        block->synchronize();
        auto undisturbed = toFloat( undisturbed_device );

        // Carry the state somewhere else entirely, then wind it back.
        (void)block->prefill( dirtying_chunk, 2 * seq_ );
        block->synchronize();

        block->restoreState( taken );

        auto& restored_device = block->prefill( second_chunk, seq_ );
        block->synchronize();
        auto restored = toFloat( restored_device );

        for ( dim_t i = 0; i < undisturbed.size(); ++i )
        {
            ASSERT_EQ( restored.data()[ i ], undisturbed.data()[ i ] )
                << "restored continuation diverged at index " << i;
        }
    }

    /// Positive control: without the restore, the same sequence must NOT match.
    TEST_F( QwenDeltaNetBlockCudaTests, DirtiedWithoutRestoreDivergesFromTheUndisturbedRun )
    {
        const shape_t shape{ batch_, seq_, kModelDim };

        auto block = builtBlock( seq_ );
        auto first_chunk = toDevice( rampHost( shape, -0.4f, 0.013f ) );
        auto second_chunk = toDevice( rampHost( shape, 0.2f, -0.009f ) );
        auto dirtying_chunk = toDevice( rampHost( shape, 0.7f, 0.021f ) );

        (void)block->prefill( first_chunk, 0 );
        block->synchronize();

        auto& undisturbed_device = block->prefill( second_chunk, seq_ );
        block->synchronize();
        auto undisturbed = toFloat( undisturbed_device );

        (void)block->prefill( dirtying_chunk, 2 * seq_ );
        block->synchronize();

        auto& dirtied_device = block->prefill( second_chunk, seq_ );
        block->synchronize();
        auto dirtied = toFloat( dirtied_device );

        float max_divergence = 0.0f;

        for ( dim_t i = 0; i < undisturbed.size(); ++i )
        {
            max_divergence = std::max( max_divergence,
                std::fabs( dirtied.data()[ i ] - undisturbed.data()[ i ] ) );
        }

        // ABSOLUTE, not a multiple of any tolerance: how far a dirtied state carries these
        // particular inputs is a property of the inputs, and the bar only has to sit above
        // bitwise noise -- of which there is none, both arms being the same kernel.
        EXPECT_GT( max_divergence, 1e-4f )
            << "dirtying the carried state changed nothing, so RestoredStateContinuesIdentically "
               "cannot be distinguishing a restored state from a dirty one";
    }

    /**
     * @brief Restoring the mixer but not the convolution windows must NOT pass.
     *
     * The block carries three things and the two convolution windows are the easy ones to
     * forget -- they are 60 KiB against the mixer's 3 MiB at the published geometry. A
     * half-restore leaves each convolution's retained rows describing tokens the mixer no
     * longer believes it has seen, and the block then produces plausible output rather than
     * an error. This is the test that makes the aggregate in StateSnapshot load-bearing.
     */
    TEST_F( QwenDeltaNetBlockCudaTests, RestoringTheMixerWithoutTheWindowsIsNotEnough )
    {
        const shape_t shape{ batch_, seq_, kModelDim };

        auto block = builtBlock( seq_ );
        auto first_chunk = toDevice( rampHost( shape, -0.4f, 0.013f ) );
        auto second_chunk = toDevice( rampHost( shape, 0.2f, -0.009f ) );
        auto dirtying_chunk = toDevice( rampHost( shape, 0.7f, 0.021f ) );

        (void)block->prefill( first_chunk, 0 );
        block->synchronize();

        auto clean = block->makeStateSnapshot();
        block->snapshotState( clean );

        auto& undisturbed_device = block->prefill( second_chunk, seq_ );
        block->synchronize();
        auto undisturbed = toFloat( undisturbed_device );

        (void)block->prefill( dirtying_chunk, 2 * seq_ );
        block->synchronize();

        // Build a deliberately partial snapshot: the dirty windows, the clean mixer.
        auto partial = block->makeStateSnapshot();
        block->snapshotState( partial );
        copy( clean.recurrent, partial.recurrent );

        block->restoreState( partial );

        auto& half_device = block->prefill( second_chunk, seq_ );
        block->synchronize();
        auto half = toFloat( half_device );

        float max_divergence = 0.0f;

        for ( dim_t i = 0; i < undisturbed.size(); ++i )
        {
            max_divergence = std::max( max_divergence,
                std::fabs( half.data()[ i ] - undisturbed.data()[ i ] ) );
        }

        EXPECT_GT( max_divergence, 1e-4f )
            << "the convolution windows made no difference to the continuation -- either "
               "they are not being carried at all, or this test has stopped exercising them";
    }

    TEST_F( QwenDeltaNetBlockCudaTests, SnapshotBeforeAnyChunkIsRefused )
    {
        auto block = builtBlock( seq_ );
        auto taken = block->makeStateSnapshot();

        // The convolution windows stand for a left context that does not exist yet. Zeros
        // would answer the question silently and wrongly.
        EXPECT_THROW( block->snapshotState( taken ), std::logic_error );
    }

    // ====================================================================
    // D. Activation pooling
    //
    // Every one of this block's twenty component outputs used to be self-allocated: 138.2
    // MiB per layer on the 27B at chunk 512, ~6.5 GiB over 48 layers, which is what capped
    // that model at 512 context and made WDDM page its weights. The three tests below cover
    // the three ways pooling goes wrong -- a slot assigned to two live values, a slot the
    // installer counts and the block counts again, and a prediction that promises an
    // installation the build never performs.
    // ====================================================================

    /**
     * @brief Pooling must not change a single output value.
     *
     * The failure it exists for is aliasing: `normed` feeds five projections, `z` has to
     * survive from its projection to the output gate, and `res1` is read again at res_2.
     * Give any of those a slot something else writes in between and the block still runs,
     * still produces finite numbers, and is wrong.
     */
    TEST_F( QwenDeltaNetBlockCudaTests, PooledWorkspaceProducesTheSameOutput )
    {
        const shape_t shape{ batch_, seq_, kModelDim };

        auto host_x = rampHost( shape, -0.4f, 0.013f );

        auto self_allocated = builtBlock( seq_ );
        auto device_a = toDevice( host_x );
        auto& out_a_device = self_allocated->prefill( device_a, 0 );
        self_allocated->synchronize();
        auto out_a = toFloat( out_a_device );

        auto workspace = makeQwenDeltaNetBlockWorkspace<DeviceType::Cuda, TensorDataType::FP32>(
            smallConfig(), Device::Cuda( 0 ), batch_, seq_, "ws." );

        auto pooled = builtBlock( seq_, &workspace );
        auto device_b = toDevice( host_x );
        auto& out_b_device = pooled->prefill( device_b, 0 );
        pooled->synchronize();
        auto out_b = toFloat( out_b_device );

        // Same kernels over the same values in the same order, so this is exact up to the
        // reduction order the buffers cannot change -- not a numerical-agreement tolerance.
        for ( dim_t i = 0; i < out_a.size(); ++i )
        {
            EXPECT_NEAR( out_b.data()[ i ], out_a.data()[ i ], 1e-6f ) << "at index " << i;
        }
    }

    /**
     * @brief An installed slot is counted by the installer, and by nobody else.
     */
    TEST_F( QwenDeltaNetBlockCudaTests, PooledWorkspaceIsNotCountedTwice )
    {
        auto workspace = makeQwenDeltaNetBlockWorkspace<DeviceType::Cuda, TensorDataType::FP32>(
            smallConfig(), Device::Cuda( 0 ), batch_, seq_, "ws." );

        auto self_allocated = builtBlock( seq_ );
        auto pooled = builtBlock( seq_, &workspace );

        const auto self_state = self_allocated->getMemoryStats().device_state_bytes;
        const auto pooled_state = pooled->getMemoryStats().device_state_bytes;

        // What the block stops holding is exactly the workspace: the recurrent state and the
        // convolution windows stay per-layer, so the difference is the slot set and nothing
        // else.
        EXPECT_EQ( self_state - pooled_state, workspace.deviceStorageBytes() );
    }

    /**
     * @brief Gate A at block level, on the pooled path (MemoryFootprint.md 7).
     *
     * The defect this pins: QwenTransformer told every block "the parent installs your
     * output" while installing nothing into the DeltaNet ones, so the prediction was short
     * by the whole slot set. Prediction and build have to agree in BOTH positions of the
     * flag, which is why this asserts twice rather than once.
     */
    TEST_F( QwenDeltaNetBlockCudaTests, GetRequiredMemoryMatchesTheBuiltFootprintEitherWay )
    {
        const BuildContext context =
            BuildContext( shape_t{ batch_, seq_, kModelDim }, RuntimeMode::Inference )
            .withPrefillSize( seq_ );

        {
            ReferenceBlock predictor( "delta_block", smallConfig(), Device::Cuda( 0 ) );
            const MemoryStats predicted = predictor.getRequiredMemory( context );

            auto built = builtBlock( seq_ );

            EXPECT_EQ( predicted.device_state_bytes, built->getMemoryStats().device_state_bytes )
                << "self-allocated";
        }

        {
            ReferenceBlock predictor( "delta_block", smallConfig(), Device::Cuda( 0 ) );
            const MemoryStats predicted =
                predictor.getRequiredMemory( context.withInstalledOutput( true ) );

            auto workspace = makeQwenDeltaNetBlockWorkspace<DeviceType::Cuda, TensorDataType::FP32>(
                smallConfig(), Device::Cuda( 0 ), batch_, seq_, "ws." );
            auto built = builtBlock( seq_, &workspace );

            EXPECT_EQ( predicted.device_state_bytes, built->getMemoryStats().device_state_bytes )
                << "pooled";
        }
    }
}
