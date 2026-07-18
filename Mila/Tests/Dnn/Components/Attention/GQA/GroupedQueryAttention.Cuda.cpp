/**
 * @file GroupedQueryAttention.Cuda.cpp
 * @brief Concrete-component tests for GroupedQueryAttention<DeviceType::Cuda, {FP32, BF16}>.
 *
 * GQA is CUDA-only (no CPU GqaOp). SCOPE NOTE: this file covers the component
 * SURFACE only — construction, accessors, build-shape validation, statelessness,
 * and type. The attention NUMERICS are deliberately not tested here:
 *
 *   - The standalone forward(concatenated_qkv) path is a no-op stub for the
 *     KV-cache backend (Rope-style FIXME in GroupedQueryAttention.ixx ~line 176:
 *     positional_op_->prefill(...) is commented out, so forward returns an
 *     uninitialized view).
 *   - The working compute paths (prefill / decode) take SEPARATE q/k/v tensors,
 *     require setState(GqaState) scratch wiring, and the KV-cache lifecycle
 *     normally driven by the owning transformer.
 *
 * Per the component/operation division of labor (Specifications/Testing.md §1),
 * the GQA attention math belongs to an operation-level CudaGqaOp test that owns
 * the GqaState scratch + cache orchestration. Tracked in BACKLOG.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>

import Mila;
import Compute.ExecutionContext;
// The KV-cache policy structs (NoKvCompression / SlidingWindowKvCache) are not
// re-exported through the Mila umbrella -- consumers import the policy module directly
// (as GroupedQueryAttention.ixx and the models do). MSVC surfaced them transitively via
// import Mila; clang does not (non-export imports are not transitive), so import them here.
import Dnn.Quantization.KvCache.Policy;

namespace Mila::Tests::Dnn::Components::Attention::GQA
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr int64_t kModelDim = 16;
        constexpr int64_t kNumHeads = 4;
        constexpr int64_t kNumKvHeads = 2;
        constexpr int64_t kHeadDim = kModelDim / kNumHeads;  // 4
        // Packed QKV trailing dim = (num_heads + 2 * num_kv_heads) * head_dim = (4 + 4) * 4 = 32.
        constexpr int64_t kPackedQkv = ( kNumHeads + 2 * kNumKvHeads ) * kHeadDim;

        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr const char* name = "Bf16";
        };

        using GqaPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

        class PrecisionNames
        {
        public:
            template<typename TPrecisionTag>
            static std::string GetName( int )
            {
                return TPrecisionTag::name;
            }
        };
    }

    template<typename TPrecisionTag>
    class GqaCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using GqaType = Mila::Dnn::GroupedQueryAttention<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;

        static_assert( GqaType::getDeviceType() == DeviceType::Cuda );
        static_assert( GqaType::getPrecision() == P );

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

        GqaConfig config()
        {
            return GqaConfig( kModelDim, kNumHeads, kNumKvHeads );
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TYPED_TEST_SUITE( GqaCudaTests, GqaPrecisions, PrecisionNames );

    // ====================================================================
    // A. Construction
    // ====================================================================

    TYPED_TEST( GqaCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::GqaType gqa( "gqa", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( gqa.getDeviceId().type, DeviceType::Cuda );
    }

    TYPED_TEST( GqaCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW(
            typename TestFixture::GqaType( "gqa", this->config(), Device::Cpu() ),
            std::invalid_argument );
    }

    TYPED_TEST( GqaCudaTests, Accessors_ReturnConfiguredDims )
    {
        typename TestFixture::GqaType gqa( "gqa", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( gqa.getModelDim(), kModelDim );
        EXPECT_EQ( gqa.getNumHeads(), kNumHeads );
        EXPECT_EQ( gqa.getNumKvHeads(), kNumKvHeads );
    }

    // ====================================================================
    // B. Build Lifecycle (preconditions — validated before op build/cache)
    // ====================================================================

    TYPED_TEST( GqaCudaTests, Forward_ThrowsBeforeBuild )
    {
        typename TestFixture::GqaType gqa( "gqa", this->config(), Device::Cuda( 0 ) );
        typename TestFixture::DeviceTensor input( Device::Cuda( 0 ), shape_t{ 2, 3, kPackedQkv } );

        EXPECT_THROW( gqa.forward( input ), std::runtime_error );
    }

    TYPED_TEST( GqaCudaTests, Build_ThrowsOnWrongTrailingDim )
    {
        typename TestFixture::GqaType gqa( "gqa", this->config(), Device::Cuda( 0 ) );

        // Trailing must be (num_heads + 2*num_kv_heads)*head_dim; model_dim alone is wrong.
        EXPECT_THROW(
            gqa.build( BuildContext( shape_t{ 2, 3, kModelDim }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    TYPED_TEST( GqaCudaTests, Build_ThrowsOnNonRank3 )
    {
        typename TestFixture::GqaType gqa( "gqa", this->config(), Device::Cuda( 0 ) );

        EXPECT_THROW(
            gqa.build( BuildContext( shape_t{ 2, kPackedQkv }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // G. Parameters (stateless) & J. Type
    // ====================================================================

    TYPED_TEST( GqaCudaTests, Parameters_AreEmpty )
    {
        typename TestFixture::GqaType gqa( "gqa", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( gqa.parameterCount(), 0u );
        EXPECT_TRUE( gqa.getParameters().empty() );
        EXPECT_TRUE( gqa.getGradients().empty() );
    }

    TYPED_TEST( GqaCudaTests, GetType_IsGroupedQueryAttention )
    {
        typename TestFixture::GqaType gqa( "gqa", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( gqa.getType(), ComponentType::GroupedQueryAttention );
    }

    // ====================================================================
    // K. KV-cache policy dispatch (SlidingWindowKvCache.md Phase 0, 7.1)
    //
    // Compile-time surface only. The bounded ring's allocation/ring-wrap/softmask
    // (the capacity < T behavior) lands in Phases 1-2 with the op-level parity
    // harness that owns the GqaState + cache lifecycle. Here we pin that the
    // SlidingWindowKvCache policy resolves through OperationTraits to the bounded
    // CudaGqaOp specialization, and that the bounded component compiles and
    // constructs end to end.
    // ====================================================================

    TYPED_TEST( GqaCudaTests, PolicyDispatch_SlidingWindowResolvesBoundedOp )
    {
        using Unbounded = Mila::Dnn::GroupedQueryAttention<
            DeviceType::Cuda, TestFixture::P, Mila::Dnn::Quant::KvCache::NoKvCompression>;
        using Bounded = Mila::Dnn::GroupedQueryAttention<
            DeviceType::Cuda, TestFixture::P, Mila::Dnn::Quant::KvCache::SlidingWindowKvCache>;

        static_assert( !Unbounded::kKvCompressed, "NoKvCompression must be inactive" );
        static_assert( Bounded::kKvCompressed, "SlidingWindowKvCache must be active" );

        static_assert( !Unbounded::OpType::isBounded(), "default policy -> unbounded op" );
        static_assert( Bounded::OpType::isBounded(), "sliding policy -> bounded op" );

        // Uncompressed ring: cache dtype falls back to the compute precision.
        static_assert( Bounded::kCacheDtype == TestFixture::P,
            "SlidingWindowKvCache carries no kStorageDtype" );

        SUCCEED();
    }

    TYPED_TEST( GqaCudaTests, Construct_BoundedPolicyComponentSucceeds )
    {
        Mila::Dnn::GroupedQueryAttention<
            DeviceType::Cuda, TestFixture::P, Mila::Dnn::Quant::KvCache::SlidingWindowKvCache>
            gqa( "gqa_bounded", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( gqa.getType(), ComponentType::GroupedQueryAttention );
        EXPECT_EQ( gqa.getNumKvHeads(), kNumKvHeads );
    }
}
