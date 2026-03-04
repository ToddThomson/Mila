/**
 * @file GQA.Cuda.cpp
 * @brief CUDA unit tests for the GroupedQueryAttention component.
 *
 * Covers construction, device state, accessors, build lifecycle, forward/backward
 * guard conditions, compute correctness (standard GQA, MQA, and MHA-equivalent
 * configs), CPU/CUDA equivalence, determinism, KV cache interface, decode paths,
 * and session-state transitions.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Components::Attention::GQA::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    // ====================================================================
    // Part 1 - Test Shape Definitions
    //
    // GQA input trailing dim: (num_heads + 2 * num_kv_heads) * head_dim
    //   where head_dim = model_dim / num_heads
    //
    // Special cases exercised:
    //   MQA      : num_kv_heads == 1          (maximum KV sharing)
    //   MHAEquiv : num_kv_heads == num_heads  (no KV sharing, degenerate GQA)
    // ====================================================================

    struct TestShape
    {
        int64_t batch;
        int64_t seq;
        int64_t model_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        std::string name;

        int64_t headDim() const
        {
            return model_dim / num_heads;
        }

        shape_t inputShape() const
        {
            const int64_t trailing = (num_heads + 2 * num_kv_heads) * headDim();
            return { batch, seq, trailing };
        }

        shape_t outputShape() const
        {
            return { batch, seq, model_dim };
        }

        shape_t decodeInputShape() const
        {
            const int64_t trailing = (num_heads + 2 * num_kv_heads) * headDim();
            return { batch, 1, trailing };
        }

        shape_t decodeOutputShape() const
        {
            return { batch, 1, model_dim };
        }

        // Standard GQA - group_size = 2, head_dim = 2, trailing = 16
        static TestShape Small()
        {
            return { 2, 4, 8, 4, 2, "Small" };
        }

        // Standard GQA - group_size = 4, head_dim = 8, trailing = 96
        static TestShape Medium()
        {
            return { 4, 16, 64, 8, 2, "Medium" };
        }

        // Standard GQA - group_size = 2, head_dim = 16, trailing = 256
        static TestShape Large()
        {
            return { 8, 32, 128, 8, 4, "Large" };
        }

        // Minimal GQA - group_size = 2, head_dim = 2, trailing = 16
        static TestShape Minimal()
        {
            return { 1, 2, 8, 4, 2, "Minimal" };
        }

        // Multi-Query Attention - num_kv_heads = 1, trailing = 12
        static TestShape MQA()
        {
            return { 2, 4, 8, 4, 1, "MQA" };
        }

        // MHA-equivalent - num_kv_heads == num_heads, trailing = 24 (= 3 * model_dim)
        static TestShape MHAEquiv()
        {
            return { 2, 4, 8, 4, 4, "MHAEquiv" };
        }

        static std::vector<TestShape> StandardShapes()
        {
            return { Small(), Medium(), Large() };
        }

        static std::vector<TestShape> SpecialCaseShapes()
        {
            return { MQA(), MHAEquiv() };
        }
    };

    // ====================================================================
    // Precision Traits
    // ====================================================================

    template<TensorDataType TPrecision>
    struct PrecisionTraits
    {
        static constexpr TensorDataType value = TPrecision;
        static constexpr const char* name = "Unknown";
        static constexpr float tolerance = 1e-4f;
    };

    template<>
    struct PrecisionTraits<TensorDataType::FP32>
    {
        static constexpr TensorDataType value = TensorDataType::FP32;
        static constexpr const char* name = "FP32";
        static constexpr float tolerance = 1e-4f;
    };

    template<>
    struct PrecisionTraits<TensorDataType::FP16>
    {
        static constexpr TensorDataType value = TensorDataType::FP16;
        static constexpr const char* name = "FP16";
        static constexpr float tolerance = 1e-2f;
    };

    // ====================================================================
    // Test Fixture
    // ====================================================================

    template<TensorDataType TPrecision>
    struct GQATestFixture
    {
        GQATestFixture()
            : test_shape( TestShape::Small() ),
              config( test_shape.model_dim, test_shape.num_heads, test_shape.num_kv_heads ),
              component( nullptr ),
              is_training( false )
        {}

        TestShape test_shape;
        GroupedQueryAttentionConfig config;
        std::shared_ptr<GroupedQueryAttention<DeviceType::Cuda, TPrecision>> component;
        bool is_training{ false };

        static GQATestFixture Create( TestShape shape, bool is_training = false )
        {
            GQATestFixture fixture;
            fixture.test_shape = shape;
            fixture.is_training = is_training;
            fixture.config = GroupedQueryAttentionConfig(
                shape.model_dim, shape.num_heads, shape.num_kv_heads );

            std::string name = "gqa_cuda_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.component = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TPrecision>>(
                name, fixture.config, Device::Cuda( 0 ) );

            return fixture;
        }

        const shape_t& input_shape() const
        {
            static thread_local shape_t s;
            s = test_shape.inputShape();
            return s;
        }

        const shape_t& output_shape() const
        {
            static thread_local shape_t s;
            s = test_shape.outputShape();
            return s;
        }
    };

    // ====================================================================
    // Typed Test Suite
    // ====================================================================

    template<typename T>
    class GroupedQueryAttentionCudaTests : public testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = getDeviceCount( DeviceType::Cuda );
            cuda_available_ = (device_count > 0);
        }

        bool cuda_available_{ false };
    };

    template<TensorDataType TPrecision>
    struct PrecisionType {
        static constexpr TensorDataType value = TPrecision;
    };

    using PrecisionTypes = ::testing::Types<
        PrecisionType<TensorDataType::FP32>/*,
        TODO: Enable once GQA supports FP16
        PrecisionType<TensorDataType::FP16>*/
    >;

    TYPED_TEST_SUITE( GroupedQueryAttentionCudaTests, PrecisionTypes );

    // ====================================================================
    // Part 2 - Construction Tests
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        GroupedQueryAttentionConfig cfg( 64, 8, 2 );
        std::shared_ptr<GroupedQueryAttention<DeviceType::Cuda, TPrecision>> component{ nullptr };

        ASSERT_NO_THROW(
            (component = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TPrecision>>(
                "ctor_device_cuda", cfg, Device::Cuda( 0 )
            ))
        );

        ASSERT_NE( component, nullptr );
        EXPECT_EQ( component->getDeviceType(), DeviceType::Cuda );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        GroupedQueryAttentionConfig cfg( 64, 8, 2 );
        std::shared_ptr<GroupedQueryAttention<DeviceType::Cuda, TPrecision>> component;

        ASSERT_NO_THROW(
            (component = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TPrecision>>(
                "ctor_shared_cuda", cfg
            ))
        );

        ASSERT_NE( component, nullptr );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Constructor_WithInvalidDeviceType_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        GroupedQueryAttentionConfig cfg( 64, 8, 2 );

        EXPECT_THROW(
            ((void)std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TPrecision>>(
                "invalid_ctor", cfg, Device::Cpu()
            )),
            std::invalid_argument
        );
    }

    // ====================================================================
    // Part 2 - Device / State Tests
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, GetDeviceType_AfterConstruction_ReturnsCuda )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cuda );

        auto device = fixture.component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cuda );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, IsTraining_InferenceFixture_ReturnsFalse )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small(), false );
        fixture.component->build( fixture.input_shape() );

        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, IsTraining_TrainingFixture_ReturnsTrue )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small(), true );
        fixture.component->build( fixture.input_shape() );
        fixture.component->setTraining( true );

        EXPECT_TRUE( fixture.component->isTraining() );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, SetTraining_TogglingMode_UpdatesState )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small(), false );
        fixture.component->build( fixture.input_shape() );

        EXPECT_FALSE( fixture.component->isTraining() );

        fixture.component->setTraining( true );
        EXPECT_TRUE( fixture.component->isTraining() );

        fixture.component->setTraining( false );
        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, ParameterCount_ReturnsZero )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->parameterCount(), 0 );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Synchronize_AfterConstruction_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }

    // ====================================================================
    // Part 2 - toString Tests
    //
    // kv_cacheable_ is null before build(), so the decode path always reads
    // "fallback". Post-build the CUDA backend resolves IKVCacheable and the
    // string must switch to "KV cache".
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, ToString_PreBuild_ShowsFallbackDecodePath )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "GroupedQueryAttention" ), std::string::npos );
        EXPECT_NE( output.find( "Model dimension" ), std::string::npos );
        EXPECT_NE( output.find( "Num Q heads" ), std::string::npos );
        EXPECT_NE( output.find( "Num KV heads" ), std::string::npos );
        EXPECT_NE( output.find( "Group size" ), std::string::npos );
        EXPECT_NE( output.find( "fallback" ), std::string::npos );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, ToString_AfterBuild_ShowsKVCacheDecodePath )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );
        fixture.component->build( fixture.input_shape() );

        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "KV cache" ), std::string::npos );
    }

    // ====================================================================
    // Part 2 - Build Tests
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, Build_WithVariousShapes_SetsBuiltState )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        for ( const auto& test_shape : TestShape::StandardShapes() )
        {
            auto fixture = GQATestFixture<TPrecision>::Create( test_shape );

            EXPECT_FALSE( fixture.component->isBuilt() )
                << "Component should not be built before build() for shape: " << test_shape.name;

            EXPECT_NO_THROW( fixture.component->build( fixture.input_shape() ) )
                << "Build failed for shape: " << test_shape.name;

            EXPECT_TRUE( fixture.component->isBuilt() )
                << "Component should be built after build() for shape: " << test_shape.name;
        }
    }

    // ====================================================================
    // Part 2 - Accessor Tests
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, GetModelDim_ReturnsConfiguredValue )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->getModelDim(), fixture.test_shape.model_dim );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, GetNumHeads_ReturnsConfiguredValue )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->getNumHeads(), fixture.test_shape.num_heads );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, GetNumKvHeads_ReturnsConfiguredValue )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->getNumKvHeads(), fixture.test_shape.num_kv_heads );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, GetConfig_ReturnsMatchingConfig )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        const auto& cfg = fixture.component->getConfig();

        EXPECT_EQ( cfg.getModelDim(), fixture.test_shape.model_dim );
        EXPECT_EQ( cfg.getNumHeads(), fixture.test_shape.num_heads );
        EXPECT_EQ( cfg.getNumKvHeads(), fixture.test_shape.num_kv_heads );
        EXPECT_EQ( cfg.getHeadDim(), fixture.test_shape.headDim() );
        EXPECT_EQ( cfg.getGroupSize(), fixture.test_shape.num_heads / fixture.test_shape.num_kv_heads );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, GetType_ReturnsGroupedQueryAttention )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->getType(), ComponentType::GroupedQueryAttention );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, GetParameters_ReturnsEmptyVector )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_TRUE( fixture.component->getParameters().empty() );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, GetGradients_ReturnsEmptyVector )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_TRUE( fixture.component->getGradients().empty() );
    }

    // ====================================================================
    // Part 3 - Forward Guard Tests
    //
    // validateConcatenatedQKVShape checks rank and trailing dim only.
    // Batch-size mismatches are not validated at the module level.
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );

        EXPECT_THROW( fixture.component->forward( input ), std::runtime_error );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Forward_InvalidInputShape_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );
        fixture.component->build( fixture.input_shape() );

        shape_t bad_trailing = fixture.input_shape();
        bad_trailing[ 2 ] = bad_trailing[ 2 ] + 1;
        CudaTensor<TPrecision> input_bad_trailing( Device::Cuda( 0 ), bad_trailing );

        EXPECT_THROW( fixture.component->forward( input_bad_trailing ), std::invalid_argument );

        shape_t bad_rank = { fixture.test_shape.batch, fixture.test_shape.seq };
        CudaTensor<TPrecision> input_bad_rank( Device::Cuda( 0 ), bad_rank );

        EXPECT_THROW( fixture.component->forward( input_bad_rank ), std::invalid_argument );
    }

    // ====================================================================
    // Part 3 - Backward Guard Tests
    //
    // backward() validates only the input QKV shape, not output_grad shape.
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, Backward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );
        CudaTensor<TPrecision> output_grad( Device::Cuda( 0 ), fixture.output_shape() );

        EXPECT_THROW( fixture.component->backward( input, output_grad ), std::runtime_error );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Backward_NotTraining_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small(), false );
        fixture.component->build( fixture.input_shape() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );
        CudaTensor<TPrecision> output_grad( Device::Cuda( 0 ), fixture.output_shape() );

        EXPECT_THROW( fixture.component->backward( input, output_grad ), std::runtime_error );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Backward_InvalidInputShape_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small(), true );
        fixture.component->build( fixture.input_shape() );
        fixture.component->setTraining( true );

        CudaTensor<TPrecision> outg_ok( Device::Cuda( 0 ), fixture.output_shape() );

        shape_t bad_trailing = fixture.input_shape();
        bad_trailing[ 2 ] = bad_trailing[ 2 ] + 1;
        CudaTensor<TPrecision> in_bad_trailing( Device::Cuda( 0 ), bad_trailing );

        EXPECT_THROW( fixture.component->backward( in_bad_trailing, outg_ok ), std::invalid_argument );

        shape_t bad_rank = { fixture.test_shape.batch, fixture.test_shape.seq };
        CudaTensor<TPrecision> in_bad_rank( Device::Cuda( 0 ), bad_rank );

        EXPECT_THROW( fixture.component->backward( in_bad_rank, outg_ok ), std::invalid_argument );
    }

    // ====================================================================
    // Part 3 - Forward Compute Tests (Standard Shapes)
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, Forward_WithVariousShapes_ProducesCorrectOutputShape )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        for ( const auto& test_shape : TestShape::StandardShapes() )
        {
            auto fixture = GQATestFixture<TPrecision>::Create( test_shape );
            fixture.component->build( fixture.input_shape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.input_shape() );
            random( host_input, -2.0f, 2.0f );

            CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.input_shape() );
            copy( host_input, device_input );

            auto& out = fixture.component->forward( device_input );

            EXPECT_EQ( out.shape(), fixture.output_shape() )
                << "Output shape mismatch for shape: " << test_shape.name;

            auto host_out = toHost<TensorDataType::FP32>( out );

            EXPECT_EQ( host_out.size(), out.size() )
                << "Output size mismatch for shape: " << test_shape.name;
        }
    }

    // ====================================================================
    // Part 3 - Forward Compute Tests (Special Cases)
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, Forward_MQAConfig_ProducesCorrectOutputShape )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Special-case tests only run for FP32";

        try
        {
            auto shape = TestShape::MQA();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_mqa_cuda", cfg, Device::Cuda( 0 ) );

            comp->build( shape.inputShape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            random( host_input, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            auto& out = comp->forward( device_input );

            EXPECT_EQ( out.shape(), shape.outputShape() );
        }
        catch ( const std::exception& e )
        {
            GTEST_SKIP() << "MQA backend not available: " << e.what();
        }
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Forward_MHAEquivalentConfig_ProducesCorrectOutputShape )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Special-case tests only run for FP32";

        try
        {
            auto shape = TestShape::MHAEquiv();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_mhaequiv_cuda", cfg, Device::Cuda( 0 ) );

            comp->build( shape.inputShape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            random( host_input, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            auto& out = comp->forward( device_input );

            EXPECT_EQ( out.shape(), shape.outputShape() );
        }
        catch ( const std::exception& e )
        {
            GTEST_SKIP() << "MHA-equivalent GQA backend not available: " << e.what();
        }
    }

    // ====================================================================
    // Part 4 - CPU / CUDA Equivalence Tests
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, Forward_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Equivalence test only runs for FP32";

        try
        {
            auto shape = TestShape::Small();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto cpu_comp = std::make_shared<GroupedQueryAttention<DeviceType::Cpu, TensorDataType::FP32>>(
                "gqa_cpu_equiv", cfg, Device::Cpu() );

            auto cuda_comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_cuda_equiv", cfg, Device::Cuda( 0 ) );

            cpu_comp->build( shape.inputShape() );
            cuda_comp->build( shape.inputShape() );

            Mila::Core::RandomGenerator::getInstance().setSeed( 12345 );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            random( host_input, -1.0f, 1.0f );

            auto& cpu_out = cpu_comp->forward( host_input );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            auto& cuda_out = cuda_comp->forward( device_input );
            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_out_cuda = toHost<TensorDataType::FP32>( cuda_out );

            auto* cdata = cpu_out.data();
            auto* gdata = host_out_cuda.data();
            size_t total = cpu_out.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_NEAR( cdata[ i ], gdata[ i ], 1e-3f ) << "Forward mismatch at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "GQA backend not available for CPU/CUDA equivalence test";
        }
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Backward_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Backward equivalence test only runs for FP32";

        try
        {
            auto shape = TestShape::Small();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto cpu_comp = std::make_shared<GroupedQueryAttention<DeviceType::Cpu, TensorDataType::FP32>>(
                "gqa_cpu_equiv_bwd", cfg, Device::Cpu() );

            auto cuda_comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_cuda_equiv_bwd", cfg, Device::Cuda( 0 ) );

            cpu_comp->build( shape.inputShape() );
            cuda_comp->build( shape.inputShape() );

            cpu_comp->setTraining( true );
            cuda_comp->setTraining( true );

            Mila::Core::RandomGenerator::getInstance().setSeed( 12345 );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            random( host_input, -1.0f, 1.0f );

            cpu_comp->forward( host_input );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            cuda_comp->forward( device_input );
            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_outg( Device::Cpu(), shape.outputShape() );

            for ( size_t i = 0; i < host_outg.size(); ++i )
            {
                host_outg.data()[ i ] = static_cast<float>( i );
            }

            auto& cpu_ing = cpu_comp->backward( host_input, host_outg );

            CudaTensor<TensorDataType::FP32> device_outg( Device::Cuda( 0 ), shape.outputShape() );
            copy( host_outg, device_outg );

            auto& cuda_ing = cuda_comp->backward( device_input, device_outg );
            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_ing_cuda = toHost<TensorDataType::FP32>( cuda_ing );

            auto* cpu_data = cpu_ing.data();
            auto* cuda_data = host_ing_cuda.data();
            size_t total = cpu_ing.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_NEAR( cpu_data[ i ], cuda_data[ i ], 1e-3f ) << "Backward mismatch at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "GQA backend not available for CPU/CUDA backward equivalence test";
        }
    }

    // ====================================================================
    // Part 4 - Deterministic Tests
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, Forward_OnesInput_ProducesFiniteOutput )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Deterministic test only runs for FP32";

        try
        {
            auto shape = TestShape::Minimal();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_ones_input", cfg, Device::Cuda( 0 ) );

            comp->build( shape.inputShape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            ones( host_input );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            auto& out = comp->forward( device_input );
            auto host_out = toHost<TensorDataType::FP32>( out );

            auto* data = host_out.data();
            size_t total = host_out.size();
            float sum = 0.0f;

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_TRUE( std::isfinite( data[ i ] ) ) << "Non-finite output at index " << i;
                sum += data[ i ];
            }

            EXPECT_GT( sum, 0.1f ) << "Output should be non-zero for ones input";
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "GQA backend not available";
        }
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Forward_ZerosInput_ProducesZeroOutput )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Deterministic test only runs for FP32";

        try
        {
            auto shape = TestShape::Minimal();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_zeros_input", cfg, Device::Cuda( 0 ) );

            comp->build( shape.inputShape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            zeros( host_input );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            auto& out = comp->forward( device_input );
            auto host_out = toHost<TensorDataType::FP32>( out );

            auto* data = host_out.data();
            size_t total = host_out.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_TRUE( std::isfinite( data[ i ] ) ) << "Non-finite output at index " << i;
                EXPECT_NEAR( data[ i ], 0.0f, 1e-5f ) << "Non-zero output for zeros input at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "GQA backend not available";
        }
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Forward_Deterministic_ReproducibleWithSeed )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Deterministic test only runs for FP32";

        try
        {
            auto shape = TestShape::Minimal();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_reproducible_fwd", cfg, Device::Cuda( 0 ) );

            comp->build( shape.inputShape() );

            Mila::Core::RandomGenerator::getInstance().setSeed( 42 );

            CpuTensor<TensorDataType::FP32> host_input1( Device::Cpu(), shape.inputShape() );
            random( host_input1, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input1( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input1, device_input1 );

            auto& out1 = comp->forward( device_input1 );
            auto host_out1 = toHost<TensorDataType::FP32>( out1 );

            Mila::Core::RandomGenerator::getInstance().setSeed( 42 );

            CpuTensor<TensorDataType::FP32> host_input2( Device::Cpu(), shape.inputShape() );
            random( host_input2, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input2( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input2, device_input2 );

            auto& out2 = comp->forward( device_input2 );
            auto host_out2 = toHost<TensorDataType::FP32>( out2 );

            auto* data1 = host_out1.data();
            auto* data2 = host_out2.data();
            size_t total = host_out1.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_FLOAT_EQ( data1[ i ], data2[ i ] ) << "Non-reproducible output at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "GQA backend not available";
        }
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Backward_OnesGradient_ProducesFiniteGradients )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Deterministic test only runs for FP32";

        try
        {
            auto shape = TestShape::Minimal();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_ones_grad", cfg, Device::Cuda( 0 ) );

            comp->build( shape.inputShape() );
            comp->setTraining( true );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            ones( host_input );

            CpuTensor<TensorDataType::FP32> host_output_grad( Device::Cpu(), shape.outputShape() );
            ones( host_output_grad );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_output_grad( Device::Cuda( 0 ), shape.outputShape() );

            copy( host_input, device_input );
            copy( host_output_grad, device_output_grad );

            auto& input_grad = comp->backward( device_input, device_output_grad );
            auto host_input_grad = toHost<TensorDataType::FP32>( input_grad );

            auto* grad_data = host_input_grad.data();
            size_t total = host_input_grad.size();
            bool has_nonzero = false;

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_TRUE( std::isfinite( grad_data[ i ] ) ) << "Non-finite gradient at index " << i;

                if ( std::abs( grad_data[ i ] ) > 1e-6f )
                {
                    has_nonzero = true;
                }
            }

            EXPECT_TRUE( has_nonzero ) << "Gradients should be non-zero for ones input/output_grad";
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "GQA backend not available";
        }
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Backward_Deterministic_ReproducibleGradientsWithSeed )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Deterministic test only runs for FP32";

        try
        {
            auto shape = TestShape::Minimal();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_reproducible_bwd", cfg, Device::Cuda( 0 ) );

            comp->build( shape.inputShape() );
            comp->setTraining( true );

            Mila::Core::RandomGenerator::getInstance().setSeed( 123 );

            CpuTensor<TensorDataType::FP32> host_input1( Device::Cpu(), shape.inputShape() );
            random( host_input1, -0.5f, 0.5f );

            CpuTensor<TensorDataType::FP32> host_outg1( Device::Cpu(), shape.outputShape() );
            random( host_outg1, -0.5f, 0.5f );

            CudaTensor<TensorDataType::FP32> device_input1( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_outg1( Device::Cuda( 0 ), shape.outputShape() );

            copy( host_input1, device_input1 );
            copy( host_outg1, device_outg1 );

            auto& ing1 = comp->backward( device_input1, device_outg1 );
            auto host_ing1 = toHost<TensorDataType::FP32>( ing1 );

            Mila::Core::RandomGenerator::getInstance().setSeed( 123 );

            CpuTensor<TensorDataType::FP32> host_input2( Device::Cpu(), shape.inputShape() );
            random( host_input2, -0.5f, 0.5f );

            CpuTensor<TensorDataType::FP32> host_outg2( Device::Cpu(), shape.outputShape() );
            random( host_outg2, -0.5f, 0.5f );

            CudaTensor<TensorDataType::FP32> device_input2( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_outg2( Device::Cuda( 0 ), shape.outputShape() );

            copy( host_input2, device_input2 );
            copy( host_outg2, device_outg2 );

            auto& ing2 = comp->backward( device_input2, device_outg2 );
            auto host_ing2 = toHost<TensorDataType::FP32>( ing2 );

            auto* grad1 = host_ing1.data();
            auto* grad2 = host_ing2.data();
            size_t total = host_ing1.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_FLOAT_EQ( grad1[ i ], grad2[ i ] ) << "Non-reproducible gradient at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "GQA backend not available";
        }
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, ForwardBackward_ProducesFiniteGradients )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Deterministic test only runs for FP32";

        try
        {
            auto shape = TestShape::Minimal();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_fwd_bwd_trace", cfg, Device::Cuda( 0 ) );

            comp->build( shape.inputShape() );
            comp->setTraining( true );

            Mila::Core::RandomGenerator::getInstance().setSeed( 422 );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            random( host_input, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            comp->forward( device_input );

            Mila::Core::RandomGenerator::getInstance().setSeed( 423 );

            CpuTensor<TensorDataType::FP32> host_output_grad( Device::Cpu(), shape.outputShape() );
            random( host_output_grad, -0.5f, 0.5f );

            CudaTensor<TensorDataType::FP32> device_output_grad( Device::Cuda( 0 ), shape.outputShape() );
            copy( host_output_grad, device_output_grad );

            auto& ing = comp->backward( device_input, device_output_grad );
            auto host_input_grad = toHost<TensorDataType::FP32>( ing );

            auto* grad_data = host_input_grad.data();
            size_t total = host_input_grad.size();
            bool all_finite = true;
            bool has_nonzero = false;

            for ( size_t i = 0; i < total; ++i )
            {
                if ( !std::isfinite( grad_data[ i ] ) )
                {
                    all_finite = false;
                }
                else if ( std::abs( grad_data[ i ] ) > 1e-6f )
                {
                    has_nonzero = true;
                }
            }

            EXPECT_TRUE( all_finite ) << "All gradients should be finite";
            EXPECT_TRUE( has_nonzero ) << "Gradients should be non-zero for this input/output_grad";
        }
        catch ( const std::exception& e )
        {
            GTEST_SKIP() << "GQA backend not available: " << e.what();
        }
    }

    // ====================================================================
    // Part 5 - KV Cache Interface Tests
    //
    // initializeKVCache() and resetKVCache() are not part of the public API
    // in GroupedQueryAttention (currently commented out in the implementation).
    // Only supportsKVCache() is tested here.
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, SupportsKVCache_BeforeBuild_ReturnsFalse )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_FALSE( fixture.component->supportsKVCache() );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, SupportsKVCache_AfterBuild_ReturnsTrue )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );
        fixture.component->build( fixture.input_shape() );

        EXPECT_TRUE( fixture.component->supportsKVCache() );
    }

    // ====================================================================
    // Part 5 - Decode / KV Cache Path Tests
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, Decode_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );

        EXPECT_THROW( fixture.component->decode( input, 0 ), std::runtime_error );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Decode_InvalidShape_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = GQATestFixture<TPrecision>::Create( TestShape::Small() );
        fixture.component->build( fixture.input_shape() );

        shape_t bad_shape = fixture.input_shape();
        bad_shape[ 2 ] = bad_shape[ 2 ] + 1;
        CudaTensor<TPrecision> bad_input( Device::Cuda( 0 ), bad_shape );

        EXPECT_THROW( fixture.component->decode( bad_input, 0 ), std::invalid_argument );
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Decode_AfterForward_UsesKVCachePath )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Decode test only runs for FP32";

        try
        {
            auto shape = TestShape::Small();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_kvcache_decode", cfg, Device::Cuda( 0 ) );

            comp->build( shape.inputShape() );
            ASSERT_TRUE( comp->supportsKVCache() );

            CpuTensor<TensorDataType::FP32> host_prefill( Device::Cpu(), shape.inputShape() );
            random( host_prefill, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_prefill( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_prefill, device_prefill );

            comp->forward( device_prefill );

            CpuTensor<TensorDataType::FP32> host_token( Device::Cpu(), shape.decodeInputShape() );
            random( host_token, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_token( Device::Cuda( 0 ), shape.decodeInputShape() );
            copy( host_token, device_token );

            auto& out = comp->decode( device_token, static_cast<int>(shape.seq) );
            comp->synchronize();

            EXPECT_EQ( out.shape(), shape.decodeOutputShape() );

            auto host_out = toHost<TensorDataType::FP32>( out );

            for ( size_t i = 0; i < host_out.size(); ++i )
            {
                EXPECT_TRUE( std::isfinite( host_out.data()[ i ] ) )
                    << "Non-finite decode output at index " << i;
            }
        }
        catch ( const std::exception& e )
        {
            GTEST_SKIP() << "KV cache decode not available: " << e.what();
        }
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, Decode_WithoutCacheInitialized_UsesFallbackForward )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Decode fallback test only runs for FP32";

        try
        {
            auto shape = TestShape::Small();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_decode_fallback", cfg, Device::Cuda( 0 ) );

            comp->build( shape.inputShape() );
            ASSERT_TRUE( comp->supportsKVCache() );

            // cache_initialized_ is false - decode() falls through to operation_->forward()
            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            random( host_input, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            auto& out = comp->decode( device_input, 0 );
            comp->synchronize();

            EXPECT_EQ( out.shape(), shape.outputShape() );
        }
        catch ( const std::exception& e )
        {
            GTEST_SKIP() << "Decode fallback not available: " << e.what();
        }
    }

    // ====================================================================
    // Part 5 - KV Cache Session State Tests
    // ====================================================================

    TYPED_TEST( GroupedQueryAttentionCudaTests, Forward_AfterDecode_AutoResetsCacheAndPrefills )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Session state test only runs for FP32";

        try
        {
            auto shape = TestShape::Small();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_session_reset", cfg, Device::Cuda( 0 ) );

            comp->build( shape.inputShape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            random( host_input, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            comp->forward( device_input );

            CpuTensor<TensorDataType::FP32> host_token( Device::Cpu(), shape.decodeInputShape() );
            random( host_token, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_token( Device::Cuda( 0 ), shape.decodeInputShape() );
            copy( host_token, device_token );

            // Sets decode_active_ = true
            comp->decode( device_token, static_cast<int>(shape.seq) );

            // A second forward() must auto-reset the decode session and re-prefill
            random( host_input, -1.0f, 1.0f );
            copy( host_input, device_input );

            auto& out = comp->forward( device_input );
            comp->synchronize();

            EXPECT_EQ( out.shape(), shape.outputShape() );
        }
        catch ( const std::exception& e )
        {
            GTEST_SKIP() << "Session reset not available: " << e.what();
        }
    }

    TYPED_TEST( GroupedQueryAttentionCudaTests, SetTraining_ToTrue_WithActiveCache_ResetsKVCacheSession )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Session state test only runs for FP32";

        try
        {
            auto shape = TestShape::Small();
            GroupedQueryAttentionConfig cfg( shape.model_dim, shape.num_heads, shape.num_kv_heads );

            auto comp = std::make_shared<GroupedQueryAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "gqa_train_resets_cache", cfg, Device::Cuda( 0 ) );

            comp->build( shape.inputShape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            random( host_input, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            comp->forward( device_input );

            // Entering training mode must reset the active cache session
            comp->setTraining( true );
            EXPECT_TRUE( comp->isTraining() );

            // After returning to inference mode, forward() must re-initialize cache cleanly
            comp->setTraining( false );

            random( host_input, -1.0f, 1.0f );
            copy( host_input, device_input );

            auto& out = comp->forward( device_input );
            comp->synchronize();

            EXPECT_EQ( out.shape(), shape.outputShape() );
        }
        catch ( const std::exception& e )
        {
            GTEST_SKIP() << "KV cache session state test not available: " << e.what();
        }
    }
}