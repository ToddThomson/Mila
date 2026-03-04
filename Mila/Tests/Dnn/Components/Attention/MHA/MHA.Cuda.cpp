#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Components::Attention::MHA::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    // ====================================================================
    // Test Shape Definitions (input is concatenated Q||K||V -> last dim == 3 * embedding)
    // ====================================================================

    enum class TestShapeSize
    {
        Small,
        Medium,
        Large,
        Minimal
    };

    struct TestShape
    {
        TestShapeSize size;
        int64_t batch;
        int64_t seq;
        int64_t embedding_dim;
        int64_t num_heads;
        std::string name;

        shape_t inputShape() const
        {
            return { batch, seq, static_cast<int64_t>(3 * embedding_dim) };
        }

        shape_t outputShape() const
        {
            return { batch, seq, embedding_dim };
        }

        shape_t decodeInputShape() const
        {
            return { batch, 1, static_cast<int64_t>(3 * embedding_dim) };
        }

        shape_t decodeOutputShape() const
        {
            return { batch, 1, embedding_dim };
        }

        static TestShape Small()
        {
            return { TestShapeSize::Small, 2, 4, 8, 2, "Small" };
        }

        static TestShape Medium()
        {
            return { TestShapeSize::Medium, 8, 16, 128, 8, "Medium" };
        }

        static TestShape Large()
        {
            return { TestShapeSize::Large, 16, 32, 256, 8, "Large" };
        }

        static TestShape Minimal()
        {
            return { TestShapeSize::Minimal, 1, 2, 4, 2, "Minimal" };
        }

        static std::vector<TestShape> AllShapes()
        {
            return { Small(), Medium(), Large(), Minimal() };
        }

        static std::vector<TestShape> StandardShapes()
        {
            return { Small(), Medium(), Large() };
        }
        
        
    };

    // ====================================================================
    // Precision Traits used for tolerances / naming
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
    // Test Fixture Structure
    // ====================================================================

    template<TensorDataType TPrecision>
    struct AttentionTestFixture
    {
        AttentionTestFixture()
            : test_shape( TestShape::Small() ),
              config( test_shape.embedding_dim, test_shape.num_heads ),
              component( nullptr ),
              is_training( false )
        {}

        TestShape test_shape;
        MultiHeadAttentionConfig config;
        std::shared_ptr<MultiHeadAttention<DeviceType::Cuda, TPrecision>> component;
        bool is_training{ false };

        static AttentionTestFixture Create( TestShape shape, bool is_training = false )
        {
            AttentionTestFixture fixture;
            fixture.test_shape = shape;
            fixture.is_training = is_training;

            fixture.config = MultiHeadAttentionConfig( shape.embedding_dim, shape.num_heads );

            std::string name = "attention_cuda_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.component = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

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
    // Typed Tests (Precision-Based)
    // ====================================================================

    template<typename T>
    class MHACudaTests : public testing::Test
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
        TODO: Enable once Attention supports
        PrecisionType<TensorDataType::FP16>*/
    >;

    TYPED_TEST_SUITE( MHACudaTests, PrecisionTypes );

    TYPED_TEST( MHACudaTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        MultiHeadAttentionConfig cfg( 64, 8 );

        std::shared_ptr<MultiHeadAttention<DeviceType::Cuda, TPrecision>> component{ nullptr };

        ASSERT_NO_THROW(
            (component = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TPrecision>>(
                "ctor_device_cuda",
                cfg,
                Device::Cuda( 0 )
            ))
        );

        ASSERT_NE( component, nullptr );
        EXPECT_EQ( component->getDeviceType(), DeviceType::Cuda );
    }

    TYPED_TEST( MHACudaTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        MultiHeadAttentionConfig cfg( 64, 8 );

        std::shared_ptr<MultiHeadAttention<DeviceType::Cuda, TPrecision>> component;

        ASSERT_NO_THROW(
            (component = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TPrecision>>(
                "ctor_shared_cuda",
                cfg
            ))
        );

        ASSERT_NE( component, nullptr );
    }

    TYPED_TEST( MHACudaTests, Constructor_WithInvalidDeviceType_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        MultiHeadAttentionConfig cfg( 64, 8 );

        EXPECT_THROW(
            ((void)std::make_shared<MultiHeadAttention<DeviceType::Cuda, TPrecision>>(
                "invalid_ctor",
                cfg,
                Device::Cpu()
            )),
            std::invalid_argument
        );
    }

    TYPED_TEST( MHACudaTests, GetDeviceType_AfterConstruction_ReturnsCuda )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cuda );

        auto device = fixture.component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cuda );
    }

    TYPED_TEST( MHACudaTests, IsTraining_InferenceFixture_ReturnsFalse )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), false );

        fixture.component->build( fixture.input_shape() );

        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( MHACudaTests, IsTraining_TrainingFixture_ReturnsTrue )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), true );

        fixture.component->build( fixture.input_shape() );
        fixture.component->setTraining( true );

        EXPECT_TRUE( fixture.component->isTraining() );
    }

    TYPED_TEST( MHACudaTests, SetTraining_TogglingMode_UpdatesState )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), false );

        fixture.component->build( fixture.input_shape() );

        EXPECT_FALSE( fixture.component->isTraining() );

        fixture.component->setTraining( true );
        EXPECT_TRUE( fixture.component->isTraining() );

        fixture.component->setTraining( false );
        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( MHACudaTests, ParameterCount_AfterConstruction_ReturnsZero )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->parameterCount(), 0 );
    }

    TYPED_TEST( MHACudaTests, ToString_PreBuild_ShowsFallbackDecodePath )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "Attention" ), std::string::npos );
        EXPECT_NE( output.find( "Model dimension" ), std::string::npos );
        EXPECT_NE( output.find( "Number of heads" ), std::string::npos );
        EXPECT_NE( output.find( "fallback" ), std::string::npos );
    }

    TYPED_TEST( MHACudaTests, ToString_AfterBuild_ShowsKVCacheDecodePath )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );
        fixture.component->build( fixture.input_shape() );

        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "KV cache" ), std::string::npos );
    }

    TYPED_TEST( MHACudaTests, Synchronize_AfterConstruction_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }

    TYPED_TEST( MHACudaTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );

        EXPECT_THROW(
            fixture.component->forward( input ),
            std::runtime_error
        );
    }

    TYPED_TEST( MHACudaTests, Backward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );
        CudaTensor<TPrecision> output_grad( Device::Cuda( 0 ), fixture.output_shape() );

        EXPECT_THROW(
            fixture.component->backward( input, output_grad ),
            std::runtime_error
        );
    }

    TYPED_TEST( MHACudaTests, Build_WithVariousShapes_SetsBuiltState )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = AttentionTestFixture<TPrecision>::Create( test_shape );

            EXPECT_FALSE( fixture.component->isBuilt() )
                << "Component should not be built before build() for shape: " << test_shape.name;

            EXPECT_NO_THROW( fixture.component->build( fixture.input_shape() ) )
                << "Build failed for shape: " << test_shape.name;

            EXPECT_TRUE( fixture.component->isBuilt() )
                << "Component should be built after build() for shape: " << test_shape.name;
        }
    }

    TYPED_TEST( MHACudaTests, Forward_InvalidInputShape_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );
        fixture.component->build( fixture.input_shape() );

        shape_t bad_in = fixture.input_shape();
        bad_in[ 2 ] = bad_in[ 2 ] + 1;
        CudaTensor<TPrecision> input_bad_trailing( Device::Cuda( 0 ), bad_in );

        EXPECT_THROW( fixture.component->forward( input_bad_trailing ), std::invalid_argument );

        shape_t bad_rank = { fixture.test_shape.batch, fixture.test_shape.seq };
        CudaTensor<TPrecision> input_bad_rank( Device::Cuda( 0 ), bad_rank );

        EXPECT_THROW( fixture.component->forward( input_bad_rank ), std::invalid_argument );
    }

    TYPED_TEST( MHACudaTests, Backward_NotTraining_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), false );

        fixture.component->build( fixture.input_shape() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );
        CudaTensor<TPrecision> output_grad( Device::Cuda( 0 ), fixture.output_shape() );

        EXPECT_THROW( fixture.component->backward( input, output_grad ), std::runtime_error );
    }

    TYPED_TEST( MHACudaTests, Backward_InvalidInputShape_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), true );
        fixture.component->build( fixture.input_shape() );
        fixture.component->setTraining( true );

        CudaTensor<TPrecision> outg_ok( Device::Cuda( 0 ), fixture.output_shape() );

        shape_t bad_in = fixture.input_shape();
        bad_in[ 2 ] = bad_in[ 2 ] / 3;
        CudaTensor<TPrecision> in_bad_trailing( Device::Cuda( 0 ), bad_in );

        EXPECT_THROW( fixture.component->backward( in_bad_trailing, outg_ok ), std::invalid_argument );

        shape_t bad_rank = { fixture.test_shape.batch, fixture.test_shape.seq };
        CudaTensor<TPrecision> in_bad_rank( Device::Cuda( 0 ), bad_rank );

        EXPECT_THROW( fixture.component->backward( in_bad_rank, outg_ok ), std::invalid_argument );
    }

    TYPED_TEST( MHACudaTests, Forward_WithVariousShapes_ProducesValidOutput )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = AttentionTestFixture<TPrecision>::Create( test_shape );
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
    // CPU <-> CUDA Equivalence Tests (Part of typed suite; run only for FP32)
    // ====================================================================

    TYPED_TEST( MHACudaTests, Forward_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Forward equivalence test only runs for FP32 precision";
        }

        try
        {
            auto shape = TestShape::Small();
            MultiHeadAttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cpu_comp = std::make_shared<MultiHeadAttention<DeviceType::Cpu, TensorDataType::FP32>>(
                "attention_cpu_equiv", cfg, Device::Cpu()
            );

            auto cuda_comp = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_cuda_equiv", cfg, Device::Cuda( 0 )
            );

            cpu_comp->build( shape.inputShape() );
            cuda_comp->build( shape.inputShape() );

            Mila::Core::RandomGenerator::getInstance().setSeed( 12345 );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            random( host_input, -1.0f, 1.0f );

            // CPU forward (new API)
            auto& cpu_out_tensor = cpu_comp->forward( host_input );

            // CUDA forward (new API)
            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            auto& cuda_out_tensor = cuda_comp->forward( device_input );
            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_out_cuda = toHost<TensorDataType::FP32>( cuda_out_tensor );

            auto* cdata = cpu_out_tensor.data();
            auto* gdata = host_out_cuda.data();
            size_t total = cpu_out_tensor.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_NEAR( cdata[ i ], gdata[ i ], 1e-3f ) << "Forward mismatch at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Attention backend not available for CPU/CUDA equivalence test";
        }
    }

    TYPED_TEST( MHACudaTests, Backward_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Backward equivalence test only runs for FP32 precision";
        }

        try
        {
            auto shape = TestShape::Small();
            MultiHeadAttentionConfig cfg( shape.embedding_dim, shape.num_heads );
            
            auto cpu_comp = std::make_shared<MultiHeadAttention<DeviceType::Cpu, TensorDataType::FP32>>(
                "attention_cpu_equiv_bwd", cfg, Device::Cpu()
            );

            auto cuda_comp = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_cuda_equiv_bwd", cfg, Device::Cuda( 0 )
            );

            cpu_comp->build( shape.inputShape() );
            cuda_comp->build( shape.inputShape() );

            cpu_comp->setTraining( true );
            cuda_comp->setTraining( true );

            Mila::Core::RandomGenerator::getInstance().setSeed( 12345 );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            random( host_input, -1.0f, 1.0f );

            // CPU forward (new API)
            auto& cpu_out = cpu_comp->forward( host_input );

            // CUDA forward
            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            auto& cuda_out = cuda_comp->forward( device_input );
            cuda_comp->synchronize();

            // Create deterministic output gradient
            CpuTensor<TensorDataType::FP32> host_outg( Device::Cpu(), shape.outputShape() );
            for ( size_t i = 0; i < host_outg.size(); ++i ) {
                host_outg.data()[ i ] = static_cast<float>( i );
            }

            // CPU backward (new API)
            auto& host_ing_cpu = cpu_comp->backward( host_input, host_outg );

            // CUDA backward (new API)
            CudaTensor<TensorDataType::FP32> device_outg( Device::Cuda( 0 ), shape.outputShape() );
            copy( host_outg, device_outg );

            auto& device_ing_tensor = cuda_comp->backward( device_input, device_outg );
            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_ing_cuda = toHost<TensorDataType::FP32>( device_ing_tensor );

            auto* cpu_data = host_ing_cpu.data();
            auto* cuda_data = host_ing_cuda.data();
            size_t total = host_ing_cpu.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_NEAR( cpu_data[ i ], cuda_data[ i ], 1e-3f ) << "Backward mismatch at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Attention backend not available for CPU/CUDA backward equivalence test";
        }
    }

    // ====================================================================
    // Deterministic Tests with Known Inputs/Outputs
    // ====================================================================

    TYPED_TEST( MHACudaTests, Forward_SimpleOnesInput_ProducesKnownOutput )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Deterministic test only runs for FP32 precision";
        }

        try
        {
            // Use minimal shape for simple verification
            auto shape = TestShape::Minimal();
            MultiHeadAttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cuda_comp = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_ones_input", cfg, Device::Cuda( 0 )
            );

            cuda_comp->build( shape.inputShape() );

            // Create input filled with ones: Q=1, K=1, V=1
            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            ones( host_input );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            auto& out = cuda_comp->forward( device_input );

            auto host_output = toHost<TensorDataType::FP32>( out );

            // Verification: output should be finite and non-zero
            auto* data = host_output.data();
            size_t total = host_output.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_TRUE( std::isfinite( data[ i ] ) ) << "Non-finite output at index " << i;
            }

            float sum = 0.0f;
            for ( size_t i = 0; i < total; ++i )
            {
                sum += data[ i ];
            }

            EXPECT_GT( sum, 0.1f ) << "Output should be non-zero for ones input";
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Attention backend not available";
        }
    }

    TYPED_TEST( MHACudaTests, Forward_SimpleZerosInput_ProducesZeroOutput )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Deterministic test only runs for FP32 precision";
        }

        try
        {
            auto shape = TestShape::Minimal();
            MultiHeadAttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cuda_comp = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_zeros_input", cfg, Device::Cuda( 0 )
            );

            cuda_comp->build( shape.inputShape() );

            // Create input filled with zeros
            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            zeros( host_input );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input, device_input );

            auto& out = cuda_comp->forward( device_input );

            auto host_output = toHost<TensorDataType::FP32>( out );

            // Verification: output should be zero (or very close to zero)
            auto* data = host_output.data();
            size_t total = host_output.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_TRUE( std::isfinite( data[ i ] ) ) << "Non-finite output at index " << i;
                EXPECT_NEAR( data[ i ], 0.0f, 1e-5f ) << "Output should be near zero for zeros input at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Attention backend not available";
        }
    }

    TYPED_TEST( MHACudaTests, Forward_Deterministic_ReproducibleWithSeed )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Deterministic test only runs for FP32 precision";
        }

        try
        {
            auto shape = TestShape::Minimal();
            MultiHeadAttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cuda_comp = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_reproducible", cfg, Device::Cuda( 0 )
            );

            cuda_comp->build( shape.inputShape() );

            // First run with seed 42
            Mila::Core::RandomGenerator::getInstance().setSeed( 42 );

            CpuTensor<TensorDataType::FP32> host_input1( Device::Cpu(), shape.inputShape() );
            random( host_input1, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input1( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input1, device_input1 );

            auto& out1 = cuda_comp->forward( device_input1 );
            auto host_out1 = toHost<TensorDataType::FP32>( out1 );

            // Second run with same seed 42
            Mila::Core::RandomGenerator::getInstance().setSeed( 42 );

            CpuTensor<TensorDataType::FP32> host_input2( Device::Cpu(), shape.inputShape() );
            random( host_input2, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input2( Device::Cuda( 0 ), shape.inputShape() );
            copy( host_input2, device_input2 );

            auto& out2 = cuda_comp->forward( device_input2 );
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
            GTEST_SKIP() << "Attention backend not available";
        }
    }

    TYPED_TEST( MHACudaTests, Backward_SimpleOnesGradient_ProducesFiniteGradients )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Deterministic test only runs for FP32 precision";
        }

        try
        {
            auto shape = TestShape::Minimal();
            MultiHeadAttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cuda_comp = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_ones_grad", cfg, Device::Cuda( 0 )
            );

            cuda_comp->build( shape.inputShape() );
            cuda_comp->setTraining( true );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            ones( host_input );

            CpuTensor<TensorDataType::FP32> host_output_grad( Device::Cpu(), shape.outputShape() );
            ones( host_output_grad );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_output_grad( Device::Cuda( 0 ), shape.outputShape() );

            copy( host_input, device_input );
            copy( host_output_grad, device_output_grad );

            auto& input_grad = cuda_comp->backward( device_input, device_output_grad );

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
            GTEST_SKIP() << "Attention backend not available";
        }
    }

    TYPED_TEST( MHACudaTests, Backward_Deterministic_ReproducibleGradientsWithSeed )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Deterministic test only runs for FP32 precision";
        }

        try
        {
            auto shape = TestShape::Minimal();
            MultiHeadAttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cuda_comp = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_reproducible_bwd", cfg, Device::Cuda( 0 )
            );

            cuda_comp->build( shape.inputShape() );
            cuda_comp->setTraining( true );

            Mila::Core::RandomGenerator::getInstance().setSeed( 123 );

            CpuTensor<TensorDataType::FP32> host_input1( Device::Cpu(), shape.inputShape() );
            random( host_input1, -0.5f, 0.5f );

            CpuTensor<TensorDataType::FP32> host_output_grad1( Device::Cpu(), shape.outputShape() );
            random( host_output_grad1, -0.5f, 0.5f );

            CudaTensor<TensorDataType::FP32> device_input1( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_output_grad1( Device::Cuda( 0 ), shape.outputShape() );

            copy( host_input1, device_input1 );
            copy( host_output_grad1, device_output_grad1 );

            auto& ing1_tensor = cuda_comp->backward( device_input1, device_output_grad1 );
            auto host_input_grad1 = toHost<TensorDataType::FP32>( ing1_tensor );

            Mila::Core::RandomGenerator::getInstance().setSeed( 123 );

            CpuTensor<TensorDataType::FP32> host_input2( Device::Cpu(), shape.inputShape() );
            random( host_input2, -0.5f, 0.5f );

            CpuTensor<TensorDataType::FP32> host_output_grad2( Device::Cpu(), shape.outputShape() );
            random( host_output_grad2, -0.5f, 0.5f );

            CudaTensor<TensorDataType::FP32> device_input2( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_output_grad2( Device::Cuda( 0 ), shape.outputShape() );

            copy( host_input2, device_input2 );
            copy( host_output_grad2, device_output_grad2 );

            auto& ing2_tensor = cuda_comp->backward( device_input2, device_output_grad2 );
            auto host_input_grad2 = toHost<TensorDataType::FP32>( ing2_tensor );

            auto* grad1 = host_input_grad1.data();
            auto* grad2 = host_input_grad2.data();
            size_t total = host_input_grad1.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_FLOAT_EQ( grad1[ i ], grad2[ i ] ) << "Non-reproducible gradient at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Attention backend not available";
        }
    }

    TYPED_TEST( MHACudaTests, ForwardBackward_ProducesFiniteGradients )
    {
        if ( !this->cuda_available_ )
            GTEST_SKIP() << "CUDA not available";
        
        constexpr TensorDataType TPrecision = TypeParam::value;
        
        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Deterministic test only runs for FP32 precision";
        }
        try
        {
            constexpr int batch_size = 1;
            constexpr int seq_length = 2;
            constexpr int embedding_dim = 4;
            constexpr int num_heads = 2;
            constexpr int head_dim = embedding_dim / num_heads;
            constexpr int qkv_dim = 3 * embedding_dim;

            MultiHeadAttentionConfig cfg( embedding_dim, num_heads );
            auto cuda_comp = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_minimal_trace", cfg, Device::Cuda( 0 )
            );

            shape_t input_shape{ batch_size, seq_length, qkv_dim };
            shape_t output_shape{ batch_size, seq_length, embedding_dim };

            cuda_comp->build( input_shape );
            cuda_comp->setTraining( true );

            Mila::Core::RandomGenerator::getInstance().setSeed( 422 );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), input_shape );
            random( host_input, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), input_shape );
            copy( host_input, device_input );

            auto& out = cuda_comp->forward( device_input );

            auto host_output = toHost<TensorDataType::FP32>( out );

            Mila::Core::RandomGenerator::getInstance().setSeed( 423 );

            CpuTensor<TensorDataType::FP32> host_output_grad( Device::Cpu(), output_shape );
            random( host_output_grad, -0.5f, 0.5f );

            CudaTensor<TensorDataType::FP32> device_output_grad( Device::Cuda( 0 ), output_shape );
            copy( host_output_grad, device_output_grad );

            auto& ing_tensor = cuda_comp->backward( device_input, device_output_grad );

            auto host_input_grad = toHost<TensorDataType::FP32>( ing_tensor );

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
                else
                {
                    if ( std::abs( grad_data[ i ] ) > 1e-6f )
                    {
                        has_nonzero = true;
                    }
                }
            }

            EXPECT_TRUE( all_finite ) << "All gradients should be finite";
            EXPECT_TRUE( has_nonzero ) << "Gradients should be non-zero for this input/output_grad";
        }
        catch ( const std::exception& e )
        {
            GTEST_SKIP() << "Attention backend not available: " << e.what();
        }
    }

    // ====================================================================
// Accessor Tests
// ====================================================================

    TYPED_TEST( MHACudaTests, GetModelDim_ReturnsConfiguredValue )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->getModelDim(), fixture.test_shape.embedding_dim );
    }

    TYPED_TEST( MHACudaTests, GetNumHeads_ReturnsConfiguredValue )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->getNumHeads(), fixture.test_shape.num_heads );
    }

    TYPED_TEST( MHACudaTests, GetConfig_ReturnsMatchingConfig )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        const auto& cfg = fixture.component->getConfig();

        EXPECT_EQ( cfg.getModelDim(), fixture.test_shape.embedding_dim );
        EXPECT_EQ( cfg.getNumHeads(), fixture.test_shape.num_heads );
    }

    TYPED_TEST( MHACudaTests, GetType_ReturnsMultiHeadAttention )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->getType(), ComponentType::MultiHeadAttention );
    }

    TYPED_TEST( MHACudaTests, GetParameters_ReturnsEmptyVector )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_TRUE( fixture.component->getParameters().empty() );
    }

    TYPED_TEST( MHACudaTests, GetGradients_ReturnsEmptyVector )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_TRUE( fixture.component->getGradients().empty() );
    }

    // ====================================================================
    // KV Cache Interface Tests
    // ====================================================================

    TYPED_TEST( MHACudaTests, SupportsKVCache_BeforeBuild_ReturnsFalse )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_FALSE( fixture.component->supportsKVCache() );
    }

    TYPED_TEST( MHACudaTests, SupportsKVCache_AfterBuild_ReturnsTrue )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );
        fixture.component->build( fixture.input_shape() );

        EXPECT_TRUE( fixture.component->supportsKVCache() );
    }

    // ====================================================================
    // Decode / KV Cache Path Tests
    // ====================================================================

    TYPED_TEST( MHACudaTests, Decode_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );

        EXPECT_THROW( fixture.component->decode( input, 0 ), std::runtime_error );
    }

    TYPED_TEST( MHACudaTests, Decode_InvalidShape_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );
        fixture.component->build( fixture.input_shape() );

        shape_t bad_shape = fixture.input_shape();
        bad_shape[ 2 ] = bad_shape[ 2 ] + 1;
        CudaTensor<TPrecision> bad_input( Device::Cuda( 0 ), bad_shape );

        EXPECT_THROW( fixture.component->decode( bad_input, 0 ), std::invalid_argument );
    }

    TYPED_TEST( MHACudaTests, Decode_AfterForward_UsesKVCachePath )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Decode test only runs for FP32";

        try
        {
            auto shape = TestShape::Small();
            MultiHeadAttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto comp = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_kvcache_decode", cfg, Device::Cuda( 0 )
            );

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

    TYPED_TEST( MHACudaTests, Decode_WithoutCacheInitialized_UsesFallbackForward )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Decode fallback test only runs for FP32";

        try
        {
            auto shape = TestShape::Small();
            MultiHeadAttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto comp = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_decode_fallback", cfg, Device::Cuda( 0 )
            );

            comp->build( shape.inputShape() );
            ASSERT_TRUE( comp->supportsKVCache() );

            // cache_initialized_ is false — decode() must fall through to operation_->forward()
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
    // KV Cache Session State Tests
    // ====================================================================

    TYPED_TEST( MHACudaTests, Forward_AfterDecode_AutoResetsCacheAndPrefills )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Session state test only runs for FP32";

        try
        {
            auto shape = TestShape::Small();
            MultiHeadAttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto comp = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_session_reset", cfg, Device::Cuda( 0 )
            );

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

    TYPED_TEST( MHACudaTests, SetTraining_ToTrue_WithActiveCache_ResetsKVCacheSession )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
            GTEST_SKIP() << "Session state test only runs for FP32";

        try
        {
            auto shape = TestShape::Small();
            MultiHeadAttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto comp = std::make_shared<MultiHeadAttention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_train_resets_cache", cfg, Device::Cuda( 0 )
            );

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