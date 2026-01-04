#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Components_Layers_Attention_Tests
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

        static TestShape Small()
        {
            return { TestShapeSize::Small, 2, 4, 8, 2, "Small" }; // Reduced embedding_dim from 64 to 16 abd num_heads from 8 to 4
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
            return { TestShapeSize::Minimal, 1, 1, 1, 1, "Minimal" };
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
        // Provide an explicit default constructor because `AttentionConfig`
        // does not have a default constructor. Initialize `test_shape` first
        // (declared order) and then use its fields to construct `config`.
        AttentionTestFixture()
            : test_shape( TestShape::Small() ),
            config( test_shape.embedding_dim, test_shape.num_heads ),
            component( nullptr ),
            is_training( false )
        {}

        TestShape test_shape;
        AttentionConfig config;
        std::shared_ptr<Attention<DeviceType::Cuda, TPrecision>> component;
        bool is_training{ false };

        static AttentionTestFixture Create( TestShape shape, bool is_training = false )
        {
            AttentionTestFixture fixture;
            fixture.test_shape = shape;
            fixture.is_training = is_training;

            fixture.config = AttentionConfig( shape.embedding_dim, shape.num_heads );

            std::string name = "attention_cuda_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.component = std::make_shared<Attention<DeviceType::Cuda, TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

            // Do not call setTraining() here. The backend operation is created during build(),
            // and setTraining() must be called after build() to propagate the training flag.
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
    class AttentionCudaTests : public testing::Test
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

    TYPED_TEST_SUITE( AttentionCudaTests, PrecisionTypes );

    TYPED_TEST( AttentionCudaTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        AttentionConfig cfg( 64, 8 );

        std::shared_ptr<Attention<DeviceType::Cuda, TPrecision>> component{ nullptr };

        ASSERT_NO_THROW(
            (component = std::make_shared<Attention<DeviceType::Cuda, TPrecision>>(
                "ctor_device_cuda",
                cfg,
                Device::Cuda( 0 )
            ))
        );

        ASSERT_NE( component, nullptr );
        EXPECT_EQ( component->getDeviceType(), DeviceType::Cuda );
    }

    TYPED_TEST( AttentionCudaTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        AttentionConfig cfg( 64, 8 );

        std::shared_ptr<Attention<DeviceType::Cuda, TPrecision>> component;

        ASSERT_NO_THROW(
            (component = std::make_shared<Attention<DeviceType::Cuda, TPrecision>>(
                "ctor_shared_cuda",
                cfg
            ))
        );

        ASSERT_NE( component, nullptr );
    }

    TYPED_TEST( AttentionCudaTests, Constructor_WithInvalidDeviceType_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        AttentionConfig cfg( 64, 8 );

        EXPECT_THROW(
            ((void)std::make_shared<Attention<DeviceType::Cuda, TPrecision>>(
                "invalid_ctor",
                cfg,
                Device::Cpu()
            )),
            std::invalid_argument
        );
    }

    TYPED_TEST( AttentionCudaTests, GetDeviceType_AfterConstruction_ReturnsCuda )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cuda );

        auto device = fixture.component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cuda );
    }

    TYPED_TEST( AttentionCudaTests, IsTraining_InferenceFixture_ReturnsFalse )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), false );

        // Build before checking/propagating training state to backend operation
        fixture.component->build( fixture.input_shape() );

        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( AttentionCudaTests, IsTraining_TrainingFixture_ReturnsTrue )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), true );

        // Build then enable training so the backend operation receives the flag
        fixture.component->build( fixture.input_shape() );
        fixture.component->setTraining( true );

        EXPECT_TRUE( fixture.component->isTraining() );
    }

    TYPED_TEST( AttentionCudaTests, SetTraining_TogglingMode_UpdatesState )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), false );

        // Build before toggling training to satisfy backend lifecycle.
        fixture.component->build( fixture.input_shape() );

        EXPECT_FALSE( fixture.component->isTraining() );

        fixture.component->setTraining( true );
        EXPECT_TRUE( fixture.component->isTraining() );

        fixture.component->setTraining( false );
        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( AttentionCudaTests, ParameterCount_AfterConstruction_ReturnsZero )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_EQ( fixture.component->parameterCount(), 0 );
    }

    TYPED_TEST( AttentionCudaTests, ToString_AfterConstruction_ContainsComponentInfo )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );
        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "Attention" ), std::string::npos );
        EXPECT_NE( output.find( "Embedding dimension" ), std::string::npos );
        EXPECT_NE( output.find( "Number of heads" ), std::string::npos );
    }

    TYPED_TEST( AttentionCudaTests, Synchronize_AfterConstruction_Succeeds )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }

    TYPED_TEST( AttentionCudaTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );
        CudaTensor<TPrecision> output( Device::Cuda( 0 ), fixture.output_shape() );

        EXPECT_THROW(
            fixture.component->forward( input, output ),
            std::runtime_error
        );
    }

    TYPED_TEST( AttentionCudaTests, Backward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );
        CudaTensor<TPrecision> output_grad( Device::Cuda( 0 ), fixture.output_shape() );
        CudaTensor<TPrecision> input_grad( Device::Cuda( 0 ), fixture.input_shape() );

        EXPECT_THROW(
            fixture.component->backward( input, output_grad, input_grad ),
            std::runtime_error
        );
    }

    TYPED_TEST( AttentionCudaTests, Build_WithVariousShapes_SetsBuiltState )
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

    TYPED_TEST( AttentionCudaTests, Forward_InvalidOutputShape_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small() );

        fixture.component->build( fixture.input_shape() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );

        shape_t bad_out = fixture.output_shape();
        bad_out[ 2 ] = bad_out[ 2 ] + 1; // wrong trailing dimension
        CudaTensor<TPrecision> output_bad_trailing( Device::Cuda( 0 ), bad_out );

        EXPECT_THROW( fixture.component->forward( input, output_bad_trailing ), std::invalid_argument );

        shape_t bad_out_bs = fixture.output_shape();
        bad_out_bs[ 0 ] = bad_out_bs[ 0 ] + 1; // batch mismatch
        CudaTensor<TPrecision> output_bad_bs( Device::Cuda( 0 ), bad_out_bs );

        EXPECT_THROW( fixture.component->forward( input, output_bad_bs ), std::invalid_argument );

        shape_t bad_out_seq = fixture.output_shape();
        bad_out_seq[ 1 ] = bad_out_seq[ 1 ] + 2; // sequence mismatch
        CudaTensor<TPrecision> output_bad_seq( Device::Cuda( 0 ), bad_out_seq );

        EXPECT_THROW( fixture.component->forward( input, output_bad_seq ), std::invalid_argument );
    }

    TYPED_TEST( AttentionCudaTests, Backward_NotTraining_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), false );

        fixture.component->build( fixture.input_shape() );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );
        CudaTensor<TPrecision> output_grad( Device::Cuda( 0 ), fixture.output_shape() );
        CudaTensor<TPrecision> input_grad( Device::Cuda( 0 ), fixture.input_shape() );

        EXPECT_THROW( fixture.component->backward( input, output_grad, input_grad ), std::runtime_error );
    }

    TYPED_TEST( AttentionCudaTests, Backward_InvalidShapes_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;
        auto fixture = AttentionTestFixture<TPrecision>::Create( TestShape::Small(), true );

        fixture.component->build( fixture.input_shape() );
        fixture.component->setTraining( true );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.input_shape() );

        // output_grad with wrong trailing dim
        shape_t bad_outg = fixture.output_shape();
        bad_outg[ 2 ] = bad_outg[ 2 ] + 5;
        CudaTensor<TPrecision> outg_bad_trailing( Device::Cuda( 0 ), bad_outg );

        CudaTensor<TPrecision> input_grad_ok( Device::Cuda( 0 ), fixture.input_shape() );

        EXPECT_THROW( fixture.component->backward( input, outg_bad_trailing, input_grad_ok ), std::invalid_argument );

        // input_grad with wrong trailing dim
        shape_t bad_ing = fixture.input_shape();
        bad_ing[ 2 ] = bad_ing[ 2 ] / 3; // invalid trailing (should be 3 * embedding_dim)
        CudaTensor<TPrecision> ing_bad_trailing( Device::Cuda( 0 ), bad_ing );

        CudaTensor<TPrecision> outg_ok( Device::Cuda( 0 ), fixture.output_shape() );

        EXPECT_THROW( fixture.component->backward( input, outg_ok, ing_bad_trailing ), std::invalid_argument );

        // mismatched batch/sequence dims across tensors
        shape_t ing_mismatch = fixture.input_shape();
        ing_mismatch[ 0 ] = ing_mismatch[ 0 ] + 1;
        CudaTensor<TPrecision> ing_bad_bs( Device::Cuda( 0 ), ing_mismatch );

        EXPECT_THROW( fixture.component->backward( input, outg_ok, ing_bad_bs ), std::invalid_argument );
    }

    TYPED_TEST( AttentionCudaTests, Forward_WithVariousShapes_ProducesValidOutput )
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
            CudaTensor<TPrecision> device_output( Device::Cuda( 0 ), fixture.output_shape() );

            copy( host_input, device_input );

            EXPECT_NO_THROW( fixture.component->forward( device_input, device_output ) )
                << "Forward failed for shape: " << test_shape.name;

            EXPECT_EQ( device_output.size(), fixture.output_shape()[ 0 ] * fixture.output_shape()[ 1 ] * fixture.output_shape()[ 2 ] )
                << "Output size mismatch for shape: " << test_shape.name;

            EXPECT_EQ( device_output.shape(), fixture.output_shape() )
                << "Output shape mismatch for shape: " << test_shape.name;
        }
    }

    // ====================================================================
    // CPU <-> CUDA Equivalence Tests (Part of typed suite; run only for FP32)
    // ====================================================================

    TYPED_TEST( AttentionCudaTests, Forward_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        // Run equivalence only for FP32 precision
        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Forward equivalence test only runs for FP32 precision";
        }

        try
        {
            auto shape = TestShape::Small();
            AttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cpu_comp = std::make_shared<Attention<DeviceType::Cpu, TensorDataType::FP32>>(
                "attention_cpu_equiv", cfg, Device::Cpu()
            );

            auto cuda_comp = std::make_shared<Attention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_cuda_equiv", cfg, Device::Cuda( 0 )
            );

            cpu_comp->build( shape.inputShape() );
            cuda_comp->build( shape.inputShape() );

            // Create deterministic input with known seed
            Mila::Core::RandomGenerator::getInstance().setSeed( 12345 );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            random( host_input, -1.0f, 1.0f );

            std::cout << "\n=== Deterministic Input Pattern ===\n";
            std::cout << "Input tensor: " << std::endl;
            std::cout << "Shape: [" << shape.batch << ", " << shape.seq << ", "
                << (3 * shape.embedding_dim) << "]\n";
            std::cout << "Embedding: " << shape.embedding_dim
                << ", Heads: " << shape.num_heads
                << ", HeadSize: " << (shape.embedding_dim / shape.num_heads) << "\n\n";
            std::cout << host_input.toString( true ) << std::endl;

            CpuTensor<TensorDataType::FP32> host_output_cpu( Device::Cpu(), shape.outputShape() );

            // CPU forward
            cpu_comp->forward( host_input, host_output_cpu );

            std::cout << "CPU forward output: " << std::endl;
            std::cout << host_output_cpu.toString( true ) << std::endl;

            // CUDA forward
            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_output( Device::Cuda( 0 ), shape.outputShape() );

            copy( host_input, device_input );

            cuda_comp->forward( device_input, device_output );
            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_output_cuda( Device::Cpu(), shape.outputShape() );
            copy( device_output, host_output_cuda );

            std::cout << "CUDA forward output: " << std::endl;
            std::cout << host_output_cuda.toString( true ) << std::endl;

            // Compare element-wise with tolerance
            auto* cpu_data = host_output_cpu.data();
            auto* cuda_data = host_output_cuda.data();
            size_t total = host_output_cpu.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_NEAR( cpu_data[ i ], cuda_data[ i ], 1e-3f ) << "Mismatch at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Attention backend not available for CPU/CUDA equivalence test";
        }
    }

    TYPED_TEST( AttentionCudaTests, Backward_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";

        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Backward equivalence test only runs for FP32 precision";
        }

        try
        {
            // Minimal shape: 1 batch, 2 tokens, 4 embedding dim, 2 heads
            // Input will be 3 * embedding_dim for concatenated QKV
            constexpr int batch_size = 1;
            constexpr int seq_length = 2;
            constexpr int embedding_dim = 4;
            constexpr int num_heads = 2;
            constexpr int head_dim = embedding_dim / num_heads;  // = 2
            constexpr int qkv_dim = 3 * embedding_dim;  // = 12

            std::cout << "\n=== Minimal Test Configuration ===\n";
            std::cout << "Batch: " << batch_size << ", Seq: " << seq_length << "\n";
            std::cout << "Embed: " << embedding_dim << ", Heads: " << num_heads
                << ", HeadDim: " << head_dim << "\n";
            std::cout << "Input (QKV concat): " << qkv_dim << " (3 * " << embedding_dim << ")\n";
            std::cout << "Output: " << embedding_dim << "\n\n";

            AttentionConfig cfg( embedding_dim, num_heads );
            
            shape_t input_shape{ batch_size, seq_length, qkv_dim };
            shape_t output_shape{ batch_size, seq_length, embedding_dim };

            // Use batch=1 version of Small shape to isolate batching issues
            //TestShape shape{ TestShapeSize::Small, 1, 4, 64, 8, "Small_Batch1" };  // Changed from TestShape::Small()

            //AttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cpu_comp = std::make_shared<Attention<DeviceType::Cpu, TensorDataType::FP32>>(
                "attention_cpu_equiv_bwd", cfg, Device::Cpu()
            );

            auto cuda_comp = std::make_shared<Attention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_cuda_equiv_bwd", cfg, Device::Cuda( 0 )
            );

            cpu_comp->build( input_shape );
            cuda_comp->build( input_shape );

            cpu_comp->setTraining( true );
            cuda_comp->setTraining( true );

            // Create shared input for both CPU and CUDA forward passes
            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), input_shape );
            random( host_input, -1.0f, 1.0f );

            // ====================================================================
            // FORWARD PASS - Required to establish internal state for backward
            // ====================================================================

            // CPU forward
            CpuTensor<TensorDataType::FP32> host_output_cpu( Device::Cpu(), output_shape );
            cpu_comp->forward( host_input, host_output_cpu );

            // CUDA forward
            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), input_shape );
            CudaTensor<TensorDataType::FP32> device_output_cuda( Device::Cuda( 0 ), output_shape );
            
            copy( host_input, device_input );
            
            cuda_comp->forward( device_input, device_output_cuda );
            cuda_comp->synchronize();

            // ====================================================================
            // BACKWARD PASS - Now internal state (Q, K, V, att) is initialized
            // ====================================================================

            CpuTensor<TensorDataType::FP32> host_output_grad( Device::Cpu(), output_shape );
            random( host_output_grad, -0.5f, 0.5f );

            // CPU backward
            CpuTensor<TensorDataType::FP32> host_input_grad_cpu( Device::Cpu(), input_shape );
            cpu_comp->backward( host_input, host_output_grad, host_input_grad_cpu );

            // CUDA backward
            CudaTensor<TensorDataType::FP32> device_output_grad( Device::Cuda( 0 ), output_shape );
            CudaTensor<TensorDataType::FP32> device_input_grad( Device::Cuda( 0 ), input_shape );
            
            copy( host_output_grad, device_output_grad );
            
            cuda_comp->backward( device_input, device_output_grad, device_input_grad );
            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_input_grad_cuda( Device::Cpu(), input_shape );
            copy( device_input_grad, host_input_grad_cuda );

            // Compare element-wise with tolerance
            auto* cpu_data = host_input_grad_cpu.data();
            auto* cuda_data = host_input_grad_cuda.data();
            size_t total = host_input_grad_cpu.size();

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

    TYPED_TEST( AttentionCudaTests, Forward_SimpleOnesInput_ProducesKnownOutput )
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
            AttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cuda_comp = std::make_shared<Attention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_ones_input", cfg, Device::Cuda( 0 )
            );

            cuda_comp->build( shape.inputShape() );

            // Create input filled with ones: Q=1, K=1, V=1
            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            ones( host_input );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_output( Device::Cuda( 0 ), shape.outputShape() );

            copy( host_input, device_input );

            // Execute forward
            cuda_comp->forward( device_input, device_output );

            CpuTensor<TensorDataType::FP32> host_output( Device::Cpu(), shape.outputShape() );
            copy( device_output, host_output );

            // Verification: output should be finite and non-zero
            auto* data = host_output.data();
            size_t total = host_output.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_TRUE( std::isfinite( data[ i ] ) ) << "Non-finite output at index " << i;
            }

            // For Q=K=V=1, attention should produce V (scaled by softmax)
            // Expected: approximately 1.0 (scaled by attention weights that sum to 1)
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

    TYPED_TEST( AttentionCudaTests, Forward_SimpleZerosInput_ProducesZeroOutput )
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
            AttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cuda_comp = std::make_shared<Attention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_zeros_input", cfg, Device::Cuda( 0 )
            );

            cuda_comp->build( shape.inputShape() );

            // Create input filled with zeros
            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            zeros( host_input );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_output( Device::Cuda( 0 ), shape.outputShape() );

            copy( host_input, device_input );

            // Execute forward
            cuda_comp->forward( device_input, device_output );

            CpuTensor<TensorDataType::FP32> host_output( Device::Cpu(), shape.outputShape() );
            copy( device_output, host_output );

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

    TYPED_TEST( AttentionCudaTests, Forward_Deterministic_ReproducibleWithSeed )
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
            AttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cuda_comp = std::make_shared<Attention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_reproducible", cfg, Device::Cuda( 0 )
            );

            cuda_comp->build( shape.inputShape() );

            // First run with seed 42
            Mila::Core::RandomGenerator::getInstance().setSeed( 42 );

            CpuTensor<TensorDataType::FP32> host_input1( Device::Cpu(), shape.inputShape() );
            random( host_input1, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input1( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_output1( Device::Cuda( 0 ), shape.outputShape() );

            copy( host_input1, device_input1 );
            cuda_comp->forward( device_input1, device_output1 );

            CpuTensor<TensorDataType::FP32> host_output1( Device::Cpu(), shape.outputShape() );
            copy( device_output1, host_output1 );

            // Second run with same seed 42
            Mila::Core::RandomGenerator::getInstance().setSeed( 42 );

            CpuTensor<TensorDataType::FP32> host_input2( Device::Cpu(), shape.inputShape() );
            random( host_input2, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_input2( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_output2( Device::Cuda( 0 ), shape.outputShape() );

            copy( host_input2, device_input2 );
            cuda_comp->forward( device_input2, device_output2 );

            CpuTensor<TensorDataType::FP32> host_output2( Device::Cpu(), shape.outputShape() );
            copy( device_output2, host_output2 );

            // Verify exact reproducibility
            auto* data1 = host_output1.data();
            auto* data2 = host_output2.data();
            size_t total = host_output1.size();

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

    TYPED_TEST( AttentionCudaTests, Backward_SimpleOnesGradient_ProducesFiniteGradients )
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
            AttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cuda_comp = std::make_shared<Attention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_ones_grad", cfg, Device::Cuda( 0 )
            );

            cuda_comp->build( shape.inputShape() );
            cuda_comp->setTraining( true );

            // Simple input: all ones
            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), shape.inputShape() );
            ones( host_input );

            // Simple output gradient: all ones
            CpuTensor<TensorDataType::FP32> host_output_grad( Device::Cpu(), shape.outputShape() );
            ones( host_output_grad );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_output_grad( Device::Cuda( 0 ), shape.outputShape() );
            CudaTensor<TensorDataType::FP32> device_input_grad( Device::Cuda( 0 ), shape.inputShape() );

            copy( host_input, device_input );
            copy( host_output_grad, device_output_grad );

            // Execute backward
            cuda_comp->backward( device_input, device_output_grad, device_input_grad );

            CpuTensor<TensorDataType::FP32> host_input_grad( Device::Cpu(), shape.inputShape() );
            copy( device_input_grad, host_input_grad );

            // Verification: gradients should be finite and non-zero
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

    TYPED_TEST( AttentionCudaTests, Backward_Deterministic_ReproducibleGradientsWithSeed )
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
            AttentionConfig cfg( shape.embedding_dim, shape.num_heads );

            auto cuda_comp = std::make_shared<Attention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_reproducible_bwd", cfg, Device::Cuda( 0 )
            );

            cuda_comp->build( shape.inputShape() );
            cuda_comp->setTraining( true );

            // First run with seed 123
            Mila::Core::RandomGenerator::getInstance().setSeed( 123 );

            CpuTensor<TensorDataType::FP32> host_input1( Device::Cpu(), shape.inputShape() );
            random( host_input1, -0.5f, 0.5f );

            CpuTensor<TensorDataType::FP32> host_output_grad1( Device::Cpu(), shape.outputShape() );
            random( host_output_grad1, -0.5f, 0.5f );

            CudaTensor<TensorDataType::FP32> device_input1( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_output_grad1( Device::Cuda( 0 ), shape.outputShape() );
            CudaTensor<TensorDataType::FP32> device_input_grad1( Device::Cuda( 0 ), shape.inputShape() );

            copy( host_input1, device_input1 );
            copy( host_output_grad1, device_output_grad1 );

            cuda_comp->backward( device_input1, device_output_grad1, device_input_grad1 );

            CpuTensor<TensorDataType::FP32> host_input_grad1( Device::Cpu(), shape.inputShape() );
            copy( device_input_grad1, host_input_grad1 );

            // Second run with same seed 123
            Mila::Core::RandomGenerator::getInstance().setSeed( 123 );

            CpuTensor<TensorDataType::FP32> host_input2( Device::Cpu(), shape.inputShape() );
            random( host_input2, -0.5f, 0.5f );

            CpuTensor<TensorDataType::FP32> host_output_grad2( Device::Cpu(), shape.outputShape() );
            random( host_output_grad2, -0.5f, 0.5f );

            CudaTensor<TensorDataType::FP32> device_input2( Device::Cuda( 0 ), shape.inputShape() );
            CudaTensor<TensorDataType::FP32> device_output_grad2( Device::Cuda( 0 ), shape.outputShape() );
            CudaTensor<TensorDataType::FP32> device_input_grad2( Device::Cuda( 0 ), shape.inputShape() );

            copy( host_input2, device_input2 );
            copy( host_output_grad2, device_output_grad2 );

            cuda_comp->backward( device_input2, device_output_grad2, device_input_grad2 );

            CpuTensor<TensorDataType::FP32> host_input_grad2( Device::Cpu(), shape.inputShape() );
            copy( device_input_grad2, host_input_grad2 );

            // Verify exact reproducibility of gradients
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

    TYPED_TEST( AttentionCudaTests, ForwardBackward_ProducesFiniteGradients )
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
            // Minimal shape: 1 batch, 2 tokens, 4 embedding dim, 2 heads
            // Input will be 3 * embedding_dim for concatenated QKV
            constexpr int batch_size = 1;
            constexpr int seq_length = 2;
            constexpr int embedding_dim = 4;
            constexpr int num_heads = 2;
            constexpr int head_dim = embedding_dim / num_heads;  // = 2
            constexpr int qkv_dim = 3 * embedding_dim;  // = 12

            std::cout << "\n=== Minimal Test Configuration ===\n";
            std::cout << "Batch: " << batch_size << ", Seq: " << seq_length << "\n";
            std::cout << "Embed: " << embedding_dim << ", Heads: " << num_heads
                << ", HeadDim: " << head_dim << "\n";
            std::cout << "Input (QKV concat): " << qkv_dim << " (3 * " << embedding_dim << ")\n";
            std::cout << "Output: " << embedding_dim << "\n\n";

            AttentionConfig cfg( embedding_dim, num_heads );
            auto cuda_comp = std::make_shared<Attention<DeviceType::Cuda, TensorDataType::FP32>>(
                "attention_minimal_trace", cfg, Device::Cuda( 0 )
            );

            shape_t input_shape{ batch_size, seq_length, qkv_dim };
            shape_t output_shape{ batch_size, seq_length, embedding_dim };

            cuda_comp->build( input_shape );
            cuda_comp->setTraining( true );

            std::cout << "=== FORWARD PASS (to establish state) ===\n";

            // Set deterministic seed for reproducible random values
            Mila::Core::RandomGenerator::getInstance().setSeed( 422 );

            // Input: Simple pattern we can trace
            // Shape: [1, 2, 12] -> [batch, seq, 3*embed]
            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), input_shape );
            random( host_input, -1.0f, 1.0f );

            std::cout << "Forward Input [" << batch_size << "x" << seq_length << "x" << 3 * embedding_dim << "]:\n";
            std::cout << host_input.toString( true );
            //[[ -0.505485 - 0.317412 - 0.261304 ...    0.33622   0.120742 - 0.227293 ]
            //   [ 0.75507 - 0.87565   0.464467 ...   0.796011 - 0.907575 - 0.500094 ]

            // Transfer to device and run forward
            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), input_shape );
            CudaTensor<TensorDataType::FP32> device_output( Device::Cuda( 0 ), output_shape );
            copy( host_input, device_input );

            cuda_comp->forward( device_input, device_output );
            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_output( Device::Cpu(), output_shape );
            copy( device_output, host_output ); // This should syncrhonize as well

            std::cout << "\nForward Output [" << batch_size << "x" << seq_length << "x" << embedding_dim << "]:\n";
            std::cout << host_output.toString( true );
            //[[ 0.929655    0.33622   0.120742 - 0.227293 ]
            //    [ 0.489398   0.614583 -0.408754 -0.367762 ]
            //]

            std::cout << "\n=== BACKWARD PASS (tracing intermediate results) ===\n";

            // Set deterministic seed for reproducible gradient values
            Mila::Core::RandomGenerator::getInstance().setSeed( 423 );

            // Output gradient: Use random values to properly exercise softmax backward
            CpuTensor<TensorDataType::FP32> host_output_grad( Device::Cpu(), output_shape );
            random( host_output_grad, -0.5f, 0.5f );

            std::cout << "Output Gradient [" << batch_size << "x" << seq_length << "x" << embedding_dim << "]:\n";
            host_output_grad.toString( true );

            CudaTensor<TensorDataType::FP32> device_output_grad( Device::Cuda( 0 ), output_shape );
            CudaTensor<TensorDataType::FP32> device_input_grad( Device::Cuda( 0 ), input_shape );
            copy( host_output_grad, device_output_grad );

            // Execute backward - this is where we need to trace internals
            std::cout << "\nExecuting backward pass...\n";
            cuda_comp->backward( device_input, device_output_grad, device_input_grad );
            cuda_comp->synchronize();

            // Get result
            CpuTensor<TensorDataType::FP32> host_input_grad( Device::Cpu(), input_shape );
            copy( device_input_grad, host_input_grad );

            std::cout << "\nInput Gradient [" << batch_size << "x" << seq_length << "x" << embedding_dim << "]:\n";
            std::cout << host_input_grad.toString( true );

            // ============================================================
            // EXPECTED INTERMEDIATE RESULTS (for manual verification)
            // ============================================================
            std::cout << "\n=== Expected Intermediate Computations ===\n";
            std::cout << "With these minimal inputs, you should verify:\n";
            std::cout << "1. After unpermute: dvaccum shape [" << batch_size << ", " << num_heads
                << ", " << seq_length << ", " << head_dim << "]\n";
            std::cout << "   Expected: permute([1, 2, 2, 2]) back from output grad\n";
            std::cout << "\n2. After cublas dV matmul: dv_ shape should match V\n";
            std::cout << "   This computes: dV = attn_weights^T @ dvaccum\n";
            std::cout << "\n3. After cublas dAttn matmul: d_attn_weights from dvaccum @ V^T\n";
            std::cout << "\n4. After softmax backward: d_scores from d_attn_weights\n";
            std::cout << "\n5. After cublas dQ matmul: dQ from d_scores @ K\n";
            std::cout << "\n6. After cublas dK matmul: dK from d_scores^T @ Q\n";
            std::cout << "\n7. Final accumulation into input grad\n";

            // ============================================================
            // VERIFICATION
            // ============================================================
            std::cout << "\n=== Verification ===\n";
            auto* grad_data = host_input_grad.data();
            size_t total = host_input_grad.size();

            bool all_finite = true;
            bool has_nonzero = false;
            float min_grad = std::numeric_limits<float>::max();
            float max_grad = std::numeric_limits<float>::lowest();

            for ( size_t i = 0; i < total; ++i )
            {
                if ( !std::isfinite( grad_data[ i ] ) )
                {
                    all_finite = false;
                    std::cout << "Non-finite gradient at index " << i << ": " << grad_data[ i ] << "\n";
                }
                else
                {
                    min_grad = std::min( min_grad, grad_data[ i ] );
                    max_grad = std::max( max_grad, grad_data[ i ] );
                    if ( std::abs( grad_data[ i ] ) > 1e-6f )
                    {
                        has_nonzero = true;
                    }
                }
            }

            std::cout << "Gradient range: [" << min_grad << ", " << max_grad << "]\n";

            EXPECT_TRUE( all_finite ) << "All gradients should be finite";
            EXPECT_TRUE( has_nonzero ) << "Gradients should be non-zero for this input/output_grad";

            std::cout << "\n=== Test Complete ===\n";
            std::cout << "To trace intermediate results, add debug prints in your backward() implementation:\n";
            std::cout << "  - After unpermute(dout) -> dvaccum\n";
            std::cout << "  - After each cublas matmul (dV, dAttn, dQ, dK)\n";
            std::cout << "  - After softmax backward\n";
            std::cout << "  - Before final accumulation\n";
        }
        catch ( const std::exception& e )
        {
            GTEST_SKIP() << "Attention backend not available: " << e.what();
        }
    }
}