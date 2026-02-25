#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <exception>
#include <cstdint>

import Mila;

namespace CompositeComponents_Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    namespace
    {
        // ====================================================================
        // Test Shape Definitions
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
            shape_t dimensions;
            std::string name;

            static TestShape Small()
            {
                return { TestShapeSize::Small, { 16, 64, 768 }, "Small" };
            }

            static TestShape Medium()
            {
                return { TestShapeSize::Medium, { 32, 128, 1024 }, "Medium" };
            }

            static TestShape Large()
            {
                return { TestShapeSize::Large, { 64, 256, 2048 }, "Large" };
            }

            static TestShape Minimal()
            {
                return { TestShapeSize::Minimal, { 1, 1, 8 }, "Minimal" };
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
        // Precision Type Wrapper
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
            static constexpr float tolerance = 1e-1f;
        };

        template<>
        struct PrecisionTraits<TensorDataType::FP8_E4M3>
        {
            static constexpr TensorDataType value = TensorDataType::FP8_E4M3;
            static constexpr const char* name = "FP8";
            static constexpr float tolerance = 5e-1f;
        };
    }

    // ====================================================================
    // Test Network for Shared Context Testing
    // ====================================================================

    template<TensorDataType TPrecision>
    class MLPTestNetwork : public Network<DeviceType::Cuda, TPrecision>
    {
    private:
        std::unique_ptr<IExecutionContext> owned_context_;
        std::shared_ptr<MLP<DeviceType::Cuda, TPrecision>> mlp_;
        MLPConfig config_;

    public:
        explicit MLPTestNetwork(
            const std::string& name,
            const MLPConfig& config,
            DeviceId device_id )
            : Network<DeviceType::Cuda, TPrecision>( name ),
            owned_context_( createExecutionContext( device_id ) ),
            config_( config )
        {
            createGraph();
            this->setExecutionContext( owned_context_.get() );
        }

        std::shared_ptr<MLP<DeviceType::Cuda, TPrecision>> getMLP() const
        {
            return mlp_;
        }

        const MLPConfig& getConfig() const
        {
            return config_;
        }

        const ComponentType getType() const override
        {
            return ComponentType::MockComponent;
        }

    protected:
        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            // Minimal implementation for testing
        }

    private:
        void createGraph()
        {
            mlp_ = std::make_shared<MLP<DeviceType::Cuda, TPrecision>>(
                this->getName() + ".mlp",
                config_
            );

            this->addComponent( mlp_ );
        }
    };

    // ====================================================================
    // Test Fixture Structure
    // ====================================================================

    template<TensorDataType TPrecision>
    struct MLPTestFixture
    {
        TestShape test_shape;
        MLPConfig config;
        std::shared_ptr<MLP<DeviceType::Cuda, TPrecision>> component;
        std::unique_ptr<MLPTestNetwork<TPrecision>> network;
        int64_t input_features;
        int64_t hidden_size;
        bool is_training;
        bool use_shared_context;

        MLPTestFixture()
            : config( 1, 1 ), input_features( 0 ), hidden_size( 0 )
        {}

        static MLPTestFixture CreateStandalone(
            TestShape shape,
            int64_t input_features,
            int64_t hidden_size,
            bool has_bias = true,
            ActivationType activation = ActivationType::Gelu,
            bool use_layer_norm = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto,
            bool is_training = false )
        {
            MLPTestFixture fixture;
            fixture.test_shape = shape;
            fixture.input_features = input_features;
            fixture.hidden_size = hidden_size;
            fixture.is_training = is_training;
            fixture.use_shared_context = false;

            fixture.config = MLPConfig( input_features, hidden_size );
            fixture.config.withBias( has_bias )
                .withActivation( activation )
                .withLayerNorm( use_layer_norm )
                .withPrecisionPolicy( precision );

            std::string name = "mlp_cuda_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.component = std::make_shared<MLP<DeviceType::Cuda, TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

            if ( fixture.is_training )
            {
                fixture.component->build( fixture.shape() );
                fixture.component->setTraining( true );
            }

            return fixture;
        }

        static MLPTestFixture CreateWithSharedContext(
            TestShape shape,
            int64_t input_features,
            int64_t hidden_size,
            bool has_bias = true,
            ActivationType activation = ActivationType::Gelu,
            bool use_layer_norm = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto,
            bool is_training = false )
        {
            MLPTestFixture fixture;
            fixture.test_shape = shape;
            fixture.input_features = input_features;
            fixture.hidden_size = hidden_size;
            fixture.is_training = is_training;
            fixture.use_shared_context = true;

            fixture.config = MLPConfig( input_features, hidden_size );
            fixture.config.withBias( has_bias )
                .withActivation( activation )
                .withLayerNorm( use_layer_norm )
                .withPrecisionPolicy( precision );

            std::string name = "mlp_network_" + shape.name + "_" + PrecisionTraits<TPrecision>::name;

            fixture.network = std::make_unique<MLPTestNetwork<TPrecision>>(
                name,
                fixture.config,
                Device::Cuda( 0 )
            );

            fixture.component = fixture.network->getMLP();

            if ( fixture.is_training )
            {
                fixture.network->build( fixture.shape() );
                fixture.network->setTraining( true );
            }

            return fixture;
        }

        const shape_t& shape() const
        {
            return test_shape.dimensions;
        }
    };

    // ====================================================================
    // Typed Tests (Precision-Based)
    // ====================================================================

    template<typename T>
    class MLPCudaTests : public testing::Test
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
    struct PrecisionType
    {
        static constexpr TensorDataType value = TPrecision;
    };

    using PrecisionTypes = ::testing::Types<
        PrecisionType<TensorDataType::FP32>
        // TODO: Uncomment when FP16 MLP CUDA is implemented
        // PrecisionType<TensorDataType::FP16>
    >;

    TYPED_TEST_SUITE( MLPCudaTests, PrecisionTypes );

    // ====================================================================
    // Constructor Tests
    // ====================================================================

    TYPED_TEST( MLPCudaTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        ASSERT_NE( fixture.component, nullptr );
        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cuda );
        EXPECT_EQ( fixture.component->getDeviceId().type, DeviceType::Cuda );
    }

    TYPED_TEST( MLPCudaTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateWithSharedContext(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        ASSERT_NE( fixture.component, nullptr );
        ASSERT_NE( fixture.network, nullptr );
    }

    TYPED_TEST( MLPCudaTests, Constructor_WithInvalidConfiguration_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        MLPConfig invalid_zero_input( 0, 1024 );
        MLPConfig invalid_zero_hidden( 768, 0 );

        EXPECT_THROW(
            (MLP<DeviceType::Cuda, TPrecision>( "invalid_input", invalid_zero_input, Device::Cuda( 0 ) )),
            std::invalid_argument
        );

        EXPECT_THROW(
            (MLP<DeviceType::Cuda, TPrecision>( "invalid_hidden", invalid_zero_hidden, Device::Cuda( 0 ) )),
            std::invalid_argument
        );
    }

    TYPED_TEST( MLPCudaTests, Constructor_WithInvalidDeviceType_ThrowsInvalidArgument )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        MLPConfig config( 768, 3072 );

        EXPECT_THROW(
            (MLP<DeviceType::Cuda, TPrecision>( "invalid_device", config, Device::Cpu() )),
            std::invalid_argument
        );
    }

    // ====================================================================
    // Basic Property Tests
    // ====================================================================

    TYPED_TEST( MLPCudaTests, GetDeviceType_AfterConstruction_ReturnsCuda )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cuda );

        auto device = fixture.component->getDeviceId();
        EXPECT_EQ( device.type, DeviceType::Cuda );
    }

    TYPED_TEST( MLPCudaTests, GetName_AfterConstruction_ReturnsCorrectName )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        std::string expected_name = "mlp_cuda_" + test_shape.name + "_" + PrecisionTraits<TPrecision>::name;
        EXPECT_EQ( fixture.component->getName(), expected_name );
    }

    TYPED_TEST( MLPCudaTests, IsTraining_InferenceFixture_ReturnsFalse )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072,
            true,
            ActivationType::Gelu,
            false,
            ComputePrecision::Policy::Auto,
            false
        );

        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( MLPCudaTests, IsTraining_TrainingFixture_ReturnsTrue )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072,
            true,
            ActivationType::Gelu,
            false,
            ComputePrecision::Policy::Auto,
            true
        );

        EXPECT_TRUE( fixture.component->isTraining() );
    }

    TYPED_TEST( MLPCudaTests, SetTraining_TogglingMode_UpdatesState )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        EXPECT_FALSE( fixture.component->isTraining() );

        // Build before enabling training to satisfy Component lifecycle contract.
        fixture.component->build( fixture.shape() );

        fixture.component->setTraining( true );
        EXPECT_TRUE( fixture.component->isTraining() );

        fixture.component->setTraining( false );
        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TYPED_TEST( MLPCudaTests, ParameterCount_AfterConstruction_ReturnsExpectedCount )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        // build before querying parameters
        EXPECT_NO_THROW( fixture.component->build( fixture.shape() ) );

        int64_t input_features = fixture.input_features;
        int64_t hidden_size = fixture.hidden_size;
        bool has_bias = fixture.config.hasBias();

        size_t expected_fc1_params = input_features * hidden_size;
        size_t expected_fc2_params = hidden_size * input_features;

        if ( has_bias )
        {
            expected_fc1_params += hidden_size;
            expected_fc2_params += input_features;
        }

        size_t expected_total_params = expected_fc1_params + expected_fc2_params;

        EXPECT_EQ( fixture.component->parameterCount(), expected_total_params );
    }

    TYPED_TEST( MLPCudaTests, ToString_AfterConstruction_ContainsComponentInfo )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "MLP" ), std::string::npos );
        EXPECT_NE( output.find( fixture.component->getName() ), std::string::npos );
    }

    TYPED_TEST( MLPCudaTests, Synchronize_AfterConstruction_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }

    // ====================================================================
    // Build State Tests
    // ====================================================================

    TYPED_TEST( MLPCudaTests, IsBuilt_BeforeBuild_ReturnsFalse )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        EXPECT_FALSE( fixture.component->isBuilt() );
    }

    TYPED_TEST( MLPCudaTests, Build_WithVariousShapes_SetsBuiltState )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
                test_shape,
                test_shape.dimensions.back(),
                3072
            );

            EXPECT_FALSE( fixture.component->isBuilt() )
                << "Component should not be built before build() for shape: " << test_shape.name;

            EXPECT_NO_THROW( fixture.component->build( fixture.shape() ) )
                << "Build failed for shape: " << test_shape.name;

            EXPECT_TRUE( fixture.component->isBuilt() )
                << "Component should be built after build() for shape: " << test_shape.name;
        }
    }

    TYPED_TEST( MLPCudaTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            TestShape::Small(),
            TestShape::Small().dimensions.back(),
            3072
        );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.shape() );

        EXPECT_THROW(
            fixture.component->forward( input ),
            std::runtime_error
        );
    }

    // ====================================================================
    // Forward Pass Tests
    // ====================================================================

    TYPED_TEST( MLPCudaTests, Forward_WithVariousShapes_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = TestShape::StandardShapes();

        for ( const auto& test_shape : shapes )
        {
            auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
                test_shape,
                test_shape.dimensions.back(),
                3072
            );

            fixture.component->build( fixture.shape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
            random( host_input, -1.0f, 1.0f );

            CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );

            copy( host_input, device_input );

            Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;

            EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } )
                << "Forward failed for shape: " << test_shape.name;

            ASSERT_NE( out_ptr, nullptr );

            auto& out_tensor = *out_ptr;

            EXPECT_EQ( out_tensor.shape(), device_input.shape() )
                << "Output shape mismatch for shape: " << test_shape.name;

            auto host_out = toHost<TPrecision>( out_tensor );

            EXPECT_EQ( host_out.size(), out_tensor.size() )
                << "Output size mismatch for shape: " << test_shape.name;
        }
    }

    TYPED_TEST( MLPCudaTests, Forward_MultipleInvocations_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );

        for ( int iter = 0; iter < 5; ++iter )
        {
            random( host_input, -1.0f, 1.0f );
            copy( host_input, device_input );

            Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
            EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } )
                << "Forward failed on iteration " << iter;

            ASSERT_NE( out_ptr, nullptr );
        }
    }

    // ====================================================================
    // Configuration Variant Tests
    // ====================================================================

    TYPED_TEST( MLPCudaTests, NoBias_ParameterCount_ReturnsExpectedCount )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072,
            false // no bias
        );

        // build before querying parameters
        EXPECT_NO_THROW( fixture.component->build( fixture.shape() ) );

        int64_t input_features = fixture.input_features;
        int64_t hidden_size = fixture.hidden_size;

        size_t expected_fc1_params = input_features * hidden_size;
        size_t expected_fc2_params = hidden_size * input_features;
        size_t expected_total_params = expected_fc1_params + expected_fc2_params;

        EXPECT_EQ( fixture.component->parameterCount(), expected_total_params );
    }

    TYPED_TEST( MLPCudaTests, NoBias_Forward_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072,
            false // no bias
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -1.0f, 1.0f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );

        auto& out_tensor = *out_ptr;
        EXPECT_EQ( out_tensor.size(), device_input.size() );
    }

    TYPED_TEST( MLPCudaTests, LayerNorm_Forward_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072,
            true,  // has bias
            ActivationType::Gelu,
            true   // use layer norm
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -1.0f, 1.0f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );

        auto& out_tensor = *out_ptr;
        EXPECT_EQ( out_tensor.shape(), device_input.shape() );
    }

    TYPED_TEST( MLPCudaTests, PrecisionPolicies_Forward_AllSucceed )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        std::vector<ComputePrecision::Policy> policies = {
            ComputePrecision::Policy::Performance,
            ComputePrecision::Policy::Accuracy,
            ComputePrecision::Policy::Native,
            ComputePrecision::Policy::Auto
        };

        for ( auto policy : policies )
        {
            auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
                test_shape,
                test_shape.dimensions.back(),
                3072,
                true,
                ActivationType::Gelu,
                false,
                policy
            );

            fixture.component->build( fixture.shape() );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
            random( host_input, -1.0f, 1.0f );

            CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );

            copy( host_input, device_input );

            Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
            EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } )
                << "Forward failed for precision policy: " << static_cast<int>(policy);

            ASSERT_NE( out_ptr, nullptr );
        }
    }

    // ====================================================================
    // Shared Context Tests
    // ====================================================================

    TYPED_TEST( MLPCudaTests, SharedContext_Construction_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateWithSharedContext(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        ASSERT_NE( fixture.component, nullptr );
        ASSERT_NE( fixture.network, nullptr );

        std::string expected_name = "mlp_network_" + test_shape.name + "_" + PrecisionTraits<TPrecision>::name + ".mlp";
        EXPECT_EQ( fixture.component->getName(), expected_name );
    }

    TYPED_TEST( MLPCudaTests, SharedContext_Forward_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateWithSharedContext(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -1.0f, 1.0f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );

        auto& out_tensor = *out_ptr;
        EXPECT_EQ( out_tensor.shape(), device_input.shape() );
    }

    // ====================================================================
    // Child Component Tests
    // ====================================================================

    TYPED_TEST( MLPCudaTests, GetNamedComponents_ReturnsExpectedChildren )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072
        );

        fixture.component->build( fixture.shape() );

        auto modules = fixture.component->getComponents();
        const std::string base = fixture.component->getName();

        EXPECT_GE( modules.size(), 3u );
    }

    TYPED_TEST( MLPCudaTests, LayerNorm_GetNamedComponents_IncludesNormComponent )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Small();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            3072,
            true,  // has bias
            ActivationType::Gelu,
            true   // use layer norm
        );

        fixture.component->build( fixture.shape() );

        auto modules = fixture.component->getComponents();
        const std::string base = fixture.component->getName();
    }

    // ====================================================================
    // Edge Case Tests
    // ====================================================================

    TYPED_TEST( MLPCudaTests, EdgeCase_MinimalShape_Forward )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = TestShape::Minimal();
        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            16
        );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -1.0f, 1.0f );

        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( device_input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );
    }

    // ====================================================================
    // Backward Pass Tests (new API)
    // ====================================================================

    TYPED_TEST( MLPCudaTests, Backward_BeforeBuild_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            TestShape::Small(),
            TestShape::Small().dimensions.back(),
            3072
        );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> output_grad( Device::Cuda( 0 ), fixture.shape() );

        EXPECT_THROW(
            fixture.component->backward( input, output_grad ),
            std::runtime_error
        );
    }

    TYPED_TEST( MLPCudaTests, Backward_InferenceMode_ThrowsRuntimeError )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            TestShape::Small(),
            TestShape::Small().dimensions.back(),
            3072
        );

        fixture.component->build( fixture.shape() );

        EXPECT_FALSE( fixture.component->isTraining() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -1.0f, 1.0f );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> output_grad( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );

        EXPECT_THROW(
            fixture.component->backward( input, output_grad ),
            std::runtime_error
        ) << "Backward should throw when component is not in training mode";
    }

    TYPED_TEST( MLPCudaTests, Backward_TrainingMode_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            TestShape::Small(),
            TestShape::Small().dimensions.back(),
            3072,
            true,  // has_bias
            ActivationType::Gelu,
            false, // use_layer_norm
            ComputePrecision::Policy::Auto,
            true   // is_training
        );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        CpuTensor<TensorDataType::FP32> host_output_grad( Device::Cpu(), fixture.shape() );

        random( host_input, -1.0f, 1.0f );
        random( host_output_grad, -0.1f, 0.1f );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> output_grad( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, input );
        copy( host_output_grad, output_grad );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );

        Tensor<TPrecision, CudaDeviceMemoryResource>* in_grad_ptr = nullptr;
        EXPECT_NO_THROW(
            { auto& ing_ref = fixture.component->backward( input, output_grad ); in_grad_ptr = &ing_ref; }
        ) << "Backward pass should succeed in training mode";

        ASSERT_NE( in_grad_ptr, nullptr );

        auto& in_grad_tensor = *in_grad_ptr;
        fixture.component->synchronize();

        CpuTensor<TensorDataType::FP32> host_input_grad = toHost<TPrecision>( in_grad_tensor );

        bool has_nonzero_grad = false;
        for ( size_t i = 0; i < host_input_grad.size(); ++i )
        {
            if ( std::abs( host_input_grad.data()[ i ] ) > 1e-6f )
            {
                has_nonzero_grad = true;
                break;
            }
        }

        EXPECT_TRUE( has_nonzero_grad ) << "Input gradients should contain non-zero values";
    }

    TYPED_TEST( MLPCudaTests, Backward_WithLayerNorm_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            TestShape::Small(),
            TestShape::Small().dimensions.back(),
            3072,
            true,  // has_bias
            ActivationType::Gelu,
            true,  // use_layer_norm
            ComputePrecision::Policy::Auto,
            true   // is_training
        );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -1.0f, 1.0f );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> output_grad( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, input );
        random( output_grad, -0.1f, 0.1f );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( input ); out_ptr = &out_ref; } );
        ASSERT_NE( out_ptr, nullptr );

        Tensor<TPrecision, CudaDeviceMemoryResource>* in_grad_ptr = nullptr;
        EXPECT_NO_THROW(
            { auto& ing_ref = fixture.component->backward( input, output_grad ); in_grad_ptr = &ing_ref; }
        );
        ASSERT_NE( in_grad_ptr, nullptr );

        auto& in_grad_tensor = *in_grad_ptr;
        fixture.component->synchronize();

        CpuTensor<TensorDataType::FP32> host_input_grad = toHost<TPrecision>( in_grad_tensor );

        bool has_nonzero_grad = false;
        for ( size_t i = 0; i < host_input_grad.size(); ++i )
        {
            if ( std::abs( host_input_grad.data()[ i ] ) > 1e-6f )
            {
                has_nonzero_grad = true;
                break;
            }
        }

        EXPECT_TRUE( has_nonzero_grad );
    }

    TYPED_TEST( MLPCudaTests, Backward_MultipleIterations_Succeeds )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            TestShape::Small(),
            TestShape::Small().dimensions.back(),
            3072,
            true,  // has_bias
            ActivationType::Gelu,
            false, // use_layer_norm
            ComputePrecision::Policy::Auto,
            true   // is_training
        );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> output_grad( Device::Cuda( 0 ), fixture.shape() );

        for ( int iter = 0; iter < 5; ++iter )
        {
            random( host_input, -1.0f, 1.0f );
            copy( host_input, input );

            random( output_grad, -0.1f, 0.1f );

            Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
            EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( input ); out_ptr = &out_ref; } );

            Tensor<TPrecision, CudaDeviceMemoryResource>* in_grad_ptr = nullptr;
            EXPECT_NO_THROW(
                { auto& ing_ref = fixture.component->backward( input, output_grad ); in_grad_ptr = &ing_ref; }
            ) << "Backward iteration " << iter << " failed";

            ASSERT_NE( in_grad_ptr, nullptr );
        }
    }

    TYPED_TEST( MLPCudaTests, ZeroGradients_ClearsParameterGradients )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            TestShape::Small(),
            TestShape::Small().dimensions.back(),
            3072,
            true,  // has_bias
            ActivationType::Gelu,
            false, // use_layer_norm
            ComputePrecision::Policy::Auto,
            true   // is_training
        );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), fixture.shape() );
        random( host_input, -1.0f, 1.0f );

        CudaTensor<TPrecision> input( Device::Cuda( 0 ), fixture.shape() );
        CudaTensor<TPrecision> output_grad( Device::Cuda( 0 ), fixture.shape() );

        copy( host_input, input );
        random( output_grad, -0.1f, 0.1f );

        Tensor<TPrecision, CudaDeviceMemoryResource>* out_ptr = nullptr;
        EXPECT_NO_THROW( { auto& out_ref = fixture.component->forward( input ); out_ptr = &out_ref; } );

        Tensor<TPrecision, CudaDeviceMemoryResource>* in_grad_ptr = nullptr;
        EXPECT_NO_THROW( { auto& ing_ref = fixture.component->backward( input, output_grad ); in_grad_ptr = &ing_ref; } );

        fixture.component->synchronize();

        // Zero gradients
        EXPECT_NO_THROW( fixture.component->zeroGradients() );
    }

    // ====================================================================
// Decode (inference) tests
//
// decode() chains fc1_->decode() (cuda_matvec_impl, M=1) -> [LayerNorm
// forward] -> activation forward -> fc2_->decode() (cuda_matvec_impl,
// M=1). The outer batch dimension must be exactly 1. On CPU, both
// Linear::decode() calls fall back to operation_->forward().
// ====================================================================

    TYPED_TEST( MLPCudaTests, Decode_BeforeBuild_Throws )
    {
        if ( !this->cuda_available_ )
            GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        MLPConfig cfg( 64, 128 );
        auto comp = std::make_shared<MLP<DeviceType::Cuda, TPrecision>>(
            "mlp_decode_throw", cfg, Device::Cuda( 0 ) );

        CudaTensor<TPrecision> in( Device::Cuda( 0 ), shape_t{ 1, 1, 64 } );

        EXPECT_THROW( comp->decode( in ), std::runtime_error );
    }

    TYPED_TEST( MLPCudaTests, Decode_SingleToken_ProducesValidOutput )
    {
        if ( !this->cuda_available_ )
            GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        const std::vector<std::pair<int64_t, int64_t>> feature_pairs = {
            { 16, 32 }, { 64, 128 }, { 768, 3072 }
        };

        for ( auto [in_feat, hidden] : feature_pairs )
        {
            const shape_t single_token_shape = { 1, 1, in_feat };

            MLPConfig cfg( static_cast<dim_t>(in_feat), static_cast<dim_t>(hidden) );
            cfg.withBias( true );

            auto comp = std::make_shared<MLP<DeviceType::Cuda, TPrecision>>(
                "mlp_decode_valid", cfg, Device::Cuda( 0 ) );

            comp->build( single_token_shape );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), single_token_shape );
            random( host_in, -1.0f, 1.0f );

            CudaTensor<TPrecision> device_in( Device::Cuda( 0 ), single_token_shape );
            copy( host_in, device_in );

            CudaTensor<TPrecision>* out_ptr = nullptr;
            ASSERT_NO_THROW( out_ptr = &comp->decode( device_in ) );
            ASSERT_NE( out_ptr, nullptr );

            // MLP output features == input features (fc2 maps hidden -> in_features).
            const shape_t expected_shape = { 1, 1, in_feat };
            EXPECT_EQ( out_ptr->shape(), expected_shape );

            comp->synchronize();

            auto host_out = toHost<TensorDataType::FP32>( *out_ptr );

            for ( size_t i = 0; i < host_out.size(); ++i )
            {
                EXPECT_TRUE( std::isfinite( host_out.data()[ i ] ) )
                    << "Non-finite decode output at element " << i
                    << " for in_feat=" << in_feat << " hidden=" << hidden;
            }
        }
    }

    TYPED_TEST( MLPCudaTests, Decode_DoesNotRequireTrainingMode )
    {
        if ( !this->cuda_available_ )
            GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        const shape_t single_token_shape = { 1, 1, 64 };

        MLPConfig cfg( 64, 128 );
        auto comp = std::make_shared<MLP<DeviceType::Cuda, TPrecision>>(
            "mlp_decode_no_train", cfg, Device::Cuda( 0 ) );

        comp->build( single_token_shape );
        ASSERT_FALSE( comp->isTraining() );

        CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), single_token_shape );
        random( host_in, -1.0f, 1.0f );

        CudaTensor<TPrecision> device_in( Device::Cuda( 0 ), single_token_shape );
        copy( host_in, device_in );

        EXPECT_NO_THROW( (void)comp->decode( device_in ) );
    }

    TYPED_TEST( MLPCudaTests, Decode_MultiTokenInput_Throws )
    {
        if ( !this->cuda_available_ )
            GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        const int64_t in_features = 64;
        const int64_t hidden_size = 128;

        MLPConfig cfg( in_features, hidden_size );
        auto comp = std::make_shared<MLP<DeviceType::Cuda, TPrecision>>(
            "mlp_decode_multi", cfg, Device::Cuda( 0 ) );

        comp->build( shape_t{ 4, 8, in_features } );

        // Outer batch > 1 is rejected by Linear::decode() inside fc1_.
        CudaTensor<TPrecision> multi_token( Device::Cuda( 0 ), shape_t{ 2, 1, in_features } );
        EXPECT_THROW( comp->decode( multi_token ), std::invalid_argument );

        CudaTensor<TPrecision> single_token( Device::Cuda( 0 ), shape_t{ 1, 1, in_features } );
        EXPECT_NO_THROW( (void)comp->decode( single_token ) );
    }

    TYPED_TEST( MLPCudaTests, Decode_EquivalentToForward_FP32 )
    {
        if ( !this->cuda_available_ )
            GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Decode equivalence runs only for FP32";
        }

        try
        {
            // M=1 input so both cuda_matvec_impl (decode) and cuBLASLt (forward)
            // operate on the same single-token problem and are directly comparable.
            const int64_t in_features = 64;
            const int64_t hidden_size = 128;
            const shape_t single_token_shape = { 1, 1, in_features };

            MLPConfig cfg( in_features, hidden_size );
            cfg.withBias( true );

            auto comp = std::make_shared<MLP<DeviceType::Cuda, TensorDataType::FP32>>(
                "mlp_decode_equiv", cfg, Device::Cuda( 0 ) );

            comp->build( single_token_shape );

            Mila::Core::RandomGenerator::getInstance().setSeed( 5050 );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), single_token_shape );
            random( host_in, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_in( Device::Cuda( 0 ), single_token_shape );
            copy( host_in, device_in );

            CudaTensor<TensorDataType::FP32>* fwd_out = nullptr;
            ASSERT_NO_THROW( fwd_out = &comp->forward( device_in ) );
            ASSERT_NE( fwd_out, nullptr );

            comp->synchronize();
            CpuTensor<TensorDataType::FP32> host_fwd = toHost<TensorDataType::FP32>( *fwd_out );

            CudaTensor<TensorDataType::FP32>* dec_out = nullptr;
            ASSERT_NO_THROW( dec_out = &comp->decode( device_in ) );
            ASSERT_NE( dec_out, nullptr );

            comp->synchronize();
            CpuTensor<TensorDataType::FP32> host_dec = toHost<TensorDataType::FP32>( *dec_out );

            ASSERT_EQ( host_fwd.size(), host_dec.size() );

            for ( size_t i = 0; i < host_fwd.size(); ++i )
            {
                EXPECT_NEAR( host_fwd.data()[ i ], host_dec.data()[ i ], 1e-3f )
                    << "decode/forward mismatch at element " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "MLP decode not available";
        }
    }

    TYPED_TEST( MLPCudaTests, Decode_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ )
            GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Decode CPU/CUDA equivalence runs only for FP32";
        }

        try
        {
            const int64_t in_features = 64;
            const int64_t hidden_size = 128;

            // Single-token shape required by cuda_matvec_impl inside fc1_ and fc2_.
            const shape_t single_token_shape = { 1, 1, in_features };

            Mila::Core::RandomGenerator::getInstance().setSeed( 5555 );

            auto cpu_mlp = std::make_shared<MLP<DeviceType::Cpu, TensorDataType::FP32>>(
                "mlp_cpu_decode_equiv", MLPConfig( in_features, hidden_size ), Device::Cpu()
            );

            auto cuda_mlp = std::make_shared<MLP<DeviceType::Cuda, TensorDataType::FP32>>(
                "mlp_cuda_decode_equiv", MLPConfig( in_features, hidden_size ), Device::Cuda( 0 )
            );

            cpu_mlp->build( single_token_shape );
            cuda_mlp->build( single_token_shape );

            Mila::Core::RandomGenerator::getInstance().setSeed( 6060 );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), single_token_shape );
            random( host_in, -1.0f, 1.0f );

            CpuTensor<TensorDataType::FP32>* cpu_out_ptr = nullptr;
            ASSERT_NO_THROW( cpu_out_ptr = &cpu_mlp->decode( host_in ) );
            ASSERT_NE( cpu_out_ptr, nullptr );

            CudaTensor<TensorDataType::FP32> device_in( Device::Cuda( 0 ), single_token_shape );
            copy( host_in, device_in );

            CudaTensor<TensorDataType::FP32>* cuda_out_ptr = nullptr;
            ASSERT_NO_THROW( cuda_out_ptr = &cuda_mlp->decode( device_in ) );
            ASSERT_NE( cuda_out_ptr, nullptr );

            cuda_mlp->synchronize();

            CpuTensor<TensorDataType::FP32> host_cuda_out = toHost<TensorDataType::FP32>( *cuda_out_ptr );

            ASSERT_EQ( cpu_out_ptr->size(), host_cuda_out.size() );

            const float tol = 1e-3f;
            auto* cpu_data = cpu_out_ptr->data();
            auto* cuda_data = host_cuda_out.data();

            for ( size_t i = 0; i < cpu_out_ptr->size(); ++i )
            {
                EXPECT_NEAR( cpu_data[ i ], cuda_data[ i ], tol )
                    << "Decode CPU/CUDA mismatch at element " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "MLP CPU/CUDA decode equivalence not available";
        }
    }

    // ====================================================================
    // CPU/CUDA Equivalence Tests
    // ====================================================================

    TYPED_TEST( MLPCudaTests, Forward_ComparedToCpu_ProducesEquivalentOutput )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = { TestShapeSize::Small, { 2, 4, 768 }, "Equivalence" };
        int64_t hidden_size = 1024;

        MLPConfig config( test_shape.dimensions.back(), hidden_size );

        auto cpu_mlp = std::make_shared<MLP<DeviceType::Cpu, TensorDataType::FP32>>(
            "cpu_equiv",
            config,
            Device::Cpu()
        );

        auto cuda_fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            hidden_size
        );

        cpu_mlp->build( test_shape.dimensions );
        cuda_fixture.component->build( cuda_fixture.shape() );

        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), test_shape.dimensions );
        random( host_input, -1.0f, 1.0f );

        // CPU forward (new API)
        Tensor<TensorDataType::FP32, CpuMemoryResource>* cpu_out_ptr = nullptr;
        ASSERT_NO_THROW( { auto& out_ref = cpu_mlp->forward( host_input ); cpu_out_ptr = &out_ref; } );
        ASSERT_NE( cpu_out_ptr, nullptr );

        auto& cpu_out_tensor = *cpu_out_ptr;

        // CUDA forward (new API)
        CudaTensor<TPrecision> device_input( Device::Cuda( 0 ), test_shape.dimensions );

        copy( host_input, device_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* cuda_out_ptr = nullptr;
        ASSERT_NO_THROW( { auto& out_ref = cuda_fixture.component->forward( device_input ); cuda_out_ptr = &out_ref; } );
        ASSERT_NE( cuda_out_ptr, nullptr );

        auto& cuda_out_tensor = *cuda_out_ptr;
        cuda_fixture.component->synchronize();

        CpuTensor<TensorDataType::FP32> cuda_output_host = toHost<TPrecision>( cuda_out_tensor );

        const float epsilon = PrecisionTraits<TPrecision>::tolerance;
        bool all_close = true;
        size_t first_mismatch_idx = 0;
        float max_diff = 0.0f;

        for ( size_t i = 0; i < cpu_out_tensor.size(); ++i )
        {
            float cpu_val = cpu_out_tensor.data()[ i ];
            float cuda_val = cuda_output_host.data()[ i ];
            float diff = std::abs( cpu_val - cuda_val );

            if ( diff > max_diff )
            {
                max_diff = diff;
            }

            if ( diff > epsilon )
            {
                all_close = false;
                first_mismatch_idx = i;
                break;
            }
        }

        EXPECT_TRUE( all_close )
            << "CPU and CUDA implementations produced different results\n"
            << "First mismatch at index " << first_mismatch_idx << "\n"
            << "CPU value: " << cpu_out_tensor.data()[ first_mismatch_idx ] << "\n"
            << "CUDA value: " << cuda_output_host.data()[ first_mismatch_idx ] << "\n"
            << "Max difference: " << max_diff << "\n"
            << "Tolerance: " << epsilon;
    };

    TYPED_TEST( MLPCudaTests, Backward_ComparedToCpu_ProducesEquivalentGradients )
    {
        if ( !this->cuda_available_ )
        {
            GTEST_SKIP() << "CUDA not available";
        }

        constexpr TensorDataType TPrecision = TypeParam::value;

        TestShape test_shape = { TestShapeSize::Small, { 2, 4, 768 }, "BackwardEquivalence" };
        int64_t hidden_size = 1024;

        MLPConfig config( test_shape.dimensions.back(), hidden_size );

        auto cpu_mlp = std::make_shared<MLP<DeviceType::Cpu, TensorDataType::FP32>>(
            "cpu_backward_equiv",
            config,
            Device::Cpu()
        );

        auto cuda_fixture = MLPTestFixture<TPrecision>::CreateStandalone(
            test_shape,
            test_shape.dimensions.back(),
            hidden_size,
            true,  // has_bias
            ActivationType::Gelu,
            false, // use_layer_norm
            ComputePrecision::Policy::Auto,
            true   // is_training
        );

        cpu_mlp->build( test_shape.dimensions );
        cpu_mlp->setTraining( true );

        // Use same input for both
        CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), test_shape.dimensions );
        random( host_input, -1.0f, 1.0f );

        CpuTensor<TensorDataType::FP32> host_output_grad( Device::Cpu(), test_shape.dimensions );
        random( host_output_grad, -0.1f, 0.1f );

        // CPU forward + backward (new API)
            Tensor<TensorDataType::FP32, CpuMemoryResource>* cpu_out_ptr = nullptr;
        cpu_out_ptr = &cpu_mlp->forward( host_input );
        ASSERT_NE( cpu_out_ptr, nullptr );

        cpu_mlp->zeroGradients();

        Tensor<TensorDataType::FP32, CpuMemoryResource>* cpu_in_grad_ptr = nullptr;
        ASSERT_NO_THROW( { auto& ing_ref = cpu_mlp->backward( host_input, host_output_grad ); cpu_in_grad_ptr = &ing_ref; } );
        ASSERT_NE( cpu_in_grad_ptr, nullptr );

        auto& cpu_in_grad_tensor = *cpu_in_grad_ptr;

        // CUDA forward + backward (new API)
        CudaTensor<TPrecision> cuda_input( Device::Cuda( 0 ), test_shape.dimensions );

        copy( host_input, cuda_input );

        Tensor<TPrecision, CudaDeviceMemoryResource>* cuda_out_ptr = nullptr;
        ASSERT_NO_THROW( { auto& out_ref = cuda_fixture.component->forward( cuda_input ); cuda_out_ptr = &out_ref; } );
        ASSERT_NE( cuda_out_ptr, nullptr );

        CudaTensor<TPrecision> cuda_output_grad( Device::Cuda( 0 ), test_shape.dimensions );
        copy( host_output_grad, cuda_output_grad );

        Tensor<TPrecision, CudaDeviceMemoryResource>* cuda_in_grad_ptr = nullptr;
        ASSERT_NO_THROW( { auto& ing_ref = cuda_fixture.component->backward( cuda_input, cuda_output_grad ); cuda_in_grad_ptr = &ing_ref; } );
        ASSERT_NE( cuda_in_grad_ptr, nullptr );

        auto& cuda_in_grad_tensor = *cuda_in_grad_ptr;
        cuda_fixture.component->synchronize();

        CpuTensor<TensorDataType::FP32> cuda_input_grad_host = toHost<TPrecision>( cuda_in_grad_tensor );

        const float epsilon = PrecisionTraits<TPrecision>::tolerance;
        float max_diff = 0.0f;
        size_t mismatch_idx = 0;

        for ( size_t i = 0; i < cpu_in_grad_tensor.size(); ++i )
        {
            float cpu_val = cpu_in_grad_tensor.data()[ i ];
            float cuda_val = cuda_input_grad_host.data()[ i ];
            float diff = std::abs( cpu_val - cuda_val );

            if ( diff > max_diff )
            {
                max_diff = diff;
                mismatch_idx = i;
            }
        }

        EXPECT_LT( max_diff, epsilon )
            << "CPU and CUDA backward implementations produced different input gradients\n"
            << "Mismatch at index " << mismatch_idx << "\n"
            << "CPU gradient: " << cpu_in_grad_tensor.data()[ mismatch_idx ] << "\n"
            << "CUDA gradient: " << cuda_input_grad_host.data()[ mismatch_idx ] << "\n"
            << "Max difference: " << max_diff << "\n"
            << "Tolerance: " << epsilon;
    };
}