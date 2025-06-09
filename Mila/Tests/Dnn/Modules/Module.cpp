#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

import Mila;

namespace Modules::Base::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
	using namespace Mila::Dnn::Serialization;

    // Test configuration class for modules
    class MockModuleConfig : public ConfigurationBase {
    public:
        MockModuleConfig() {
            // Default to "unnamed" name from base class
            withName( "mock_module" ); // Initialize with default name
        }
    };

    // Memory resource selector based on device type
    template<DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == Compute::DeviceType::Cuda,
        Compute::CudaMemoryResource,
        Compute::CpuMemoryResource>;

    // Mock implementation of the Module class for testing
    template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidTensorType<TInput>&& ValidFloatTensorType<TOutput>
    class MockModule : public Module<TDeviceType, TInput, TOutput> {
    public:
        using MR = typename Module<TDeviceType, TInput, TOutput>::MR;
        using ModuleBase = Module<TDeviceType, TInput, TOutput>;

        // Constructor with device name and config
        explicit MockModule( const std::string& device_name, const MockModuleConfig& config = MockModuleConfig() )
            : ModuleBase( device_name, config ) {
            // Name is already set in the config
        }

        // Constructor with context and config
        explicit MockModule( std::shared_ptr<DeviceContext> context, const MockModuleConfig& config = MockModuleConfig() )
            : ModuleBase( context, config ) {
            // Name is already set in the config
        }

        // Implementation of abstract methods
        size_t parameterCount() const override {
            return 0;
        }

        void save( ModelArchive& archive ) const override {
            // Dummy implementation
        }

        void load( ModelArchive& archive ) override {
            // Dummy implementation
        }

        std::string toString() const override {
            std::ostringstream oss;
            oss << "MockModule: " << this->getName() << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << "Training: " << (this->isTraining() ? "true" : "false") << std::endl;
            oss << "Precision: " << static_cast<int>(this->getPrecisionPolicy()) << std::endl;
            return oss.str();
        }

        // Add a public method to access protected methods for testing
        std::string testParametersToString() const {
            return this->parametersToString();
        }

        std::string testStateToString() const {
            return this->stateToString();
        }
    };

    // Test data structure for Module tests
    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput>
    struct ModuleTestData {
        std::shared_ptr<MockModule<TDevice, TInput, TOutput>> module;
        bool is_training;

        // Create a test data structure with device name
        static ModuleTestData Create(
            const std::string& device_str,
            bool is_training = false )
        {
            ModuleTestData data;
            data.is_training = is_training;

            // Create config with proper settings
            MockModuleConfig config;
            config.withName( "mock_module" )
                .withTraining( is_training );

            data.module = std::make_shared<MockModule<TDevice, TInput, TOutput>>( device_str, config );
            return data;
        }

        // Create a test data structure with device context
        static ModuleTestData CreateWithContext(
            std::shared_ptr<DeviceContext> context,
            bool is_training = false )
        {
            ModuleTestData data;
            data.is_training = is_training;

            // Create config with proper settings
            MockModuleConfig config;
            config.withName( "mock_module_context" )
                .withTraining( is_training );

            data.module = std::make_shared<MockModule<TDevice, TInput, TOutput>>( context, config );
            return data;
        }
    };

    class ModuleTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Test parameters will be initialized on demand
        }

        void TearDown() override {
            // Clean up resources
            cpu_float_data_.module.reset();
            cuda_float_data_.module.reset();
            training_cpu_float_data_.module.reset();
            training_cuda_float_data_.module.reset();
            context_cpu_float_data_.module.reset();
            context_cuda_float_data_.module.reset();
        }

        // Factory methods to lazily create test data as needed
        ModuleTestData<Compute::DeviceType::Cpu, float>& CpuFloatData() {
            if ( !cpu_float_data_.module ) {
                cpu_float_data_ = ModuleTestData<Compute::DeviceType::Cpu, float>::Create( "CPU" );
            }
            return cpu_float_data_;
        }

        ModuleTestData<Compute::DeviceType::Cuda, float>& CudaFloatData() {
            if ( !cuda_float_data_.module ) {
                cuda_float_data_ = ModuleTestData<Compute::DeviceType::Cuda, float>::Create( "CUDA:0" );
            }
            return cuda_float_data_;
        }

        ModuleTestData<Compute::DeviceType::Cpu, float>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.module ) {
                training_cpu_float_data_ = ModuleTestData<Compute::DeviceType::Cpu, float>::Create( "CPU", true );
            }
            return training_cpu_float_data_;
        }

        ModuleTestData<Compute::DeviceType::Cuda, float>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.module ) {
                training_cuda_float_data_ = ModuleTestData<Compute::DeviceType::Cuda, float>::Create( "CUDA:0", true );
            }
            return training_cuda_float_data_;
        }

        ModuleTestData<Compute::DeviceType::Cpu, float>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = ModuleTestData<Compute::DeviceType::Cpu, float>::CreateWithContext( cpu_context );
            }
            return context_cpu_float_data_;
        }

        ModuleTestData<Compute::DeviceType::Cuda, float>& ContextCudaFloatData() {
            if ( !context_cuda_float_data_.module ) {
                auto cuda_context = std::make_shared<DeviceContext>( "CUDA:0" );
                context_cuda_float_data_ = ModuleTestData<Compute::DeviceType::Cuda, float>::CreateWithContext( cuda_context );
            }
            return context_cuda_float_data_;
        }

        // Test data objects - initialized on demand
        ModuleTestData<Compute::DeviceType::Cpu, float> cpu_float_data_;
        ModuleTestData<Compute::DeviceType::Cuda, float> cuda_float_data_;
        ModuleTestData<Compute::DeviceType::Cpu, float> training_cpu_float_data_;
        ModuleTestData<Compute::DeviceType::Cuda, float> training_cuda_float_data_;
        ModuleTestData<Compute::DeviceType::Cpu, float> context_cpu_float_data_;
        ModuleTestData<Compute::DeviceType::Cuda, float> context_cuda_float_data_;
    };

    // Common test function templates
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestGetName( const ModuleTestData<TDevice, TInput, TOutput>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.module->getName(), expected_name );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestParameterCount( const ModuleTestData<TDevice, TInput, TOutput>& data, size_t expected_count ) {
        EXPECT_EQ( data.module->parameterCount(), expected_count );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestPrint( const ModuleTestData<TDevice, TInput, TOutput>& data, const std::string& expected_substring ) {
        std::string output = data.module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestTrainingMode( const ModuleTestData<TDevice, TInput, TOutput>& data, bool expected_mode ) {
        EXPECT_EQ( data.module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestDeviceType( const ModuleTestData<TDevice, TInput, TOutput>& data ) {
        auto device_context = data.module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestPrecision( const ModuleTestData<TDevice, TInput, TOutput>& data ) {
        auto precision = data.module->getPrecisionPolicy();
        EXPECT_EQ( precision, ComputePrecision::Policy::Auto ); // Default policy

        // Note: The Module now gets precision from its config, so we can't directly test setting it
        // except by creating a new module with different config
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestHelperMethods( const ModuleTestData<TDevice, TInput, TOutput>& data ) {
        // Test parametersToString and stateToString helper methods
        std::string params_str = data.module->testParametersToString();
        std::string state_str = data.module->testStateToString();

        // Both should be empty in our mock module
        EXPECT_EQ( params_str, "Parameter count: 0\n" );
        EXPECT_EQ( state_str, "" );
    }

    // Test for constructor exception handling
    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestInvalidDeviceName() {
        // Try to create a module with an invalid device name
        MockModuleConfig config;
        config.withName( "test_module" );

        EXPECT_THROW( (std::make_shared<MockModule<TDevice, TInput, TOutput>>( "INVALID_DEVICE", config )), std::runtime_error );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestNullDeviceContext() {
        // Try to create a module with a null device context
        MockModuleConfig config;
        config.withName( "test_module" );

        EXPECT_THROW(
            (std::make_shared<MockModule<TDevice, TInput, TOutput>>(
                std::shared_ptr<Mila::Dnn::Compute::DeviceContext>{nullptr}, config )),
            std::invalid_argument
        );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestMismatchedDeviceContext() {
        // Create a context with the opposite device type
        auto mismatched_device = (TDevice == DeviceType::Cuda) ? "CPU" : "CUDA:0";
        auto mismatched_context = std::make_shared<DeviceContext>( mismatched_device );

        MockModuleConfig config;
        config.withName( "test_module" );

        // Try to create a module with a mismatched device context
        EXPECT_THROW(
            (std::make_shared<MockModule<TDevice, TInput, TOutput>>( mismatched_context, config )),
            std::runtime_error
        );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestEmptyNameInConfig() {
        // Create config with empty name which should fail validation
        MockModuleConfig config;
        config.withName( "" );

        // Module constructor should throw during validation
        EXPECT_THROW(
            (std::make_shared<MockModule<TDevice, TInput, TOutput>>( "CPU", config )),
            std::invalid_argument
        );
    }

    // CPU Tests
    TEST_F( ModuleTests, Cpu_GetName ) {
        TestGetName<Compute::DeviceType::Cpu, float>( CpuFloatData(), "mock_module" );
    }

    TEST_F( ModuleTests, Cpu_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cpu, float>( CpuFloatData(), 0 );
    }

    TEST_F( ModuleTests, Cpu_Print ) {
        TestPrint<Compute::DeviceType::Cpu, float>( CpuFloatData(), "MockModule: mock_module" );
    }

    TEST_F( ModuleTests, Cpu_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cpu, float>( CpuFloatData(), false );
    }

    TEST_F( ModuleTests, Cpu_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( ModuleTests, Cpu_Precision ) {
        TestPrecision<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( ModuleTests, Cpu_HelperMethods ) {
        TestHelperMethods<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    // CPU Training Mode Tests
    TEST_F( ModuleTests, Cpu_Training_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cpu, float>( TrainingCpuFloatData(), true );
    }

    // Context Construction Tests
    TEST_F( ModuleTests, Context_Cpu_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    // CUDA Tests
    TEST_F( ModuleTests, Cuda_GetName ) {
        TestGetName<Compute::DeviceType::Cuda, float>( CudaFloatData(), "mock_module" );
    }

    TEST_F( ModuleTests, Cuda_ParameterCount ) {
        TestParameterCount<Compute::DeviceType::Cuda, float>( CudaFloatData(), 0 );
    }

    TEST_F( ModuleTests, Cuda_Print ) {
        TestPrint<Compute::DeviceType::Cuda, float>( CudaFloatData(), "MockModule: mock_module" );
    }

    TEST_F( ModuleTests, Cuda_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, float>( CudaFloatData(), false );
    }

    TEST_F( ModuleTests, Cuda_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( ModuleTests, Cuda_Precision ) {
        TestPrecision<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    // CUDA Training Mode Tests
    TEST_F( ModuleTests, Cuda_Training_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, float>( TrainingCudaFloatData(), true );
    }

    // Context Construction Tests
    TEST_F( ModuleTests, Context_Cuda_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cuda, float>( ContextCudaFloatData() );
    }

    // Config Tests
    TEST_F( ModuleTests, ModuleConfig_DefaultValues ) {
        MockModuleConfig config;
        EXPECT_EQ( config.getName(), "mock_module" );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Auto );
        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( ModuleTests, ModuleConfig_CustomValues ) {
        MockModuleConfig config;
        config.withName( "custom_module" )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance )
            .withTraining( true );

        EXPECT_EQ( config.getName(), "custom_module" );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
        EXPECT_TRUE( config.isTraining() );
    }

    // Error handling tests
    TEST_F( ModuleTests, InvalidDeviceName ) {
        TestInvalidDeviceName<Compute::DeviceType::Cpu, float>();
    }

    TEST_F( ModuleTests, NullDeviceContext ) {
        TestNullDeviceContext<Compute::DeviceType::Cpu, float>();
    }

    TEST_F( ModuleTests, MismatchedDeviceContext_Cpu ) {
        TestMismatchedDeviceContext<Compute::DeviceType::Cpu, float>();
    }

    TEST_F( ModuleTests, MismatchedDeviceContext_Cuda ) {
        TestMismatchedDeviceContext<Compute::DeviceType::Cuda, float>();
    }

    TEST_F( ModuleTests, EmptyNameInConfig ) {
        TestEmptyNameInConfig<Compute::DeviceType::Cpu, float>();
    }

    TEST_F( ModuleTests, ModuleReflectsConfigTrainingMode ) {
        // Create a config with training enabled
        MockModuleConfig training_config;
        training_config.withName( "training_module" )
            .withTraining( true );

        // Create module with this config
        auto module = std::make_shared<MockModule<DeviceType::Cpu, float>>( "CPU", training_config );

        // Verify the module has training mode set according to the config
        EXPECT_TRUE( module->isTraining() );
    }
}