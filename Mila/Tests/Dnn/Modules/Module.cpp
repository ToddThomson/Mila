#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
// #include <miniz.h>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

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

        // Constructor with device name
        explicit MockModule( const std::string& device_name, ComputePrecision::Policy policy = ComputePrecision::Policy::Auto )
            : ModuleBase( device_name, policy ) {
            this->setName( "mock_module" );
        }

        // Constructor with device context
        explicit MockModule( std::shared_ptr<DeviceContext> context, ComputePrecision::Policy policy = ComputePrecision::Policy::Auto )
            : ModuleBase( context, policy ) {
            this->setName( "mock_module_context" );
        }

        // Implementation of abstract methods
        size_t parameterCount() const override {
            return 0;
        }

        void save( mz_zip_archive& zip ) const override {
            // Dummy implementation
        }

        void load( mz_zip_archive& zip ) override {
            // Dummy implementation
        }

        std::string toString() const override {
            std::ostringstream oss;
            oss << "MockModule: " << this->getName() << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << "Training: " << (this->isTraining() ? "true" : "false") << std::endl;
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
            data.module = std::make_shared<MockModule<TDevice, TInput, TOutput>>( device_str );

            if ( is_training ) {
                data.module->setTraining( true );
            }

            return data;
        }

        // Create a test data structure with device context
        static ModuleTestData CreateWithContext(
            std::shared_ptr<DeviceContext> context,
            bool is_training = false )
        {
            ModuleTestData data;
            data.is_training = is_training;
            data.module = std::make_shared<MockModule<TDevice, TInput, TOutput>>( context );

            if ( is_training ) {
                data.module->setTraining( true );
            }

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
    void TestSetName( const ModuleTestData<TDevice, TInput, TOutput>& data, const std::string& new_name ) {
        data.module->setName( new_name );
        EXPECT_EQ( data.module->getName(), new_name );
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
    void TestComputePrecision( const ModuleTestData<TDevice, TInput, TOutput>& data ) {
        auto precision = data.module->getComputePrecision();
        EXPECT_EQ( precision.getPolicy(), ComputePrecision::Policy::Auto ); // Default policy

        // Test changing the policy
        data.module->setComputePrecisionMode( ComputePrecision::Policy::Performance );
        EXPECT_EQ( data.module->getComputePrecision().getPolicy(), ComputePrecision::Policy::Performance );
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
        EXPECT_THROW( ( std::make_shared<MockModule<TDevice, TInput, TOutput>>( "INVALID_DEVICE" ) ), std::runtime_error );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestNullDeviceContext() {
        // Try to create a module with a null device context
        EXPECT_THROW(
            ( std::make_shared<MockModule<TDevice, TInput, TOutput>>( 
                std::shared_ptr<Mila::Dnn::Compute::DeviceContext>{nullptr} ) ),
            std::invalid_argument
        );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestMismatchedDeviceContext() {
        // Create a context with the opposite device type
        auto mismatched_device = (TDevice == DeviceType::Cuda) ? "CPU" : "CUDA:0";
        auto mismatched_context = std::make_shared<DeviceContext>( mismatched_device );

        // Try to create a module with a mismatched device context
        EXPECT_THROW(
            (std::make_shared<MockModule<TDevice, TInput, TOutput>>( mismatched_context )),
            std::runtime_error
        );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestSetInvalidName() {
        auto module = std::make_shared<MockModule<TDevice, TInput, TOutput>>(
            TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU"
        );

        // Try to set an empty name
        EXPECT_THROW( module->setName( "" ), std::invalid_argument );
    }

    // CPU Tests
    TEST_F( ModuleTests, Cpu_GetName ) {
        TestGetName<Compute::DeviceType::Cpu, float>( CpuFloatData(), "mock_module" );
    }

    TEST_F( ModuleTests, Cpu_SetName ) {
        TestSetName<Compute::DeviceType::Cpu, float>( CpuFloatData(), "new_module_name" );
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

    TEST_F( ModuleTests, Cpu_ComputePrecision ) {
        TestComputePrecision<Compute::DeviceType::Cpu, float>( CpuFloatData() );
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

    TEST_F( ModuleTests, Cuda_SetName ) {
        TestSetName<Compute::DeviceType::Cuda, float>( CudaFloatData(), "new_cuda_module_name" );
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

    TEST_F( ModuleTests, Cuda_ComputePrecision ) {
        TestComputePrecision<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    // CUDA Training Mode Tests
    TEST_F( ModuleTests, Cuda_Training_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, float>( TrainingCudaFloatData(), true );
    }

    // Context Construction Tests
    TEST_F( ModuleTests, Context_Cuda_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cuda, float>( ContextCudaFloatData() );
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

    TEST_F( ModuleTests, SetInvalidName ) {
        TestSetInvalidName<Compute::DeviceType::Cpu, float>();
    }
}