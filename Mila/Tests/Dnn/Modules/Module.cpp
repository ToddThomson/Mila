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

    class MockModuleConfig : public ConfigurationBase {
    public:
        MockModuleConfig() {
            withName( "mock_module" );
        }
    };

    template<DeviceType TDevice>
    using MemoryResourceType = std::conditional_t<TDevice == Compute::DeviceType::Cuda,
        Compute::CudaDeviceMemoryResource,
        Compute::CpuMemoryResource>;

    template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidTensorType<TInput>&& ValidFloatTensorType<TOutput>
    class MockModule : public Module<TDeviceType, TInput, TOutput> {
    public:
        using MR = typename Module<TDeviceType, TInput, TOutput>::MR;
        using ModuleBase = Module<TDeviceType, TInput, TOutput>;

        explicit MockModule( const std::string& device_name, const MockModuleConfig& config = MockModuleConfig() )
            : ModuleBase( device_name, config ) {}

        explicit MockModule( std::shared_ptr<DeviceContext> context, const MockModuleConfig& config = MockModuleConfig() )
            : ModuleBase( context, config ) {}

        void forward( const Tensor<TInput, MR>& input, Tensor<TOutput, MR>& output ) override {
            // Mock implementation: just copy input to output
            if ( input.size() != output.size() ) {
                throw std::runtime_error( "Input and output tensors must have the same size" );
            }

            if constexpr ( std::is_same_v<TInput, TOutput> ) {
                // If input and output types are the same, direct copy
                for ( size_t i = 0; i < input.size(); ++i ) {
                    output.data()[ i ] = input.data()[ i ];
                }
            }
            else {
                // Type conversion if needed
                for ( size_t i = 0; i < input.size(); ++i ) {
                    output.data()[ i ] = static_cast<TOutput>( input.data()[ i ] );
                }
            }
        }

        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TOutput, MR>& output_grad,
            Tensor<TInput, MR>& input_grad ) override {
            // Mock implementation: just copy output_grad to input_grad
            if ( output_grad.size() != input_grad.size() ) {
                throw std::runtime_error( "Output gradient and input gradient tensors must have the same size" );
            }

            if constexpr ( std::is_same_v<TInput, TOutput> ) {
                // If input and output types are the same, direct copy
                for ( size_t i = 0; i < output_grad.size(); ++i ) {
                    input_grad.data()[ i ] = output_grad.data()[ i ];
                }
            }
            else {
                // Type conversion if needed
                for ( size_t i = 0; i < output_grad.size(); ++i ) {
                    input_grad.data()[ i ] = static_cast<TInput>( output_grad.data()[ i ] );
                }
            }
        }

        size_t parameterCount() const override {
            return 0;
        }

        void save( ModelArchive& archive ) const override {}

        void load( ModelArchive& archive ) override {}

        std::string toString() const override {
            std::ostringstream oss;
            oss << "MockModule: " << this->getName() << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << "Training: " << (this->isTraining() ? "true" : "false") << std::endl;
            oss << "Precision: " << static_cast<int>(this->getPrecisionPolicy()) << std::endl;
            return oss.str();
        }

        std::string testParametersToString() const {
            return this->parametersToString();
        }

        std::string testStateToString() const {
            return this->stateToString();
        }
    };

    template<DeviceType TDevice, typename TInput = float, typename TOutput = TInput>
    struct ModuleTestData {
        std::shared_ptr<MockModule<TDevice, TInput, TOutput>> module;
        bool is_training;

        static ModuleTestData Create(
            const std::string& device_str,
            bool is_training = false )
        {
            ModuleTestData data;
            data.is_training = is_training;

            MockModuleConfig config;
            config.withName( "mock_module" )
                .withTraining( is_training );

            data.module = std::make_shared<MockModule<TDevice, TInput, TOutput>>( device_str, config );
            return data;
        }

        static ModuleTestData CreateWithContext(
            std::shared_ptr<DeviceContext> context,
            bool is_training = false )
        {
            ModuleTestData data;
            data.is_training = is_training;

            MockModuleConfig config;
            config.withName( "mock_module_context" )
                .withTraining( is_training );

            data.module = std::make_shared<MockModule<TDevice, TInput, TOutput>>( context, config );
            return data;
        }
    };

    class ModuleTests : public ::testing::Test {
    protected:
        void SetUp() override {}

        void TearDown() override {
            cpu_float_data_.module.reset();
            cuda_float_data_.module.reset();
            training_cpu_float_data_.module.reset();
            training_cuda_float_data_.module.reset();
            context_cpu_float_data_.module.reset();
            context_cuda_float_data_.module.reset();
        }

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

        ModuleTestData<Compute::DeviceType::Cpu, float> cpu_float_data_;
        ModuleTestData<Compute::DeviceType::Cuda, float> cuda_float_data_;
        ModuleTestData<Compute::DeviceType::Cpu, float> training_cpu_float_data_;
        ModuleTestData<Compute::DeviceType::Cuda, float> training_cuda_float_data_;
        ModuleTestData<Compute::DeviceType::Cpu, float> context_cpu_float_data_;
        ModuleTestData<Compute::DeviceType::Cuda, float> context_cuda_float_data_;
    };

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
        EXPECT_EQ( precision, ComputePrecision::Policy::Auto );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestHelperMethods( const ModuleTestData<TDevice, TInput, TOutput>& data ) {
        std::string params_str = data.module->testParametersToString();
        std::string state_str = data.module->testStateToString();

        EXPECT_EQ( params_str, "Parameter count: 0\n" );
        EXPECT_EQ( state_str, "" );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestForwardBackward( const ModuleTestData<TDevice, TInput, TOutput>& data ) {
        using MR = MemoryResourceType<TDevice>;

        std::vector<size_t> shape = { 2, 3 };
        Tensor<TInput, MR> input( shape );
        Tensor<TOutput, MR> output( shape );
        Tensor<TInput, MR> input_grad( shape );
        Tensor<TOutput, MR> output_grad( shape );

        for ( size_t i = 0; i < input.size(); ++i ) {
            input.data()[ i ] = static_cast<TInput>( i + 1.0f );
            output_grad.data()[ i ] = static_cast<TOutput>( 0.5f );
        }

        EXPECT_NO_THROW( data.module->forward( input, output ) );
        EXPECT_NO_THROW( data.module->backward( input, output_grad, input_grad ) );

        for ( size_t i = 0; i < input.size(); ++i ) {
            EXPECT_FLOAT_EQ( static_cast<float>( output.data()[ i ] ), static_cast<float>( input.data()[ i ] ) );
            EXPECT_FLOAT_EQ( static_cast<float>( input_grad.data()[ i ] ), static_cast<float>( output_grad.data()[ i ] ) );
        }
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestInvalidDeviceName() {
        MockModuleConfig config;
        config.withName( "test_module" );

        EXPECT_THROW( (std::make_shared<MockModule<TDevice, TInput, TOutput>>( "INVALID_DEVICE", config )), std::runtime_error );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestNullDeviceContext() {
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
        auto mismatched_device = (TDevice == DeviceType::Cuda) ? "CPU" : "CUDA:0";
        auto mismatched_context = std::make_shared<DeviceContext>( mismatched_device );

        MockModuleConfig config;
        config.withName( "test_module" );

        EXPECT_THROW(
            (std::make_shared<MockModule<TDevice, TInput, TOutput>>( mismatched_context, config )),
            std::runtime_error
        );
    }

    template<DeviceType TDevice, typename TInput, typename TOutput = TInput>
    void TestEmptyNameInConfig() {
        MockModuleConfig config;
        config.withName( "" );

        EXPECT_THROW(
            (std::make_shared<MockModule<TDevice, TInput, TOutput>>( "CPU", config )),
            std::invalid_argument
        );
    }

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

    TEST_F( ModuleTests, Cpu_ForwardBackward ) {
        TestForwardBackward<Compute::DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( ModuleTests, Cpu_Training_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cpu, float>( TrainingCpuFloatData(), true );
    }

    TEST_F( ModuleTests, Context_Cpu_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

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

    TEST_F( ModuleTests, Cuda_ForwardBackward ) {
        TestForwardBackward<Compute::DeviceType::Cuda, float>( CudaFloatData() );
    }

    TEST_F( ModuleTests, Cuda_Training_TrainingMode ) {
        TestTrainingMode<Compute::DeviceType::Cuda, float>( TrainingCudaFloatData(), true );
    }

    TEST_F( ModuleTests, Context_Cuda_DeviceType ) {
        TestDeviceType<Compute::DeviceType::Cuda, float>( ContextCudaFloatData() );
    }

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
        MockModuleConfig training_config;
        training_config.withName( "training_module" )
            .withTraining( true );

        auto module = std::make_shared<MockModule<DeviceType::Cpu, float>>( "CPU", training_config );

        EXPECT_TRUE( module->isTraining() );
    }
}