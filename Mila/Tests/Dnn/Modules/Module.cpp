#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

import Mila;

namespace Dnn::Modules::ModuleBase::Tests
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

    /**
     * @brief Mock module for testing base Module functionality
     *
     * Uses ITensor interface for forward/backward, demonstrating the new
     * polymorphic design that accepts any compatible memory resource.
     *
     * @tparam TDeviceType The compute device (CPU or CUDA)
     */
    template<DeviceType TDeviceType>
    class MockModule : public Module<TDeviceType> {
    public:
        using ModuleBase = Module<TDeviceType>;

        explicit MockModule( const std::string& device_name, const MockModuleConfig& config = MockModuleConfig() )
            : ModuleBase( device_name, config ) {
        }

        explicit MockModule( std::shared_ptr<ExecutionContext> exec_context, const MockModuleConfig& config = MockModuleConfig() )
            : ModuleBase( exec_context, config ) {
        }

        /**
         * @brief Forward pass using ITensor polymorphic interface
         *
         * Validates tensor compatibility and performs identity operation
         * (copies input to output) for testing purposes.
         */
        void forward( const ITensor& input, ITensor& output ) override {
            // Validate device compatibility
            this->validateTensorDevice( input, "input" );
            this->validateTensorDevice( output, "output" );

            // Validate data type compatibility
            if (input.getDataType() != output.getDataType()) {
                throw std::runtime_error( "Input and output data types must match" );
            }

            // Validate size compatibility
            if (input.size() != output.size()) {
                throw std::runtime_error( "Input and output tensors must have the same size" );
            }

            // Simple identity operation using memory resource
            //auto* mr = output.getMemoryResource();
            //if (mr) {
            //    const size_t element_size = TensorDataTypeTraits<input.getDataType()>::size_in_bytes;
            //    const size_t bytes = input.size() * element_size;

            //    // Use public data() for host-accessible tensors
            //    if constexpr (TDeviceType == DeviceType::Cpu) {
            //        std::memcpy(
            //            const_cast<void*>(static_cast<const void*>(output.data())),
            //            static_cast<const void*>(input.data()),
            //            bytes
            //        );
            //    }
            //}
        }

        /**
         * @brief Backward pass using ITensor polymorphic interface
         *
         * Validates tensor compatibility and copies output gradient to
         * input gradient for testing purposes.
         */
        void backward(
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) override {

            // Validate device compatibility
            this->validateTensorDevice( input, "input" );
            this->validateTensorDevice( output_grad, "output_grad" );
            this->validateTensorDevice( input_grad, "input_grad" );

            // Validate data type compatibility
            if (output_grad.getDataType() != input_grad.getDataType()) {
                throw std::runtime_error( "Gradient data types must match" );
            }

            // Validate size compatibility
            if (output_grad.size() != input_grad.size()) {
                throw std::runtime_error( "Gradient tensors must have the same size" );
            }

            // Simple gradient pass-through using memory resource
            //auto* mr = input_grad.getMemoryResource();
            //if (mr) {
            //    const size_t element_size = TensorDataTypeTraits<output_grad.getDataType()>::size_in_bytes;
            //    const size_t bytes = output_grad.size() * element_size;

            //    // Use public data() for host-accessible tensors
            //    if constexpr (TDeviceType == DeviceType::Cpu) {
            //        std::memcpy(
            //            const_cast<void*>(static_cast<const void*>(input_grad.data())),
            //            static_cast<const void*>(output_grad.data()),
            //            bytes
            //        );
            //    }
            //}
        }

        size_t parameterCount() const override {
            return 0;
        }

        void save( ModelArchive& archive ) const override {}

        void load( ModelArchive& archive ) override {}

        std::string toString() const override {
            std::ostringstream oss;
            oss << "MockModule: " << this->getName() << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceType() ) << std::endl;
            oss << "Training: " << (this->isTraining() ? "true" : "false") << std::endl;
            oss << "Precision: " << static_cast<int>(this->getPrecisionPolicy()) << std::endl;
            return oss.str();
        }

        // Test helpers to access protected methods
        std::string testParametersToString() const {
            return this->parametersToString();
        }

        std::string testStateToString() const {
            return this->stateToString();
        }
    };

    /**
     * @brief Test data structure holding module and test configuration
     */
    template<DeviceType TDeviceType>
    struct ModuleTestData {
        std::shared_ptr<MockModule<TDeviceType>> module;
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

            data.module = std::make_shared<MockModule<TDeviceType>>( device_str, config );
            return data;
        }

        static ModuleTestData CreateWithContext(
            std::shared_ptr<ExecutionContext> exec_context,
            bool is_training = false )
        {
            ModuleTestData data;
            data.is_training = is_training;

            MockModuleConfig config;
            config.withName( "mock_module_context" )
                .withTraining( is_training );

            data.module = std::make_shared<MockModule<TDeviceType>>( exec_context, config );
            return data;
        }
    };

    class ModuleTests : public ::testing::Test {
    protected:
        void SetUp() override {}

        void TearDown() override {
            cpu_data_.module.reset();
            cuda_data_.module.reset();
            training_cpu_data_.module.reset();
            training_cuda_data_.module.reset();
            context_cpu_data_.module.reset();
            context_cuda_data_.module.reset();
        }

        ModuleTestData<DeviceType::Cpu>& CpuData() {
            if (!cpu_data_.module) {
                cpu_data_ = ModuleTestData<DeviceType::Cpu>::Create( "CPU" );
            }
            return cpu_data_;
        }

        ModuleTestData<DeviceType::Cuda>& CudaData() {
            if (!cuda_data_.module) {
                cuda_data_ = ModuleTestData<DeviceType::Cuda>::Create( "CUDA:0" );
            }
            return cuda_data_;
        }

        ModuleTestData<DeviceType::Cpu>& TrainingCpuData() {
            if (!training_cpu_data_.module) {
                training_cpu_data_ = ModuleTestData<DeviceType::Cpu>::Create( "CPU", true );
            }
            return training_cpu_data_;
        }

        ModuleTestData<DeviceType::Cuda>& TrainingCudaData() {
            if (!training_cuda_data_.module) {
                training_cuda_data_ = ModuleTestData<DeviceType::Cuda>::Create( "CUDA:0", true );
            }
            return training_cuda_data_;
        }

        ModuleTestData<DeviceType::Cpu>& ContextCpuData() {
            if (!context_cpu_data_.module) {
                auto device_ctx = DeviceContext::create( "CPU" );
                auto exec_context = ExecutionContext::create( device_ctx );
                context_cpu_data_ = ModuleTestData<DeviceType::Cpu>::CreateWithContext( exec_context );
            }
            return context_cpu_data_;
        }

        ModuleTestData<DeviceType::Cuda>& ContextCudaData() {
            if (!context_cuda_data_.module) {
                auto device_ctx = DeviceContext::create( "CUDA:0" );
                auto exec_context = ExecutionContext::create( device_ctx );
                context_cuda_data_ = ModuleTestData<DeviceType::Cuda>::CreateWithContext( exec_context );
            }
            return context_cuda_data_;
        }

        ModuleTestData<DeviceType::Cpu> cpu_data_;
        ModuleTestData<DeviceType::Cuda> cuda_data_;
        ModuleTestData<DeviceType::Cpu> training_cpu_data_;
        ModuleTestData<DeviceType::Cuda> training_cuda_data_;
        ModuleTestData<DeviceType::Cpu> context_cpu_data_;
        ModuleTestData<DeviceType::Cuda> context_cuda_data_;
    };

    // ====================================================================
    // Test Helper Functions
    // ====================================================================

    template<DeviceType TDeviceType>
    void TestGetName( const ModuleTestData<TDeviceType>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.module->getName(), expected_name );
    }

    template<DeviceType TDeviceType>
    void TestParameterCount( const ModuleTestData<TDeviceType>& data, size_t expected_count ) {
        EXPECT_EQ( data.module->parameterCount(), expected_count );
    }

    template<DeviceType TDeviceType>
    void TestPrint( const ModuleTestData<TDeviceType>& data, const std::string& expected_substring ) {
        std::string output = data.module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDeviceType>
    void TestTrainingMode( const ModuleTestData<TDeviceType>& data, bool expected_mode ) {
        EXPECT_EQ( data.module->isTraining(), expected_mode );
    }

    template<DeviceType TDeviceType>
    void TestDeviceType( const ModuleTestData<TDeviceType>& data ) {
        auto exec_context = data.module->getExecutionContext();
        EXPECT_NE( exec_context, nullptr );

        auto device_context = exec_context->getDeviceContext();
        EXPECT_NE( device_context, nullptr );

        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDeviceType );
        EXPECT_EQ( data.module->getDeviceType(), TDeviceType );
    }

    template<DeviceType TDeviceType>
    void TestPrecision( const ModuleTestData<TDeviceType>& data ) {
        auto precision = data.module->getPrecisionPolicy();
        EXPECT_EQ( precision, ComputePrecision::Policy::Auto );
    }

    template<DeviceType TDeviceType>
    void TestHelperMethods( const ModuleTestData<TDeviceType>& data ) {
        std::string params_str = data.module->testParametersToString();
        std::string state_str = data.module->testStateToString();

        EXPECT_EQ( params_str, "Parameters:\nTotal parameter count: 0\n" );
        EXPECT_EQ( state_str, "" );
    }

    /**
     * @brief Tests forward and backward passes with ITensor interface
     *
     * Demonstrates that modules can accept tensors with any compatible
     * memory resource for the same device type.
     */
    template<DeviceType TDeviceType>
    void TestForwardBackward( const ModuleTestData<TDeviceType>& data ) {
        // Determine appropriate memory resource for device
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda,
            CudaDeviceMemoryResource,
            CpuMemoryResource>;

        std::vector<size_t> shape = { 2, 3 };

        // Get device context from module's execution context
        auto exec_ctx = data.module->getExecutionContext();
        auto device_ctx = exec_ctx->getDeviceContext();

        // Create tensors
        Tensor<TensorDataType::FP32, MR> input( device_ctx, shape );
        Tensor<TensorDataType::FP32, MR> output( device_ctx, shape );
        Tensor<TensorDataType::FP32, MR> input_grad( device_ctx, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( device_ctx, shape );

        // Initialize input and output_grad
        if constexpr (TDeviceType == DeviceType::Cpu) {
            auto* input_data = input.data();
            auto* grad_data = output_grad.data();

            for (size_t i = 0; i < input.size(); ++i) {
                input_data[i] = static_cast<float>( i + 1.0f );
                grad_data[i] = 0.5f;
            }
        }
        else {
            // For CUDA, initialize via host tensor and transfer
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( "CPU", shape );
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_grad( "CPU", shape );

            auto* host_input_data = host_input.data();
            auto* host_grad_data = host_grad.data();

            for (size_t i = 0; i < host_input.size(); ++i) {
                host_input_data[i] = static_cast<float>( i + 1.0f );
                host_grad_data[i] = 0.5f;
            }

            // Transfer to device using TensorOps
            /*TensorOps<CudaComputeDeviceTag>::copy( host_input, input, exec_ctx );
            TensorOps<CudaComputeDeviceTag>::copy( host_grad, output_grad, exec_ctx );*/
            exec_ctx->synchronize();
        }

        // Test forward pass via ITensor interface
        EXPECT_NO_THROW( data.module->forward( input, output ) );

        // Test backward pass via ITensor interface
        EXPECT_NO_THROW( data.module->backward( input, output_grad, input_grad ) );

        // Verify results
        if constexpr (TDeviceType == DeviceType::Cpu) {
            auto* output_data = output.data();
            auto* input_grad_data = input_grad.data();

            for (size_t i = 0; i < input.size(); ++i) {
                EXPECT_FLOAT_EQ( output_data[i], static_cast<float>( i + 1.0f ) );
                EXPECT_FLOAT_EQ( input_grad_data[i], 0.5f );
            }
        }
        else {
            // For CUDA, transfer back to host and verify
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_output( "CPU", shape );
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_input_grad( "CPU", shape );

            /*TensorOps<CudaComputeDeviceTag>::copy( output, host_output, exec_ctx );
            TensorOps<CudaComputeDeviceTag>::copy( input_grad, host_input_grad, exec_ctx );*/
            exec_ctx->synchronize();

            auto* host_output_data = host_output.data();
            auto* host_input_grad_data = host_input_grad.data();

            for (size_t i = 0; i < host_output.size(); ++i) {
                EXPECT_FLOAT_EQ( host_output_data[i], static_cast<float>( i + 1.0f ) );
                EXPECT_FLOAT_EQ( host_input_grad_data[i], 0.5f );
            }
        }
    }

    template<DeviceType TDeviceType>
    void TestInvalidDeviceName() {
        MockModuleConfig config;
        config.withName( "test_module" );

        EXPECT_THROW(
            (std::make_shared<MockModule<TDeviceType>>( "INVALID_DEVICE", config )),
            std::runtime_error
        );
    }

    template<DeviceType TDeviceType>
    void TestNullExecutionContext() {
        MockModuleConfig config;
        config.withName( "test_module" );

        std::shared_ptr<ExecutionContext> null_context;

        EXPECT_THROW(
            (std::make_shared<MockModule<TDeviceType>>( null_context, config )),
            std::invalid_argument
        );
    }

    template<DeviceType TDeviceType>
    void TestEmptyNameInConfig() {
        MockModuleConfig config;
        config.withName( "" );

        EXPECT_THROW(
            (std::make_shared<MockModule<TDeviceType>>( "CPU", config )),
            std::invalid_argument
        );
    }

    // ====================================================================
    // CPU Tests
    // ====================================================================

    TEST_F( ModuleTests, Cpu_GetName ) {
        TestGetName( CpuData(), "mock_module" );
    }

    TEST_F( ModuleTests, Cpu_ParameterCount ) {
        TestParameterCount( CpuData(), 0 );
    }

    TEST_F( ModuleTests, Cpu_Print ) {
        TestPrint( CpuData(), "MockModule: mock_module" );
    }

    TEST_F( ModuleTests, Cpu_TrainingMode ) {
        TestTrainingMode( CpuData(), false );
    }

    TEST_F( ModuleTests, Cpu_DeviceType ) {
        TestDeviceType( CpuData() );
    }

    TEST_F( ModuleTests, Cpu_Precision ) {
        TestPrecision( CpuData() );
    }

    TEST_F( ModuleTests, Cpu_HelperMethods ) {
        TestHelperMethods( CpuData() );
    }

    TEST_F( ModuleTests, Cpu_ForwardBackward ) {
        TestForwardBackward( CpuData() );
    }

    TEST_F( ModuleTests, Cpu_Training_TrainingMode ) {
        TestTrainingMode( TrainingCpuData(), true );
    }

    TEST_F( ModuleTests, Context_Cpu_DeviceType ) {
        TestDeviceType( ContextCpuData() );
    }

    // ====================================================================
    // CUDA Tests
    // ====================================================================

    TEST_F( ModuleTests, Cuda_GetName ) {
        TestGetName( CudaData(), "mock_module" );
    }

    TEST_F( ModuleTests, Cuda_ParameterCount ) {
        TestParameterCount( CudaData(), 0 );
    }

    TEST_F( ModuleTests, Cuda_Print ) {
        TestPrint( CudaData(), "MockModule: mock_module" );
    }

    TEST_F( ModuleTests, Cuda_TrainingMode ) {
        TestTrainingMode( CudaData(), false );
    }

    TEST_F( ModuleTests, Cuda_DeviceType ) {
        TestDeviceType( CudaData() );
    }

    TEST_F( ModuleTests, Cuda_Precision ) {
        TestPrecision( CudaData() );
    }

    TEST_F( ModuleTests, Cuda_ForwardBackward ) {
        TestForwardBackward( CudaData() );
    }

    TEST_F( ModuleTests, Cuda_Training_TrainingMode ) {
        TestTrainingMode( TrainingCudaData(), true );
    }

    TEST_F( ModuleTests, Context_Cuda_DeviceType ) {
        TestDeviceType( ContextCudaData() );
    }

    // ====================================================================
    // Configuration Tests
    // ====================================================================

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

    // ====================================================================
    // Error Handling Tests
    // ====================================================================

    TEST_F( ModuleTests, InvalidDeviceName_Cpu ) {
        TestInvalidDeviceName<DeviceType::Cpu>();
    }

    TEST_F( ModuleTests, InvalidDeviceName_Cuda ) {
        TestInvalidDeviceName<DeviceType::Cuda>();
    }

    TEST_F( ModuleTests, NullExecutionContext_Cpu ) {
        TestNullExecutionContext<DeviceType::Cpu>();
    }

    TEST_F( ModuleTests, NullExecutionContext_Cuda ) {
        TestNullExecutionContext<DeviceType::Cuda>();
    }

    TEST_F( ModuleTests, EmptyNameInConfig_Cpu ) {
        TestEmptyNameInConfig<DeviceType::Cpu>();
    }

    TEST_F( ModuleTests, EmptyNameInConfig_Cuda ) {
        TestEmptyNameInConfig<DeviceType::Cuda>();
    }

    TEST_F( ModuleTests, ModuleReflectsConfigTrainingMode ) {
        MockModuleConfig training_config;
        training_config.withName( "training_module" )
            .withTraining( true );

        auto module = std::make_shared<MockModule<DeviceType::Cpu>>( "CPU", training_config );

        EXPECT_TRUE( module->isTraining() );
    }
}