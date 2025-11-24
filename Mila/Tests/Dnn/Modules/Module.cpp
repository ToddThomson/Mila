#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

import Mila;

namespace Dnn::Modules::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    class MockModuleConfig : public ModuleConfig
    {
    public:
        MockModuleConfig()
        {
            this->withName( "mock_module" );
        }

        void validate() const override
        {
            // Basic validation: name must not be empty
            if (getName().empty())
            {
                throw std::invalid_argument( "MockModuleConfig: name must not be empty" );
            }
		}

        std::string toString() const
        {
            std::ostringstream oss;
            oss << "MockModuleConfig(name=" << getName()
                << ", precision_policy=" << static_cast<int>(getPrecisionPolicy())
                << ")";
            return oss.str();
        }
    };

    /**
     * @brief Mock module for testing base Module functionality
     *
     * Implements the slimmer Module<TDeviceType, TPrecision> interface introduced in Module.ixx.
     * The mock owns an ExecutionContext for tests and exposes minimal helpers used
     * by the unit tests (execution context access, simple string helpers).
     *
     * @tparam TDeviceType The compute device (CPU or CUDA)
     * @tparam TPrecision  The tensor data precision (e.g. FP32)
     */
    template<DeviceType TDeviceType, TensorDataType TPrecision>
    class MockModule : public Module<TDeviceType, TPrecision>
    {
    public:
        using ModuleBase = Module<TDeviceType, TPrecision>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;

        explicit MockModule( const MockModuleConfig& config, std::shared_ptr<ExecutionContextType> exec_context )
            : config_( config ), exec_context_( exec_context )
        {
            // Basic validation to keep behavior aligned with tests
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null" );
            }

            // Validate config (throws if name is empty)
            config_.validate();
        }

        // Expose execution context for tests
        std::shared_ptr<ExecutionContextType> getExecutionContext() const
        {
            return exec_context_;
        }

        // ====================================================================
        // Module interface implementation
        // ====================================================================

        void forward( const ITensor& input, ITensor& output )
        {
            validateTensorDevice( input, "input" );
            validateTensorDevice( output, "output" );

            if (input.getDataType() != output.getDataType())
            {
                throw std::runtime_error( "Input and output data types must match" );
            }

            if (input.size() != output.size())
            {
                throw std::runtime_error( "Input and output tensors must have the same size" );
            }

            // Minimal semantics for mock: no actual data copy required for these tests
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            validateTensorDevice( input, "input" );
            validateTensorDevice( output_grad, "output_grad" );
            validateTensorDevice( input_grad, "input_grad" );

            if (output_grad.getDataType() != input_grad.getDataType())
            {
                throw std::runtime_error( "Gradient data types must match" );
            }

            if (output_grad.size() != input_grad.size())
            {
                throw std::runtime_error( "Gradient tensors must have the same size" );
            }

            // Minimal semantics for mock
        }

        void synchronize() override
        {
            exec_context_->synchronize();
        }

        size_t parameterCount() const override
        {
            return 0;
        }

        std::vector<ITensor*> getParameters() const override
        {
            return {};
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "MockModule: " << getName() << std::endl;
            oss << "Training: " << (this->isTraining() ? "true" : "false") << std::endl;
            return oss.str();
        }

        // ====================================================================
        // Build / lifecycle
        // ====================================================================

        void build( const shape_t& /*input_shape*/ ) override
        {
            if (is_built_)
            {
                return;
            }

            // Mock has no parameters to allocate but implements the contract.
            is_built_ = true;
        }

        bool isBuilt() const override
        {
            return is_built_;
        }

        // ====================================================================
        // State/config helpers required by tests
        // ====================================================================

        std::string getName() const override
        {
            return config_.getName();
        }

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return exec_context_->getDevice();
        }

        // Expose precision policy for tests
        ComputePrecision::Policy getPrecisionPolicy() const
        {
            return config_.getPrecisionPolicy();
        }

        // Test helpers to mimic original test expectations
        std::string testParametersToString() const
        {
            return parametersToString();
        }

        std::string testStateToString() const
        {
            return stateToString();
        }

    private:
        // Simple device validation helper used by the mock
        void validateTensorDevice( const ITensor& tensor, const char* name ) const
        {
            if (tensor.getDeviceType() != TDeviceType)
            {
                throw std::invalid_argument( std::string( name ) + ": tensor device type mismatch" );
            }
        }

        std::string parametersToString() const
        {
            std::ostringstream oss;
            oss << "Parameters:\n";
            oss << "Total parameter count: " << parameterCount() << std::endl;
            return oss.str();
        }

        std::string stateToString() const
        {
            // Empty state for this simple mock
            return std::string();
        }

        MockModuleConfig config_;
        std::shared_ptr<ExecutionContextType> exec_context_;
        bool is_built_{ false };
    };

    /**
     * @brief Test data structure holding module and test configuration
     *
     * TPrecision defaults to FP32 so existing tests remain concise while
     * exercising the new Module<TDevice, TPrecision> API.
     */
    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    struct ModuleTestData
    {
        std::shared_ptr<MockModule<TDeviceType, TPrecision>> module;
        bool is_training;

        // Create by device id (now create ExecutionContext and forward to module ctor)
        static ModuleTestData Create(
            int device_id,
            bool is_training = false )
        {
            ModuleTestData data;
            data.is_training = is_training;

            MockModuleConfig config;
            config.withName( "mock_module" );

            // Build an execution context appropriate for the device type
            std::shared_ptr<typename MockModule<TDeviceType, TPrecision>::ExecutionContextType> exec_ctx;
            if constexpr (TDeviceType == DeviceType::Cuda)
            {
                exec_ctx = std::make_shared<typename MockModule<TDeviceType, TPrecision>::ExecutionContextType>( device_id );
            }
            else
            {
                exec_ctx = std::make_shared<typename MockModule<TDeviceType, TPrecision>::ExecutionContextType>();
            }

            data.module = std::make_shared<MockModule<TDeviceType, TPrecision>>( config, exec_ctx );

            if (data.is_training)
            {
                data.module->setTraining( true );
            }

            return data;
        }

        static ModuleTestData CreateWithContext(
            std::shared_ptr<typename MockModule<TDeviceType, TPrecision>::ExecutionContextType> exec_context,
            bool is_training = false )
        {
            ModuleTestData data;
            data.is_training = is_training;

            MockModuleConfig config;
            config.withName( "mock_module_context" );

            data.module = std::make_shared<MockModule<TDeviceType, TPrecision>>( config, exec_context );

            if (data.is_training)
            {
                data.module->setTraining( true );
            }

            return data;
        }
    };

    class ModuleTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
        }

        void TearDown() override
        {
            cpu_data_.module.reset();
            cuda_data_.module.reset();
            training_cpu_data_.module.reset();
            training_cuda_data_.module.reset();
            context_cpu_data_.module.reset();
            context_cuda_data_.module.reset();
        }

        ModuleTestData<DeviceType::Cpu> & CpuData()
        {
            if (!cpu_data_.module)
            {
                // CPU device id: 0 (CPU implementations may ignore the id)
                cpu_data_ = ModuleTestData<DeviceType::Cpu>::Create( 0 );
            }
            return cpu_data_;
        }

        ModuleTestData<DeviceType::Cuda> & CudaData()
        {
            if (!cuda_data_.module)
            {
                cuda_data_ = ModuleTestData<DeviceType::Cuda>::Create( 0 );
            }
            return cuda_data_;
        }

        ModuleTestData<DeviceType::Cpu> & TrainingCpuData()
        {
            if (!training_cpu_data_.module)
            {
                training_cpu_data_ = ModuleTestData<DeviceType::Cpu>::Create( 0, true );
            }
            return training_cpu_data_;
        }

        ModuleTestData<DeviceType::Cuda> & TrainingCudaData()
        {
            if (!training_cuda_data_.module)
            {
                training_cuda_data_ = ModuleTestData<DeviceType::Cuda>::Create( 0, true );
            }
            return training_cuda_data_;
        }

        ModuleTestData<DeviceType::Cpu> & ContextCpuData()
        {
            if (!context_cpu_data_.module)
            {
                // Create a CPU execution context explicitly (CPU execution context has default ctor)
                auto exec_context = std::make_shared<typename MockModule<DeviceType::Cpu, TensorDataType::FP32>::ExecutionContextType>();
                context_cpu_data_ = ModuleTestData<DeviceType::Cpu>::CreateWithContext( exec_context );
            }
            return context_cpu_data_;
        }

        ModuleTestData<DeviceType::Cuda> & ContextCudaData()
        {
            if (!context_cuda_data_.module)
            {
                // Create a CUDA execution context for device 0
                auto exec_context = std::make_shared<typename MockModule<DeviceType::Cuda, TensorDataType::FP32>::ExecutionContextType>( 0 );
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

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestGetName( const ModuleTestData<TDeviceType, TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.module->getName(), expected_name );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestParameterCount( const ModuleTestData<TDeviceType, TPrecision>& data, size_t expected_count )
    {
        EXPECT_EQ( data.module->parameterCount(), expected_count );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestPrint( const ModuleTestData<TDeviceType, TPrecision>& data, const std::string& expected_substring )
    {
        std::string output = data.module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestTrainingMode( const ModuleTestData<TDeviceType, TPrecision>& data, bool expected_mode )
    {
        EXPECT_EQ( data.module->isTraining(), expected_mode );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestDeviceType( const ModuleTestData<TDeviceType, TPrecision>& data )
    {
        auto exec_context = data.module->getExecutionContext();
        EXPECT_NE( exec_context, nullptr );

        auto device = exec_context->getDevice();
        EXPECT_NE( device, nullptr );

        EXPECT_EQ( device->getDeviceType(), TDeviceType );
        EXPECT_EQ( data.module->getDeviceType(), TDeviceType );
    }

    // Minimal precision test to match new Module API (mock exposes precision via config)
    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestPrecision( const ModuleTestData<TDeviceType, TPrecision>& data )
    {
        EXPECT_EQ( data.module->getPrecisionPolicy(), ComputePrecision::Policy::Auto );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestHelperMethods( const ModuleTestData<TDeviceType, TPrecision>& data )
    {
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
    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestForwardBackward( const ModuleTestData<TDeviceType, TPrecision>& data )
    {
        // Determine appropriate memory resource for device
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda,
            CudaDeviceMemoryResource,
            CpuMemoryResource>;

        std::vector<int64_t> shape = { 2, 3 };

        // Get execution context & device from module
        auto exec_ctx = data.module->getExecutionContext();
        auto device = exec_ctx->getDevice();

        // Create tensors using device pointer (Tensor ctor accepts device)
        Tensor<TensorDataType::FP32, MR> input( device, shape );
        Tensor<TensorDataType::FP32, MR> output( device, shape );
        Tensor<TensorDataType::FP32, MR> input_grad( device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( device, shape );

        // Initialize input and output_grad
        if constexpr (TDeviceType == DeviceType::Cpu)
        {
            auto* input_data = input.data();
            auto* grad_data = output_grad.data();

            for (size_t i = 0; i < input.size(); ++i)
            {
                input_data[i] = static_cast<float>( i + 1.0f );
                grad_data[i] = 0.5f;
            }
        }
        else
        {
            // For CUDA, initialize via host tensor and transfer
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( "CPU", shape );
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_grad( "CPU", shape );

            auto* host_input_data = host_input.data();
            auto* host_grad_data = host_grad.data();

            for (size_t i = 0; i < host_input.size(); ++i)
            {
                host_input_data[i] = static_cast<float>( i + 1.0f );
                host_grad_data[i] = 0.5f;
            }

            // Transfer to device using TensorOps if available (kept minimal here)
            exec_ctx->synchronize();
        }

        // Test forward pass via ITensor interface
        EXPECT_NO_THROW( data.module->forward( input, output ) );

        // Test backward pass via ITensor interface
        EXPECT_NO_THROW( data.module->backward( input, output_grad, input_grad ) );

        // Verify basic tensor properties (do not assume mock implements identity copy)
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( input_grad.size(), output_grad.size() );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestInvalidDeviceId()
    {
        MockModuleConfig config;
        config.withName( "test_module" );

        if constexpr (TDeviceType == DeviceType::Cuda)
        {
            // Negative device id should be invalid for CUDA: creating the ExecutionContext must fail
            EXPECT_THROW(
                (std::make_shared<typename MockModule<TDeviceType, TPrecision>::ExecutionContextType>( -1 )),
                std::invalid_argument
            );
        }
        else
        {
            // CPU may accept the id (ignored), ensure construction succeeds
            auto exec_ctx = std::make_shared<typename MockModule<TDeviceType, TPrecision>::ExecutionContextType>();
            EXPECT_NO_THROW(
                (std::make_shared<MockModule<TDeviceType, TPrecision>>( config, exec_ctx ))
            );
        }
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestNullExecutionContext()
    {
        MockModuleConfig config;
        config.withName( "test_module" );

        std::shared_ptr<typename MockModule<TDeviceType, TPrecision>::ExecutionContextType> null_context;

        EXPECT_THROW(
            (std::make_shared<MockModule<TDeviceType, TPrecision>>( config, null_context )),
            std::invalid_argument
        );
    }

    // ====================================================================
    // CPU Tests
    // ====================================================================

    TEST_F( ModuleTests, Cpu_GetName )
    {
        TestGetName( CpuData(), "mock_module" );
    }

    TEST_F( ModuleTests, Cpu_ParameterCount )
    {
        TestParameterCount( CpuData(), 0 );
    }

    TEST_F( ModuleTests, Cpu_Print )
    {
        TestPrint( CpuData(), "MockModule: mock_module" );
    }

    TEST_F( ModuleTests, Cpu_TrainingMode )
    {
        TestTrainingMode( CpuData(), false );
    }

    TEST_F( ModuleTests, Cpu_DeviceType )
    {
        TestDeviceType( CpuData() );
    }

    TEST_F( ModuleTests, Cpu_Precision )
    {
        TestPrecision( CpuData() );
    }

    TEST_F( ModuleTests, Cpu_HelperMethods )
    {
        TestHelperMethods( CpuData() );
    }

    TEST_F( ModuleTests, Cpu_ForwardBackward )
    {
        TestForwardBackward( CpuData() );
    }

    TEST_F( ModuleTests, Cpu_Training_TrainingMode )
    {
        TestTrainingMode( TrainingCpuData(), true );
    }

    TEST_F( ModuleTests, Context_Cpu_DeviceType )
    {
        TestDeviceType( ContextCpuData() );
    }

    // ====================================================================
    // CUDA Tests
    // ====================================================================

    TEST_F( ModuleTests, Cuda_GetName )
    {
        TestGetName( CudaData(), "mock_module" );
    }

    TEST_F( ModuleTests, Cuda_ParameterCount )
    {
        TestParameterCount( CudaData(), 0 );
    }

    TEST_F( ModuleTests, Cuda_Print )
    {
        TestPrint( CudaData(), "MockModule: mock_module" );
    }

    TEST_F( ModuleTests, Cuda_TrainingMode )
    {
        TestTrainingMode( CudaData(), false );
    }

    TEST_F( ModuleTests, Cuda_DeviceType )
    {
        TestDeviceType( CudaData() );
    }

    TEST_F( ModuleTests, Cuda_Precision )
    {
        TestPrecision( CudaData() );
    }

    TEST_F( ModuleTests, Cuda_ForwardBackward )
    {
        TestForwardBackward( CudaData() );
    }

    TEST_F( ModuleTests, Cuda_Training_TrainingMode )
    {
        TestTrainingMode( TrainingCudaData(), true );
    }

    TEST_F( ModuleTests, Context_Cuda_DeviceType )
    {
        TestDeviceType( ContextCudaData() );
    }

    // ====================================================================
    // Configuration Tests
    // ====================================================================

    TEST_F( ModuleTests, ModuleConfig_DefaultValues )
    {
        MockModuleConfig config;
        EXPECT_EQ( config.getName(), "mock_module" );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Auto );
    }

    TEST_F( ModuleTests, ModuleConfig_CustomValues )
    {
        MockModuleConfig config;
        config.withName( "custom_module" )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance );

        EXPECT_EQ( config.getName(), "custom_module" );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
    }

    // ====================================================================
    // Error Handling Tests
    // ====================================================================

    TEST_F( ModuleTests, InvalidDeviceId_Cpu )
    {
        TestInvalidDeviceId<DeviceType::Cpu>();
    }

    TEST_F( ModuleTests, InvalidDeviceId_Cuda )
    {
        TestInvalidDeviceId<DeviceType::Cuda>();
    }

    TEST_F( ModuleTests, NullExecutionContext_Cpu )
    {
        TestNullExecutionContext<DeviceType::Cpu>();
    }

    TEST_F( ModuleTests, NullExecutionContext_Cuda )
    {
        TestNullExecutionContext<DeviceType::Cuda>();
    }
}