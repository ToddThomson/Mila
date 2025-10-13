#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <stdexcept>

import Mila;

namespace Mila::Dnn::Compute::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Mock concrete BinaryOperation for CPU FP32 that implements simple element-wise add
    class MockAddOp : public BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using Base = BinaryOperation<DeviceType::Cpu, TensorDataType::FP32>;
        using TensorType = typename Base::TensorType;
        using Parameters = typename Base::Parameters;
        using OutputState = typename Base::OutputState;

        explicit MockAddOp( OperationType op_type, std::shared_ptr<ExecutionContext<DeviceType::Cpu>> ctx )
            : Base( op_type, ctx ) {
        }

        std::string getName() const override {
            return "MockAddOp";
        }

        void forward(
            const ITensor& inputA,
            const ITensor& inputB,
            const Parameters& /*parameters*/,
            ITensor& output,
            OutputState& /*output_state*/ ) const override
        {
            // Support scalar broadcasting and equal-shape elementwise add
            //if (inputA.isScalar() && inputB.isScalar())
            //{
            //    output.reshape( {} ); // scalar
            //    output.item() = inputA.item() + inputB.item();
            //    return;
            //}

            //if (inputA.isScalar())
            //{
            //    output.reshape( inputB.shape() );
            //    auto scalar = inputA.item();
            //    auto* out = output.data();
            //    auto* b = inputB.data();
            //    for (size_t i = 0; i < output.size(); ++i) out[i] = scalar + b[i];
            //    return;
            //}

            //if (inputB.isScalar())
            //{
            //    output.reshape( inputA.shape() );
            //    auto scalar = inputB.item();
            //    auto* out = output.data();
            //    auto* a = inputA.data();
            //    for (size_t i = 0; i < output.size(); ++i) out[i] = a[i] + scalar;
            //    return;
            //}

            // Both tensors must have identical shape for this simple mock
            if (inputA.shape() != inputB.shape())
            {
                throw std::invalid_argument( "Shapes must match for element-wise add" );
            }

            //output.reshape( inputA.shape() );

			const auto& a = dynamic_cast<const TensorType&>(inputA);
			const auto& b = dynamic_cast<const TensorType&>(inputB);
			auto& out = dynamic_cast<TensorType&>(output);

            auto* out_ptr = out.data();
            auto* a_ptr = a.data();
            auto* b_ptr = b.data();

            for (size_t i = 0; i < out.size(); ++i)
            {
                out_ptr[i] = a_ptr[i] + b_ptr[i];
            }
        }
    };

    class BinaryOperationTests : public ::testing::Test {
    protected:
        void SetUp() override {
            // Create a CPU execution context for constructing operations and tensors
            exec_ctx_ = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            op_ = std::make_shared<MockAddOp>( OperationType::LinearOp, exec_ctx_ );
        }

        void TearDown() override {
            op_.reset();
            exec_ctx_.reset();
        }

        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_ctx_;
        std::shared_ptr<MockAddOp> op_;
    };

    TEST_F( BinaryOperationTests, GetNameAndDevice ) {
        ASSERT_NE( op_, nullptr );
        EXPECT_EQ( op_->getName(), "MockAddOp" );
        EXPECT_EQ( op_->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( op_->getExecutionContext()->getDevice()->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( op_->getOperationType(), OperationType::LinearOp );
    }

    TEST_F( BinaryOperationTests, Forward_ElementwiseAdd ) {
        using T = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>;

        std::vector<size_t> shape = { 2, 3 };
        auto device = exec_ctx_->getDevice();

        T a( device, shape );
        T b( device, shape );
        T out( device, shape ); // preallocated

        // Initialize inputs
        for (size_t i = 0; i < a.size(); ++i)
        {
            a.data()[i] = static_cast<float>( i + 1 );
            b.data()[i] = static_cast<float>( (i + 1) * 2 );
        }

        MockAddOp::Parameters params;
        MockAddOp::OutputState state;

        EXPECT_NO_THROW( op_->forward( a, b, params, out, state ) );

        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_FLOAT_EQ( out.data()[i], a.data()[i] + b.data()[i] );
        }
    }

    TEST_F( BinaryOperationTests, Forward_ScalarAndTensorBroadcast ) {
        using T = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>;

        auto device = exec_ctx_->getDevice();

        // Scalar + tensor
        T scalar( device, std::vector<size_t>{} );
        T vec( device, std::vector<size_t>{ 4 } );
        T out( device, std::vector<size_t>{ 4 } );

        scalar.item() = 2.5f;
        for (size_t i = 0; i < vec.size(); ++i) vec.data()[i] = static_cast<float>( i );

        MockAddOp::Parameters params;
        MockAddOp::OutputState state;

        EXPECT_NO_THROW( op_->forward( scalar, vec, params, out, state ) );
        EXPECT_EQ( out.shape(), vec.shape() );
        for (size_t i = 0; i < out.size(); ++i)
        {
            EXPECT_FLOAT_EQ( out.data()[i], vec.data()[i] + 2.5f );
        }

        // Tensor + scalar
        T out2( device, std::vector<size_t>{} );
        scalar.item() = -1.0f;
        EXPECT_NO_THROW( op_->forward( vec, scalar, params, out2, state ) );
        EXPECT_EQ( out2.shape(), vec.shape() );
        for (size_t i = 0; i < out2.size(); ++i)
        {
            EXPECT_FLOAT_EQ( out2.data()[i], vec.data()[i] - 1.0f );
        }

        // Scalar + scalar -> scalar
        T s1( device, std::vector<size_t>{} );
        T s2( device, std::vector<size_t>{} );
        T sout( device, std::vector<size_t>{} );

        s1.item() = 1.5f;
        s2.item() = 0.5f;
        EXPECT_NO_THROW( op_->forward( s1, s2, params, sout, state ) );
        EXPECT_TRUE( sout.isScalar() );
        EXPECT_FLOAT_EQ( sout.item(), 2.0f );
    }

    TEST_F( BinaryOperationTests, Forward_MismatchedShapesThrows ) {
        using T = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>;
        auto device = exec_ctx_->getDevice();

        T a( device, std::vector<size_t>{ 2, 2 } );
        T b( device, std::vector<size_t>{ 3 } );
        T out( device, std::vector<size_t>{} );

        MockAddOp::Parameters params;
        MockAddOp::OutputState state;

        EXPECT_THROW( op_->forward( a, b, params, out, state ), std::invalid_argument );
    }

    TEST_F( BinaryOperationTests, Backward_DefaultThrows ) {
        using T = Tensor<TensorDataType::FP32, Compute::CpuMemoryResource>;
        auto device = exec_ctx_->getDevice();

        T a( device, std::vector<size_t>{ 2 } );
        T b( device, std::vector<size_t>{ 2 } );
        T out( device, std::vector<size_t>{ 2 } );
        T out_grad( device, std::vector<size_t>{ 2 } );
        T a_grad( device, std::vector<size_t>{} );
        T b_grad( device, std::vector<size_t>{} );

        MockAddOp::Parameters params_vec;
        std::shared_ptr<ITensor> params_ptr; // nullptr allowed
        std::vector<std::shared_ptr<T>> param_grads;
        MockAddOp::OutputState state;

        // The base BinaryOperation::backward throws by default; ensure it's thrown when not overridden.
        EXPECT_THROW(
            op_->backward( a, b, out, out_grad, params_ptr, param_grads, a_grad, b_grad, state ),
            std::runtime_error );
    }
}