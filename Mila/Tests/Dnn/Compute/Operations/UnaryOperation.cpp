/**
 * @file UnaryOperation.cpp
 * @brief Compile-time unit tests for UnaryOperation interface and aliases.
 *
 * Tests focus on compile-time traits, aliases and type definitions only to avoid
 * constructing abstract/base classes that require runtime ExecutionContext plumbing.
 */

#include <gtest/gtest.h>
#include <type_traits>
#include <vector>
#include <memory>

import Mila;

namespace Dnn::Compute::Operations::Tests
{
	using namespace Mila::Dnn;
	using namespace Mila::Dnn::Compute;

	// Test fixture class - group tests by project namespace and class
	class UnaryOperationTest : public ::testing::Test {
	protected:
		void SetUp() override {}
		void TearDown() override {}
	};

	TEST_F( UnaryOperationTest, CpuFp32IsAbstractAndTraitsCorrect ) {
		using Op = UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>;

		// Class should be abstract because forward() is pure virtual.
		static_assert(std::is_abstract_v<Op>);
		EXPECT_TRUE( std::is_abstract_v<Op> );

		// Compile-time constants inherited from OperationBase
		static_assert(Op::data_type == TensorDataType::FP32);
		static_assert(Op::device_type == DeviceType::Cpu);

		EXPECT_EQ( Op::getDataType(), TensorDataType::FP32 );
		EXPECT_EQ( std::string_view( Op::getDataTypeName() ), std::string_view( Op::DataTypeTraits::type_name ) );

		// Floating / integer type traits
		static_assert(Op::isFloatType());
		EXPECT_TRUE( Op::isFloatType() );
		EXPECT_FALSE( Op::isIntegerType() );
	}

	TEST_F( UnaryOperationTest, ParametersAndOutputState ) {
		using Op = UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>;

		// Parameters should be a vector of shared_ptr to ITensor (type-erased)
		// FIXME:
		//using ExpectedParams = std::vector<std::shared_ptr<Mila::Dnn::ITensor>>;
		//static_assert(std::is_same_v<typename Op::Parameters, ExpectedParams>);
		//EXPECT_TRUE( (std::is_same_v<typename Op::Parameters, ExpectedParams>) );

		//// OutputState expected to be vector of shared_ptr<ITensor> (type-erased) per UnaryOperation definition
		//static_assert(std::is_same_v<typename Op::OutputState, ExpectedParams>);
		//EXPECT_TRUE( (std::is_same_v<typename Op::OutputState, ExpectedParams>) );
	}

	TEST_F( UnaryOperationTest, CpuAndCudaAliases ) {
		// Default alias types (defaults to FP32)
		static_assert(std::is_same_v<CpuUnaryOperation<>, UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>>);
		static_assert(std::is_same_v<CudaUnaryOperation<>, UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>);

		// Explicit alias usage
		using MyCpuFp16 = CpuUnaryOperation<TensorDataType::FP16>;
		static_assert(std::is_same_v<MyCpuFp16, UnaryOperation<DeviceType::Cpu, TensorDataType::FP16>>);

		SUCCEED();
	}

	TEST_F( UnaryOperationTest, ConstexprQueries ) {
		// Ensure compile-time helpers are usable in constexpr context
		constexpr auto dtype = UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::getDataType();
		constexpr auto name = UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::getDataTypeName();
		constexpr auto elemSize = UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::getElementSize();

		static_assert(dtype == TensorDataType::FP32);
		static_assert(name.size() > 0);
		static_assert(elemSize == UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::DataTypeTraits::size_in_bytes);

		EXPECT_EQ( dtype, TensorDataType::FP32 );
		EXPECT_EQ( std::string_view( name ), std::string_view( UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::DataTypeTraits::type_name ) );
	}
}