/**
 * @file OperationBase.cpp
 * @brief Unit tests for OperationBase compile-time traits and aliases.
 *
 * These tests avoid constructing OperationBase instances because the module's
 * runtime API for ExecutionContext is validated elsewhere. Tests focus on
 * compile-time constants, type traits and aliases provided by OperationBase.
 */

#include <gtest/gtest.h>
#include <type_traits>
#include <string_view>

import Mila;

namespace Dnn::Compute::Operations::Tests
{
	using namespace Mila::Dnn;
	using namespace Mila::Dnn::Compute;

	// Test fixture class - group tests by project namespace and class as requested
	class OperationBaseTest : public ::testing::Test {
	protected:
		void SetUp() override {}
		void TearDown() override {}
	};

	TEST_F( OperationBaseTest, CpuFp32Traits ) {
		using Op = Operation<DeviceType::Cpu, TensorDataType::FP32>;

		// Compile-time constants should be exposed correctly
		static_assert(Op::data_type == TensorDataType::FP32);
		static_assert(Op::device_type == DeviceType::Cpu);

		//EXPECT_EQ( Op::getDataType(), TensorDataType::FP32 );
		//EXPECT_EQ( std::string_view( Op::getDataTypeName() ), std::string_view( Op::DataTypeTraits::type_name ) );
		//EXPECT_EQ( Op::getElementSize(), Op::DataTypeTraits::size_in_bytes );

		// FP32 is floating point type
		//static_assert(Op::isFloatType());
		//EXPECT_TRUE( Op::isFloatType() );
		//EXPECT_FALSE( Op::isIntegerType() );
	}

	//TEST_F( OperationBaseTest, Int8Traits ) {
	//	using Op = OperationBase<DeviceType::Cpu, TensorDataType::INT8>;

	//	static_assert(Op::data_type == TensorDataType::INT8);
	//	EXPECT_EQ( Op::getDataType(), TensorDataType::INT8 );
	//	EXPECT_EQ( std::string_view( Op::getDataTypeName() ), std::string_view( Op::DataTypeTraits::type_name ) );
	//	EXPECT_EQ( Op::getElementSize(), Op::DataTypeTraits::size_in_bytes );

	//	// INT8 should be integer type
	//	static_assert(Op::isIntegerType());
	//	EXPECT_TRUE( Op::isIntegerType() );
	//	EXPECT_FALSE( Op::isFloatType() );
	//}

	//TEST_F( OperationBaseTest, CpuAndCudaAliases ) {
	//	// Default alias types (defaults to FP32)
	//	static_assert(std::is_same_v<CpuOperationBase<>, OperationBase<DeviceType::Cpu, TensorDataType::FP32>>);
	//	static_assert(std::is_same_v<CudaOperationBase<>, OperationBase<DeviceType::Cuda, TensorDataType::FP32>>);

	//	// Explicit alias usage
	//	using MyCpuFp16 = CpuOperationBase<TensorDataType::FP16>;
	//	static_assert(std::is_same_v<MyCpuFp16, OperationBase<DeviceType::Cpu, TensorDataType::FP16>>);

	//	SUCCEED();
	//}

	//TEST_F( OperationBaseTest, CompileTimeQueries ) {
	//	// Verify compile-time helper functions are constexpr-callable
	//	constexpr auto dtype = OperationBase<DeviceType::Cpu, TensorDataType::FP32>::getDataType();
	//	constexpr auto name = OperationBase<DeviceType::Cpu, TensorDataType::FP32>::getDataTypeName();
	//	constexpr auto elemSize = OperationBase<DeviceType::Cpu, TensorDataType::FP32>::getElementSize();

	//	static_assert(dtype == TensorDataType::FP32);
	//	static_assert(name.size() > 0);
	//	static_assert(elemSize == OperationBase<DeviceType::Cpu, TensorDataType::FP32>::DataTypeTraits::size_in_bytes);

	//	EXPECT_EQ( dtype, TensorDataType::FP32 );
	//	EXPECT_EQ( std::string_view( name ), std::string_view( OperationBase<DeviceType::Cpu, TensorDataType::FP32>::DataTypeTraits::type_name ) );
	//}
}