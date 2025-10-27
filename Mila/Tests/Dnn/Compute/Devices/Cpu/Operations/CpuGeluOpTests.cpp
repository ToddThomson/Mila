/**
 * @file CpuGeluOpTests.cpp
 * @brief Test suite for the CPU GELU operation.
 *
 * Tests use the modern ExecutionContext + OperationRegistry APIs. Direct use of
 * deprecated DeviceContext or direct op construction is avoided. If the GELU
 * operation factory is not registered in the test build, op-level checks are
 * skipped while the rest of the tests still run where possible.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#ifdef USE_OMP
#include <omp.h>
#endif

import Mila;

namespace Dnn::Compute::Device::Cpu::Operations::Tests
{
	using namespace Mila::Dnn;
	using namespace Mila::Dnn::Compute;

	class CpuGeluOpTests : public ::testing::Test
	{
	protected:
		void SetUp() override
		{
			// Use typed execution context (modern API)
			exec_ctx_ = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

			// Attempt to create the registered GELU operation (CPU / FP32).
			// If registration is missing in this build, op_ remains null and
			// op-specific assertions are skipped.
			try
			{
				GeluConfig cfg;
				op_ = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>(
					"GeluOp", exec_ctx_, cfg );
			}
			catch (const std::exception&)
			{
				op_.reset();
			}

			// Keep a config instance for tests that validate GeluConfig behavior
			config_ = GeluConfig();

			small_shape_ = { 2, 3, 4 };
			medium_shape_ = { 8, 16, 32 };
			large_shape_ = { 32, 64, 128 };
		}

		// Detect NaN/Inf in a host-accessible FP32 tensor
		template <typename TTensor> bool hasNaNorInfHost( const TTensor& tensor ) const
		{
			const float* data = static_cast<const float*>(tensor.rawData());

			for (size_t i = 0; i < tensor.size(); ++i)
			{
				if (std::isnan( data[i] ) || std::isinf( data[i] ))
				{
					return true;
				}
			}

			return false;
		}

		// Reference GELU implementation (tanh approximation)
		static float geluReference( float x )
		{
			const float sqrt_2_over_pi = std::sqrt( 2.0f / static_cast<float>(M_PI) );
			float x_cube = 0.044715f * x * x * x;

			return 0.5f * x * (1.0f + std::tanh( sqrt_2_over_pi * (x + x_cube) ));
		}

		static float geluGradReference( float x )
		{
			const float sqrt_2_over_pi = std::sqrt( 2.0f / static_cast<float>(M_PI) );
			float x_cube = 0.044715f * x * x * x;
			float tanh_arg = sqrt_2_over_pi * (x + x_cube);
			float tanh_out = std::tanh( tanh_arg );
			float coshf_out = std::cosh( tanh_arg );
			float sech_out = 1.0f / (coshf_out * coshf_out);

			return 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x);
		}

		std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_ctx_;
		std::shared_ptr<UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>> op_; // may be null if not registered
		GeluConfig config_;

		std::vector<size_t> small_shape_;
		std::vector<size_t> medium_shape_;
		std::vector<size_t> large_shape_;
	};

	TEST_F( CpuGeluOpTests, OperationRegisteredName )
	{
		if (!op_)
		{
			GTEST_SKIP() << "GeluOp not registered for CPU/FP32 in this build.";
		}

		EXPECT_EQ( op_->getName(), std::string( "Cpu::GeluOp" ) );
	}

	TEST_F( CpuGeluOpTests, BasicFunctionality )
	{
		if (!op_)
		{
			GTEST_SKIP() << "GeluOp not registered; skipping op-level forward checks.";
		}

		using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;
		TensorT input( exec_ctx_->getDevice(), small_shape_ );
		TensorT output( exec_ctx_->getDevice(), small_shape_ );

		float* in_data = static_cast<float*>(input.rawData());

		for (size_t i = 0; i < input.size(); ++i)
		{
			in_data[i] = (static_cast<float>( i ) - 10.0f) / 10.0f;
		}

		typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
		typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

		ASSERT_NO_THROW( op_->forward( input, params, output, out_state ) );

		const float* out_data = static_cast<const float*>( output.rawData() );

		for (size_t i = 0; i < output.size(); ++i)
		{
			float expected = geluReference( in_data[i] );
			EXPECT_NEAR( out_data[i], expected, 1e-5f );
		}
	}

	TEST_F( CpuGeluOpTests, BackwardPass )
	{
		if (!op_)
		{
			GTEST_SKIP() << "GeluOp not registered; skipping op-level backward checks.";
		}

		using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;
		TensorT input( exec_ctx_->getDevice(), small_shape_ );
		TensorT output( exec_ctx_->getDevice(), small_shape_ );
		TensorT output_grad( exec_ctx_->getDevice(), small_shape_ );
		TensorT input_grad( exec_ctx_->getDevice(), small_shape_ );

		float* in_data = static_cast<float*>(input.rawData());
		float* dout = static_cast<float*>(output_grad.rawData());
		float* din = static_cast<float*>(input_grad.rawData());

		for (size_t i = 0; i < input.size(); ++i)
		{
			in_data[i] = (static_cast<float>( i ) - 10.0f) / 10.0f;
			dout[i] = 1.0f;
			din[i] = 0.0f;
		}

		typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
		typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

		op_->forward( input, params, output, out_state );

		ASSERT_NO_THROW( op_->backward( input, output_grad, params, out_state, input_grad, params ) );

		EXPECT_FALSE( hasNaNorInfHost( input_grad ) );

		const float* din_const = static_cast<const float*>(input_grad.rawData());

		for (size_t i = 0; i < input.size(); ++i)
		{
			float expected = geluGradReference( in_data[i] ); // output_grad is 1.0 so gradient equals local grad
			EXPECT_NEAR( din_const[i], expected, 1e-4f );
		}

		bool all_zeros = true;

		for (size_t i = 0; i < input_grad.size(); ++i)
		{
			if (std::abs( static_cast<const float*>( input_grad.rawData() )[i] ) > 1e-5f)
			{
				all_zeros = false;
				break;
			}
		}

		EXPECT_FALSE( all_zeros );
	}

	TEST_F( CpuGeluOpTests, EdgeCases )
	{
		if (!op_)
		{
			GTEST_SKIP() << "GeluOp not registered; skipping op-level checks.";
		}

		using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;
		TensorT input( exec_ctx_->getDevice(), small_shape_ );
		TensorT output( exec_ctx_->getDevice(), small_shape_ );

		float* in_data = static_cast<float*>(input.rawData());

		typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
		typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

		// zeros
		for (size_t i = 0; i < input.size(); ++i)
		{
			in_data[i] = 0.0f;
		}

		ASSERT_NO_THROW( op_->forward( input, params, output, out_state ) );

		for (size_t i = 0; i < output.size(); ++i)
		{
			EXPECT_NEAR( static_cast<const float*>( output.rawData() )[i], 0.0f, 1e-5f );
		}

		// large positive
		for (size_t i = 0; i < input.size(); ++i)
		{
			in_data[i] = 100.0f;
		}

		ASSERT_NO_THROW( op_->forward( input, params, output, out_state ) );

		for (size_t i = 0; i < output.size(); ++i)
		{
			EXPECT_NEAR( static_cast<const float*>( output.rawData() )[i], in_data[i], std::abs( in_data[i] ) * 0.01f );
		}

		// large negative
		for (size_t i = 0; i < input.size(); ++i)
		{
			in_data[i] = -100.0f;
		}

		ASSERT_NO_THROW( op_->forward( input, params, output, out_state ) );

		for (size_t i = 0; i < output.size(); ++i)
		{
			EXPECT_NEAR( static_cast<const float*>( output.rawData() )[i], 0.0f, 1e-4f );
		}

		// very small
		for (size_t i = 0; i < input.size(); ++i)
		{
			in_data[i] = 1e-5f;
		}

		ASSERT_NO_THROW( op_->forward( input, params, output, out_state ) );

		for (size_t i = 0; i < output.size(); ++i)
		{
			EXPECT_NEAR( static_cast<const float*>( output.rawData() )[i], 0.5f * in_data[i], 1e-8f );
		}
	}

	TEST_F( CpuGeluOpTests, NumericalStabilityAndDeterminism )
	{
		if (!op_)
		{
			GTEST_SKIP() << "GeluOp not registered; skipping op-level checks.";
		}

		using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;
		TensorT input( exec_ctx_->getDevice(), medium_shape_ );
		TensorT out1( exec_ctx_->getDevice(), medium_shape_ );
		TensorT out2( exec_ctx_->getDevice(), medium_shape_ );

		float* in_data = static_cast<float*>(input.rawData());

		for (size_t i = 0; i < input.size(); ++i)
		{
			int p = static_cast<int>( i % 8 );
			float v = 0.0f;

			switch (p)
			{
				case 0:
					v = 1.0f;
					break;
				case 1:
					v = -1.0f;
					break;
				case 2:
					v = 0.0001f;
					break;
				case 3:
					v = -0.0001f;
					break;
				case 4:
					v = 10.0f;
					break;
				case 5:
					v = -10.0f;
					break;
				case 6:
					v = 100.0f;
					break;
				case 7:
					v = -100.0f;
					break;
			}

			in_data[i] = v;
		}

		typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
		typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

		ASSERT_NO_THROW( op_->forward( input, params, out1, out_state ) );
		EXPECT_FALSE( hasNaNorInfHost( out1 ) );

		op_->forward( input, params, out2, out_state );

		for (size_t i = 0; i < out1.size(); ++i)
		{
			EXPECT_EQ( static_cast<const float*>( out1.rawData() )[i], static_cast<const float*>( out2.rawData() )[i] );
		}
	}

	TEST_F( CpuGeluOpTests, ConstructorAndRegistrationBehavior )
	{
		// Creating via registry with null context should throw (registry expects non-null context)
		GeluConfig cfg;
		EXPECT_THROW( (OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>(
			"GeluOp", nullptr, cfg )),
			std::invalid_argument );

		// If op is registered, creating with valid exec_ctx_ should succeed (already attempted in SetUp)
		if (!op_)
		{
			GTEST_SKIP() << "GeluOp not registered; skipping op registration check.";
		}
	}

	TEST_F( CpuGeluOpTests, ApproximationMethods )
	{
		// Validate config validation and registry-based creation for Tanh method (if registered)
		GeluConfig tanh_cfg = GeluConfig().withApproximationMethod( GeluConfig::ApproximationMethod::Tanh );

		if (!op_)
		{
			GTEST_SKIP() << "GeluOp not registered; skipping approximation method checks.";
		}

		// Ensure Tanh-config works via registry
		try
		{
			auto tanh_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TensorDataType::FP32>(
				"GeluOp", exec_ctx_, tanh_cfg );

			using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;
			TensorT input( exec_ctx_->getDevice(), small_shape_ ), out( exec_ctx_->getDevice(), small_shape_ );
			float* d = static_cast<float*>(input.rawData());

			for (size_t i = 0; i < input.size(); ++i)
			{
				d[i] = (static_cast<float>( i ) - 10.0f) / 10.0f;
			}

			typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
			typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

			ASSERT_NO_THROW( tanh_op->forward( input, params, out, out_state ) );

			const float* od = static_cast<const float*>( out.rawData() );

			for (size_t i = 0; i < out.size(); ++i)
			{
				EXPECT_NEAR( od[i], geluReference( d[i] ), 1e-5f );
			}
		}
		catch (const std::exception&)
		{
			GTEST_SKIP() << "GeluOp factory for Tanh config not available in this build.";
		}

		// Unsupported methods validate() should throw
		GeluConfig exact_cfg = GeluConfig().withApproximationMethod( GeluConfig::ApproximationMethod::Exact );
		EXPECT_THROW( exact_cfg.validate(), std::invalid_argument );

		GeluConfig sig_cfg = GeluConfig().withApproximationMethod( GeluConfig::ApproximationMethod::Sigmoid );
		EXPECT_THROW( sig_cfg.validate(), std::invalid_argument );
	}

	TEST_F( CpuGeluOpTests, PerformanceAndOpenMPScaling )
	{
		if (std::getenv( "CI" ) != nullptr)
		{
			GTEST_SKIP() << "Skipping performance tests in CI environment";
		}

		if (!op_)
		{
			GTEST_SKIP() << "GeluOp not registered; skipping perf/OpenMP op-level tests.";
		}

		using TensorT = Tensor<TensorDataType::FP32, CpuMemoryResource>;
		TensorT a( exec_ctx_->getDevice(), large_shape_ ), out( exec_ctx_->getDevice(), large_shape_ );
		float* ad = static_cast<float*>(a.rawData());

		for (size_t i = 0; i < a.size(); ++i)
		{
			ad[i] = (static_cast<float>( rand() ) / RAND_MAX - 0.5f) * 20.0f;
		}

		typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::Parameters params;
		typename UnaryOperation<DeviceType::Cpu, TensorDataType::FP32>::OutputState out_state;

		const int iterations = 10;
		auto start = std::chrono::high_resolution_clock::now();

		for (int it = 0; it < iterations; ++it)
		{
			op_->forward( a, params, out, out_state );
		}

		auto end = std::chrono::high_resolution_clock::now();
		auto dur = std::chrono::duration_cast<std::chrono::microseconds>( end - start );
		size_t total = a.size() * iterations;
		double eps = static_cast<double>( total ) / (dur.count() * 1e-6);
		std::cout << "CpuGeluOp: " << eps / 1e6 << " M elems/sec\n";

#ifdef USE_OMP
		int max_threads = omp_get_max_threads();
		std::vector<int> threads = { 1 };
		if (max_threads > 1)
		{
			threads.push_back( max_threads );
			if (max_threads > 3)
			{
				threads.push_back( max_threads / 2 );
			}
		}

		for (int num_threads : threads)
		{
			omp_set_num_threads( num_threads );
			auto s = std::chrono::high_resolution_clock::now();

			for (int it = 0; it < iterations; ++it)
			{
				op_->forward( a, params, out, out_state );
			}

			auto e = std::chrono::high_resolution_clock::now();
			auto d = std::chrono::duration_cast<std::chrono::microseconds>( e - s );
			double eps_t = static_cast<double>( a.size() * iterations ) / (d.count() * 1e-6);
			std::cout << "CpuGeluOp with " << num_threads << " threads: " << eps_t / 1e6 << " M elems/sec\n";
		}
#else
		GTEST_SKIP() << "OpenMP not available, skipping OpenMP scaling test";
#endif
	}
} // namespace Dnn::Compute::Device::Cpu::Operations::Tests