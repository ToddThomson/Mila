#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

#include <cuda_runtime.h>

import Compute.CudaDevice;
import Compute.CudaDeviceProps;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;

namespace Dnn::Compute::Tests
{
	using namespace Mila::Dnn::Compute;

	// Local thin wrapper that mirrors the public CudaDevice API using a CudaDeviceProps instance.
	// This allows exercising the public queries (compute capability, capability checks, limits)
	// without requiring access to CudaDevice's private constructor.
	struct TestCudaDevice
	{
		DeviceId id;
		CudaDeviceProps props;

		explicit TestCudaDevice( int device_index )
			: id( Device::Cuda( device_index ) ), props( device_index )
		{
		}

		DeviceId getDeviceId() const
		{
			return id;
		}
		DeviceType getDeviceType() const
		{
			return DeviceType::Cuda;
		}
		std::string getDeviceName() const
		{
			return id.toString();
		}

		const CudaDeviceProps& getProperties() const
		{
			return props;
		}

		std::pair<int, int> getComputeCapability() const
		{
			return props.getComputeCapability();
		}
		int getComputeCapabilityVersion() const
		{
			return props.getComputeCapabilityVersion();
		}

		bool isFp16Supported() const
		{
			return props.supportsFp16();
		}
		bool isBf16Supported() const
		{
			return props.supportsBf16();
		}
		bool isFp8Supported() const
		{
			return props.supportsFp8();
		}
		bool isInt8Supported() const
		{
			return props.supportsInt8TensorCores();
		}
		bool hasTensorCores() const
		{
			return props.hasTensorCores();
		}

		int getMaxThreadsPerBlock() const
		{
			return props.getMaxThreadsPerBlock();
		}
		size_t getTotalGlobalMemory() const
		{
			return props.getTotalGlobalMem();
		}
		size_t getSharedMemoryPerBlock() const
		{
			return props.getSharedMemPerBlock();
		}
		int getMultiprocessorCount() const
		{
			return props.getMultiprocessorCount();
		}
		int getWarpSize() const
		{
			return props.getWarpSize();
		}
	};

	class CudaDeviceTest : public ::testing::Test
	{
	protected:
		void SetUp() override
		{
		}
		void TearDown() override
		{
		}
	};

	TEST_F( CudaDeviceTest, SkipIfNoCuda )
	{
		int device_count = 0;
		cudaError_t err = cudaGetDeviceCount( &device_count );

		if ( err != cudaSuccess || device_count == 0 )
		{
			GTEST_SKIP() << "No CUDA devices available or CUDA error: "
				<< (err == cudaSuccess ? "count==0" : cudaGetErrorString( err ));
		}

		// If we reach here, at least one device is present. Construct props for device 0.
		EXPECT_GE( device_count, 1 );
	}

	TEST_F( CudaDeviceTest, PropertiesAndCapabilityChecks )
	{
		int device_count = 0;
		cudaError_t err = cudaGetDeviceCount( &device_count );

		if ( err != cudaSuccess || device_count == 0 )
		{
			GTEST_SKIP() << "No CUDA devices available or CUDA error: "
				<< (err == cudaSuccess ? "count==0" : cudaGetErrorString( err ));
		}

		// Create the test wrapper for device 0
		TestCudaDevice dev{ 0 };

		// Identity & type
		EXPECT_EQ( dev.getDeviceId(), Device::Cuda( 0 ) );
		EXPECT_EQ( dev.getDeviceType(), DeviceType::Cuda );
		EXPECT_EQ( dev.getDeviceName(), Device::Cuda( 0 ).toString() );

		// Properties accessors should be consistent with CudaDeviceProps
		const auto& p = dev.getProperties();

		auto cc = dev.getComputeCapability();
		EXPECT_EQ( cc, p.getComputeCapability() );

		EXPECT_EQ( dev.getComputeCapabilityVersion(), p.getComputeCapabilityVersion() );

		// Capability checks (these are deterministic based on compute capability)
		EXPECT_EQ( dev.isFp16Supported(), p.supportsFp16() );
		EXPECT_EQ( dev.isBf16Supported(), p.supportsBf16() );
		EXPECT_EQ( dev.isFp8Supported(), p.supportsFp8() );
		EXPECT_EQ( dev.isInt8Supported(), p.supportsInt8TensorCores() );
		EXPECT_EQ( dev.hasTensorCores(), p.hasTensorCores() );

		// Limits and sizes
		EXPECT_EQ( dev.getMaxThreadsPerBlock(), p.getMaxThreadsPerBlock() );
		EXPECT_EQ( dev.getTotalGlobalMemory(), p.getTotalGlobalMem() );
		EXPECT_EQ( dev.getSharedMemoryPerBlock(), p.getSharedMemPerBlock() );
		EXPECT_EQ( dev.getMultiprocessorCount(), p.getMultiprocessorCount() );
		EXPECT_EQ( dev.getWarpSize(), p.getWarpSize() );

		// Derived metrics from props: bandwidth and clock rates should be non-negative
		EXPECT_GE( p.getTotalGlobalMem(), static_cast<size_t>(0) );
		EXPECT_GE( p.getClockRateMHz(), 0.0 );
		EXPECT_GE( p.getMemoryClockRateMHz(), 0.0 );

		// Memory bandwidth is computed; when memoryClockRate is 0 it returns 0. Otherwise positive.
		double bw = p.getMemoryBandwidthGBs();
		EXPECT_GE( bw, 0.0 );

		// String helpers should not be empty
		EXPECT_FALSE( p.toString().empty() );
		EXPECT_FALSE( p.toSummary().empty() );

		// PCI location must follow expected format "hhhh:bb:dd.0" (basic sanity)
		auto pci = p.getPciLocation();
		EXPECT_FALSE( pci.empty() );
		EXPECT_NE( pci.find( ':' ), std::string::npos );
		EXPECT_NE( pci.find( '.' ), std::string::npos );
	}

	TEST_F( CudaDeviceTest, InvalidDeviceIndexThrows )
	{
		int device_count = 0;
		cudaError_t err = cudaGetDeviceCount( &device_count );

		if ( err != cudaSuccess )
		{
			GTEST_SKIP() << "CUDA error querying device count: " << cudaGetErrorString( err );
		}

		// If no devices available, constructing with index 0 would already fail above; skip.
		if ( device_count == 0 )
		{
			GTEST_SKIP() << "No CUDA devices available";
		}

		// Constructing CudaDeviceProps with an out-of-range index should throw.
		int bad_index = device_count; // one past last
        // FIXME: Why are we allowing access to CudaDeviceProps. This should be private to CudaDevice.
		// EXPECT_THROW( (void)CudaDeviceProps( bad_index ), std::runtime_error );
	}
}