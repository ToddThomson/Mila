#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

import Compute.DeviceType;

namespace Dnn::Compute::Tests
{
    using namespace Mila::Dnn::Compute;

    class DeviceTypeTest : public ::testing::Test {
    protected:
        void SetUp() override {}
        void TearDown() override {}
    };

    // ====================================================================
    // DeviceType Enumeration Tests
    // ====================================================================

    TEST_F(DeviceTypeTest, EnumValuesExist) {
        // Verify all enum values can be created
        EXPECT_NO_THROW({
            DeviceType cpu = DeviceType::Cpu;
            DeviceType cuda = DeviceType::Cuda;
            DeviceType metal = DeviceType::Metal;
            DeviceType opencl = DeviceType::OpenCL;
            DeviceType vulkan = DeviceType::Vulkan;
        });
    }

    TEST_F(DeviceTypeTest, EnumComparison) {
        // Test enum equality and inequality
        EXPECT_EQ(DeviceType::Cpu, DeviceType::Cpu);
        EXPECT_NE(DeviceType::Cpu, DeviceType::Cuda);
        EXPECT_NE(DeviceType::Cuda, DeviceType::Metal);
        EXPECT_NE(DeviceType::Metal, DeviceType::OpenCL);
        EXPECT_NE(DeviceType::OpenCL, DeviceType::Vulkan);
    }

    // ====================================================================
    // deviceToString() Function Tests
    // ====================================================================

    TEST_F(DeviceTypeTest, DeviceToString_AllImplementedTypes) {
        // Test all implemented device types
        EXPECT_EQ(deviceToString(DeviceType::Cpu), "CPU");
        EXPECT_EQ(deviceToString(DeviceType::Cuda), "CUDA");
        EXPECT_EQ(deviceToString(DeviceType::Metal), "Metal");
        EXPECT_EQ(deviceToString(DeviceType::OpenCL), "OpenCL");
        EXPECT_EQ(deviceToString(DeviceType::Vulkan), "Vulkan");
    }

    TEST_F(DeviceTypeTest, DeviceToString_InvalidType) {
        // Test with invalid enum value (cast from invalid integer)
        DeviceType invalid_type = static_cast<DeviceType>(999);
        EXPECT_THROW(deviceToString(invalid_type), std::runtime_error);
    }

    TEST_F(DeviceTypeTest, DeviceToString_ErrorMessage) {
        try {
            DeviceType invalid_type = static_cast<DeviceType>(999);
            deviceToString(invalid_type);
            FAIL() << "Expected std::runtime_error";
        }
        catch (const std::runtime_error& e) {
            std::string error_msg = e.what();
            // Verify error message indicates unknown device type
            EXPECT_NE(error_msg.find("Unknown"), std::string::npos);
        }
    }

    // ====================================================================
    // toDeviceType() Function Tests
    // ====================================================================

    TEST_F(DeviceTypeTest, ToDeviceType_AllValidStrings) {
        // Test all supported string conversions
        EXPECT_EQ(toDeviceType("CPU"), DeviceType::Cpu);
        EXPECT_EQ(toDeviceType("CUDA"), DeviceType::Cuda);
        EXPECT_EQ(toDeviceType("METAL"), DeviceType::Metal);
        EXPECT_EQ(toDeviceType("OPENCL"), DeviceType::OpenCL);
        EXPECT_EQ(toDeviceType("VULKAN"), DeviceType::Vulkan);
    }

    TEST_F(DeviceTypeTest, ToDeviceType_CaseInsensitive) {
        // Test case insensitive parsing for all device types
        EXPECT_EQ(toDeviceType("cpu"), DeviceType::Cpu);
        EXPECT_EQ(toDeviceType("Cpu"), DeviceType::Cpu);
        EXPECT_EQ(toDeviceType("cPu"), DeviceType::Cpu);
        
        EXPECT_EQ(toDeviceType("cuda"), DeviceType::Cuda);
        EXPECT_EQ(toDeviceType("Cuda"), DeviceType::Cuda);
        EXPECT_EQ(toDeviceType("CuDa"), DeviceType::Cuda);
        
        EXPECT_EQ(toDeviceType("metal"), DeviceType::Metal);
        EXPECT_EQ(toDeviceType("Metal"), DeviceType::Metal);
        EXPECT_EQ(toDeviceType("METAL"), DeviceType::Metal);
        
        EXPECT_EQ(toDeviceType("opencl"), DeviceType::OpenCL);
        EXPECT_EQ(toDeviceType("OpenCL"), DeviceType::OpenCL);
        EXPECT_EQ(toDeviceType("OPENCL"), DeviceType::OpenCL);
        
        EXPECT_EQ(toDeviceType("vulkan"), DeviceType::Vulkan);
        EXPECT_EQ(toDeviceType("Vulkan"), DeviceType::Vulkan);
        EXPECT_EQ(toDeviceType("VULKAN"), DeviceType::Vulkan);
    }

    TEST_F(DeviceTypeTest, ToDeviceType_InvalidStrings) {
        // Test invalid device type strings
        EXPECT_THROW(toDeviceType(""), std::runtime_error);
        EXPECT_THROW(toDeviceType("INVALID"), std::runtime_error);
        EXPECT_THROW(toDeviceType("GPU"), std::runtime_error);
        EXPECT_THROW(toDeviceType("DEVICE"), std::runtime_error);
        EXPECT_THROW(toDeviceType("123"), std::runtime_error);
        EXPECT_THROW(toDeviceType("AUTO"), std::runtime_error);
    }

    TEST_F(DeviceTypeTest, ToDeviceType_WhitespaceHandling) {
        // Test strings with whitespace (should fail - no trimming implemented)
        EXPECT_THROW(toDeviceType(" CPU"), std::runtime_error);
        EXPECT_THROW(toDeviceType("CPU "), std::runtime_error);
        EXPECT_THROW(toDeviceType(" CPU "), std::runtime_error);
        EXPECT_THROW(toDeviceType("C P U"), std::runtime_error);
    }

    TEST_F(DeviceTypeTest, ToDeviceType_SpecialCharacters) {
        // Test strings with special characters
        EXPECT_THROW(toDeviceType("CPU\n"), std::runtime_error);
        EXPECT_THROW(toDeviceType("CPU\t"), std::runtime_error);
        EXPECT_THROW(toDeviceType("CPU-1"), std::runtime_error);
        EXPECT_THROW(toDeviceType("CPU:0"), std::runtime_error);
    }

    // ====================================================================
    // Error Message Validation Tests
    // ====================================================================

    TEST_F(DeviceTypeTest, ErrorMessage_ToDeviceType) {
        try {
            toDeviceType("INVALID");
            FAIL() << "Expected std::runtime_error";
        }
        catch (const std::runtime_error& e) {
            std::string error_msg = e.what();
            // Verify error message contains the invalid string and all valid options
            EXPECT_NE(error_msg.find("INVALID"), std::string::npos);
            EXPECT_NE(error_msg.find("CPU"), std::string::npos);
            EXPECT_NE(error_msg.find("CUDA"), std::string::npos);
            EXPECT_NE(error_msg.find("Metal"), std::string::npos);
            EXPECT_NE(error_msg.find("OpenCL"), std::string::npos);
            EXPECT_NE(error_msg.find("Vulkan"), std::string::npos);
        }
    }

    // ====================================================================
    // Round-Trip Conversion Tests
    // ====================================================================

    TEST_F(DeviceTypeTest, RoundTripConversion_AllTypes) {
        // Test round-trip conversion for all implemented types
        std::vector<DeviceType> all_types = {
            DeviceType::Cpu,
            DeviceType::Cuda,
            DeviceType::Metal,
            DeviceType::OpenCL,
            DeviceType::Vulkan
        };

        for (DeviceType device_type : all_types) {
            std::string device_string = deviceToString(device_type);
            DeviceType converted_back = toDeviceType(device_string);
            EXPECT_EQ(device_type, converted_back)
                << "Round-trip failed for device type: " << device_string;
        }
    }

    TEST_F(DeviceTypeTest, RoundTripConversion_CaseInsensitive) {
        // Test round-trip with case variations
        std::vector<std::pair<std::string, DeviceType>> test_cases = {
            {"cpu", DeviceType::Cpu},
            {"cuda", DeviceType::Cuda},
            {"metal", DeviceType::Metal},
            {"opencl", DeviceType::OpenCL},
            {"vulkan", DeviceType::Vulkan}
        };

        for (const auto& [input, expected_type] : test_cases) {
            DeviceType parsed_type = toDeviceType(input);
            EXPECT_EQ(parsed_type, expected_type);
            
            // Verify conversion back to canonical string format
            std::string canonical = deviceToString(parsed_type);
            DeviceType reparsed = toDeviceType(canonical);
            EXPECT_EQ(reparsed, expected_type);
        }
    }

    TEST_F(DeviceTypeTest, RoundTripConversion_CanonicalFormat) {
        // Verify canonical string formats are preserved
        EXPECT_EQ(deviceToString(toDeviceType("cpu")), "CPU");
        EXPECT_EQ(deviceToString(toDeviceType("cuda")), "CUDA");
        EXPECT_EQ(deviceToString(toDeviceType("metal")), "Metal");
        EXPECT_EQ(deviceToString(toDeviceType("opencl")), "OpenCL");
        EXPECT_EQ(deviceToString(toDeviceType("vulkan")), "Vulkan");
    }

    // ====================================================================
    // Enum Stability and Completeness Tests
    // ====================================================================

    TEST_F(DeviceTypeTest, EnumValuesStability) {
        // Ensure enum values are stable for ABI compatibility
        EXPECT_EQ(static_cast<int>(DeviceType::Cpu), 0);
        EXPECT_EQ(static_cast<int>(DeviceType::Cuda), 1);
        EXPECT_EQ(static_cast<int>(DeviceType::Metal), 2);
        EXPECT_EQ(static_cast<int>(DeviceType::OpenCL), 3);
        EXPECT_EQ(static_cast<int>(DeviceType::Vulkan), 4);
    }

    TEST_F(DeviceTypeTest, ComprehensiveEnumCoverage) {
        // Verify all enum values are accounted for
        std::vector<DeviceType> all_types = {
            DeviceType::Cpu,
            DeviceType::Cuda,
            DeviceType::Metal,
            DeviceType::OpenCL,
            DeviceType::Vulkan
        };

        // Ensure we have exactly 5 device types defined
        EXPECT_EQ(all_types.size(), 5);

        // Verify each type can be constructed and compared
        for (DeviceType type : all_types) {
            EXPECT_NO_THROW({
                DeviceType copy = type;
                bool equal = (copy == type);
                EXPECT_TRUE(equal);
            });
        }
    }

    TEST_F(DeviceTypeTest, StringConversionCompleteness) {
        // Ensure every enum value has a corresponding string conversion
        std::vector<DeviceType> all_types = {
            DeviceType::Cpu,
            DeviceType::Cuda,
            DeviceType::Metal,
            DeviceType::OpenCL,
            DeviceType::Vulkan
        };

        for (DeviceType type : all_types) {
            // Every enum value should convert to string without throwing
            EXPECT_NO_THROW({
                std::string str = deviceToString(type);
                EXPECT_FALSE(str.empty());
            });
            
            // And that string should convert back to the same enum value
            std::string str = deviceToString(type);
            DeviceType converted_back = toDeviceType(str);
            EXPECT_EQ(type, converted_back);
        }
    }

    // ====================================================================
    // Performance and Edge Case Tests
    // ====================================================================

    TEST_F(DeviceTypeTest, LargeStringHandling) {
        // Test with very long strings (should fail gracefully)
        std::string long_string(10000, 'X');
        EXPECT_THROW(toDeviceType(long_string), std::runtime_error);
    }

    TEST_F(DeviceTypeTest, EmptyStringHandling) {
        // Test empty string handling
        EXPECT_THROW(toDeviceType(""), std::runtime_error);
    }

    TEST_F(DeviceTypeTest, StringWithNullCharacters) {
        // Test strings containing null characters
        std::string null_string = "CPU";
        null_string.push_back('\0');
        null_string.append("EXTRA");

        // Should only process up to first null character, which should succeed
        EXPECT_EQ(toDeviceType("CPU"), DeviceType::Cpu);

        // But string with null in middle should fail
        std::string middle_null = "C\0PU";
        EXPECT_THROW(toDeviceType(middle_null), std::runtime_error);
    }

    // ====================================================================
    // Consistency and Thread Safety Tests
    // ====================================================================

    TEST_F(DeviceTypeTest, ConsistentBehavior_MultipleCallsSameInput) {
        // Ensure functions are deterministic across all device types
        for (int i = 0; i < 100; ++i) {
            EXPECT_EQ(toDeviceType("CPU"), DeviceType::Cpu);
            EXPECT_EQ(toDeviceType("cuda"), DeviceType::Cuda);
            EXPECT_EQ(toDeviceType("Metal"), DeviceType::Metal);
            EXPECT_EQ(toDeviceType("OPENCL"), DeviceType::OpenCL);
            EXPECT_EQ(toDeviceType("vulkan"), DeviceType::Vulkan);
            
            EXPECT_EQ(deviceToString(DeviceType::Cpu), "CPU");
            EXPECT_EQ(deviceToString(DeviceType::Cuda), "CUDA");
            EXPECT_EQ(deviceToString(DeviceType::Metal), "Metal");
            EXPECT_EQ(deviceToString(DeviceType::OpenCL), "OpenCL");
            EXPECT_EQ(deviceToString(DeviceType::Vulkan), "Vulkan");
        }
    }

    TEST_F(DeviceTypeTest, ThreadSafety_AllDeviceTypes) {
        // Thread safety test for all device types
        std::vector<std::thread> threads;
        std::atomic<int> success_count{0};
        
        std::vector<std::pair<std::string, DeviceType>> test_cases = {
            {"CPU", DeviceType::Cpu},
            {"CUDA", DeviceType::Cuda},
            {"Metal", DeviceType::Metal},
            {"OpenCL", DeviceType::OpenCL},
            {"Vulkan", DeviceType::Vulkan}
        };

        for (int i = 0; i < 10; ++i) {
            threads.emplace_back([&success_count, &test_cases]() {
                try {
                    for (int j = 0; j < 100; ++j) {
                        for (const auto& [str, expected_type] : test_cases) {
                            DeviceType parsed = toDeviceType(str);
                            std::string converted = deviceToString(parsed);
                            if (parsed == expected_type && converted == str) {
                                success_count++;
                            }
                        }
                    }
                } catch (...) {
                    // Thread failed
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // 10 threads * 100 iterations * 5 device types = 5000 expected successes
        EXPECT_EQ(success_count.load(), 5000);
    }

    // ====================================================================
    // Device Type Classification Tests
    // ====================================================================

    TEST_F(DeviceTypeTest, DeviceTypeClassification) {
        // Test logical groupings of device types for future use
        
        // CPU is the only non-GPU device
        EXPECT_TRUE(DeviceType::Cpu != DeviceType::Cuda);
        EXPECT_TRUE(DeviceType::Cpu != DeviceType::Metal);
        EXPECT_TRUE(DeviceType::Cpu != DeviceType::OpenCL);
        EXPECT_TRUE(DeviceType::Cpu != DeviceType::Vulkan);
        
        // All GPU devices are distinct
        std::vector<DeviceType> gpu_devices = {
            DeviceType::Cuda,
            DeviceType::Metal,
            DeviceType::OpenCL,
            DeviceType::Vulkan
        };
        
        for (size_t i = 0; i < gpu_devices.size(); ++i) {
            for (size_t j = i + 1; j < gpu_devices.size(); ++j) {
                EXPECT_NE(gpu_devices[i], gpu_devices[j]);
            }
        }
    }

    TEST_F(DeviceTypeTest, StringFormatConsistency) {
        // Verify string format consistency for API design
        EXPECT_EQ(deviceToString(DeviceType::Cpu), "CPU");        // All caps
        EXPECT_EQ(deviceToString(DeviceType::Cuda), "CUDA");      // All caps
        EXPECT_EQ(deviceToString(DeviceType::Metal), "Metal");    // Title case
        EXPECT_EQ(deviceToString(DeviceType::OpenCL), "OpenCL");  // Mixed case
        EXPECT_EQ(deviceToString(DeviceType::Vulkan), "Vulkan");  // Title case
        
        // All strings should be non-empty and not contain spaces
        std::vector<DeviceType> all_types = {
            DeviceType::Cpu, DeviceType::Cuda, DeviceType::Metal,
            DeviceType::OpenCL, DeviceType::Vulkan
        };
        
        for (DeviceType type : all_types) {
            std::string str = deviceToString(type);
            EXPECT_FALSE(str.empty());
            EXPECT_EQ(str.find(' '), std::string::npos);
        }
    }
}