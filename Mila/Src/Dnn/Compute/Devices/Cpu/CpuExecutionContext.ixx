/**
 * @file CpuExecutionContext.ixx
 * @brief CPU-specific execution context specialization.
 */

module;
#include <memory>
#include <string>
#include <stdexcept>

export module Compute.CpuExecutionContext;

import Compute.ExecutionContext;
import Compute.ComputeDevice;
import Compute.CpuDevice;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU execution context specialization.
     *
     * Minimal execution context for CPU operations. CPU execution does not require
     * streams or library handles, so this implementation provides the interface
     * without additional overhead.
     */
    export template<>
        class ExecutionContext<DeviceType::Cpu> {
        public:
            /**
             * @brief Constructs CPU execution context.
             *
             * @param device_id Ignored for CPU (CPU has no device ID)
             */
            explicit ExecutionContext( int device_id = -1 )
                : device_( std::make_shared<CpuDevice>() ) {
            }

            /**
             * @brief Constructs CPU execution context from device.
             *
             * @param device CPU device instance
             * @throws std::invalid_argument If device is null or not CPU
             */
            explicit ExecutionContext( std::shared_ptr<ComputeDevice> device )
                : device_( device ) {

                if (!device_) {
                    throw std::invalid_argument( "Device cannot be null" );
                }

                if (device_->getDeviceType() != DeviceType::Cpu) {
                    throw std::invalid_argument(
                        "CpuExecutionContext requires CPU device"
                    );
                }
            }

            ~ExecutionContext() = default;

            ExecutionContext( const ExecutionContext& ) = delete;
            ExecutionContext& operator=( const ExecutionContext& ) = delete;
            ExecutionContext( ExecutionContext&& ) noexcept = default;
            ExecutionContext& operator=( ExecutionContext&& ) noexcept = default;

            /**
             * @brief Synchronizes CPU execution (no-op).
             */
            void synchronize() {
                // CPU operations are synchronous
            }

            /**
             * @brief Gets the CPU device.
             */
            std::shared_ptr<ComputeDevice> getDevice() const {
                return device_;
            }

            /**
             * @brief Gets the device type (always CPU).
             */
            static constexpr DeviceType getDeviceType() {
                return DeviceType::Cpu;
            }

            /**
             * @brief Gets the device name.
             */
            std::string getDeviceName() const {
                return device_->getDeviceName();
            }

            /**
             * @brief Gets the device ID (always -1 for CPU).
             */
            int getDeviceId() const {
                return -1;
            }

            /**
             * @brief Checks if this is a CUDA device (always false).
             */
            static constexpr bool isCudaDevice() {
                return false;
            }

            /**
             * @brief Checks if this is a CPU device (always true).
             */
            static constexpr bool isCpuDevice() {
                return true;
            }

        private:
            std::shared_ptr<ComputeDevice> device_;
    };
}