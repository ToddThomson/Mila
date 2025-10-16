/**
 * @file Module.ixx
 * @brief Defines the base Module class for the Mila deep neural network framework.
 *
 * The Module class provides a unified interface for neural network layers with proper
 * support for abstract tensor data types, scalar tensors, and device-agnostic operations.
 */

module;
#include <string>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <type_traits>
#include <sstream>
#include <format>

export module Dnn.Module;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ConfigurationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CpuExecutionContext;
import Compute.CudaExecutionContext;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.Precision;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Abstract base class for all modules in the Mila DNN framework.
     *
     * The Module class provides a common interface for all neural network layers
     * and components, enabling consistent handling of parameters, state, and
     * execution context. Uses templated ExecutionContext for compile-time device
     * type safety and zero-overhead abstraction.
     *
     * Key architectural features:
     * - Modules own ExecutionContext<TDeviceType> for stream management
     * - ExecutionContext manages streams and library handles for compute operations
     * - Template parameter provides compile-time device type checking
     * - Multiple modules can share the same device but have independent streams
     * - No virtual function overhead for device-specific operations
     *
     * Key features:
     * - Abstract data type system using TensorDataType enumeration
     * - Device-agnostic design (CPU, CUDA)
     * - Support for scalar tensors (bias values, learning rates, etc.)
     * - Type-safe parameter and state management
     * - Serialization support for model persistence
     * - Independent execution streams for proper module isolation
     *
     * @tparam TDeviceType The device type (Cpu or Cuda) on which the module will operate.
     */
    export template<DeviceType TDeviceType>
    class Module 
    {
        public:
            /**
             * @brief Type alias for the device-specific execution context
             */
            using ExecutionContextType = ExecutionContext<TDeviceType>;

            /**
             * @brief Constructor with device ID.
             *
             * Creates a module with an execution context for the specified device ID.
             * Each module gets its own execution context (stream) even if sharing a device.
             *
             * @param device_id The device ID to use (0-based for CUDA, -1 or 0 for CPU).
             * @param config Configuration for the module including training mode and precision.
             * @throws std::invalid_argument If device_id is invalid for the device type.
             * @throws std::runtime_error If execution context creation fails.
             */
            explicit Module( int device_id, const ConfigurationBase& config )
                : execution_context_( std::make_shared<ExecutionContextType>( device_id ) ),
                config_( config ),
                training_mode_( config.isTraining() ) {

                config.validate();
            }

            /**
             * @brief Constructor with a specific execution context.
             *
             * Allows modules to share an execution context (stream) if desired, or for
             * advanced scenarios where the caller manages execution context lifecycle.
             *
             * @param exec_context The execution context to use for this module.
             * @param config Configuration for the module including training mode and precision.
             * @throws std::invalid_argument If the provided context is nullptr.
             */
            explicit Module( std::shared_ptr<ExecutionContextType> exec_context, const ConfigurationBase& config )
                : execution_context_( exec_context ),
                config_( config ),
                training_mode_( config.isTraining() ) {

                if (!exec_context)
                {
                    throw std::invalid_argument(
                        "ExecutionContext cannot be nullptr. Please provide a valid ExecutionContext."
                    );
                }

                config.validate();
            }

            /**
             * @brief Virtual destructor for proper cleanup in derived classes.
             */
            virtual ~Module() = default;

            /**
             * @brief Forward pass of the module.
             *
             * Performs the forward computation of the module, transforming input
             * to output according to the module's learned parameters. Operations
             * use the module's execution context (stream).
             *
             * @param input The input tensor to the module.
             * @param output The output tensor from the module (will be resized if needed).
             *
             * @note Input and output may have different shapes depending on the layer
             * @note Supports scalar tensors for special cases (e.g., loss values)
             * @note All operations execute on this module's stream
             */
            virtual void forward( const ITensor& input, ITensor& output ) = 0;

            /**
             * @brief Backward pass of the module.
             *
             * Computes gradients with respect to inputs given gradients with respect
             * to outputs. Updates parameter gradients internally. Operations use the
             * module's execution context (stream).
             *
             * @param input The input tensor used in the forward pass.
             * @param output_grad The gradient of the loss with respect to the output.
             * @param input_grad The gradient of the loss with respect to the input (computed).
             *
             * @note Only called when training_mode_ is true
             * @note Scalar gradients supported for reduction operations
             * @note All operations execute on this module's stream
             */
            virtual void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) = 0;

            /**
             * @brief Get the execution context for this module.
             *
             * Returns the execution context that manages this module's stream and
             * library handles. TensorOps should receive this context to execute
             * operations on the correct stream.
             *
             * @return std::shared_ptr<ExecutionContext<TDeviceType>> The execution context.
             */
            std::shared_ptr<ExecutionContextType> getExecutionContext() const {
                return execution_context_;
            }

            /**
             * @brief Synchronize this module's execution stream.
             *
             * Blocks until all operations submitted to this module's execution context
             * have completed. Useful for timing, debugging, or ensuring completion
             * before accessing results.
             */
            void synchronize() {
                execution_context_->synchronize();
            }

            /**
             * @brief Set the training mode of this module.
             *
             * Training mode affects behavior of certain layers (e.g., dropout, batch norm).
             * When false, the module operates in inference mode.
             *
             * @param is_training True if the module is in training mode, false for inference.
             */
            virtual void setTraining( bool is_training ) {
                training_mode_ = is_training;
            }

            /**
             * @brief Check if the module is in training mode.
             *
             * @return true If the module is in training mode.
             * @return false If the module is in inference mode.
             */
            bool isTraining() const {
                return training_mode_;
            }

            /**
             * @brief Get the number of trainable parameters in the module.
             *
             * Counts only the parameters in this specific module (not child modules).
             * This includes all learnable weights and biases, including scalar parameters.
             *
             * @return size_t Total number of parameter elements across all parameter tensors.
             *
             * @note Scalar parameters contribute 1 to the count
             * @note Empty tensors contribute 0 to the count
             */
            virtual size_t parameterCount() const = 0;

            /**
             * @brief Get the name of the module.
             *
             * @return std::string Name of the module.
             */
            std::string getName() const {
                return config_.getName();
            }

            /**
             * @brief Get the precision policy for computations.
             *
             * @return const ComputePrecision::Policy& The configured precision policy.
             */
            const ComputePrecision::Policy& getPrecisionPolicy() const {
                return config_.getPrecisionPolicy();
            }

            /**
             * @brief Get the device type of the execution context.
             *
             * @return Compute::DeviceType The device type (CPU or CUDA).
             */
            static constexpr Compute::DeviceType getDeviceType() {
                return TDeviceType;
            }

            /**
             * @brief Get the device name (e.g., "CPU", "CUDA:0").
             *
             * @return std::string The device name.
             */
            std::string getDeviceName() const {
                return execution_context_->getDeviceName();
            }

            /**
             * @brief Get the device ID (-1 for CPU, 0+ for CUDA).
             *
             * @return int The device ID.
             */
            int getDeviceId() const {
                return execution_context_->getDeviceId();
            }

            /**
             * @brief Checks if this module is for a CUDA device.
             */
            static constexpr bool isCudaDevice() {
                return ExecutionContextType::isCudaDevice();
            }

            /**
             * @brief Checks if this module is for a CPU device.
             */
            static constexpr bool isCpuDevice() {
                return ExecutionContextType::isCpuDevice();
            }

            /**
             * @brief Save the module state to a model archive.
             *
             * Serializes the module's parameters and state to the provided archive.
             * This enables model persistence for later reuse. Supports all tensor
             * types including scalars.
             *
             * @param archive The archive to save the state to.
             *
             * @note Scalar parameters are serialized as rank-0 tensors
             * @note Abstract data types are preserved during serialization
             */
            virtual void save( ModelArchive& archive ) const = 0;

            /**
             * @brief Load the module state from a model archive.
             *
             * Deserializes the module's parameters and state from the provided archive.
             * This enables loading pre-trained models for inference or continued training.
             *
             * @param archive The archive to load the state from.
             *
             * @throws std::runtime_error If data types or shapes don't match expected structure
             *
             * @note Validates that loaded data types match expected types
             * @note Scalar parameters are loaded as rank-0 tensors
             */
            virtual void load( ModelArchive& archive ) = 0;

            /**
             * @brief Convert the module to a string representation.
             *
             * This should include relevant information about the module structure,
             * parameters, configuration, and data types for debugging and logging purposes.
             *
             * @return std::string String representation of the module.
             *
             * @note Includes information about scalar vs multi-dimensional parameters
             * @note Shows abstract data type (e.g., FP32, FP16)
             */
            virtual std::string toString() const = 0;

            /**
             * @brief Overload the << operator to print the module information.
             *
             * @param os Output stream.
             * @param module Module to print.
             * @return std::ostream& Reference to the output stream.
             */
            friend std::ostream& operator<<( std::ostream& os, const Module& module ) {
                os << module.toString();
                return os;
            }

        protected:
            /**
             * @brief Validates that a tensor is compatible with this module's device type
             *
             * Checks that the tensor's device type matches the module's compile-time
             * device type requirement, throwing a clear error message if incompatible.
             *
             * @param tensor Tensor to validate
             * @param tensor_name Name for error messages (e.g., "input", "output")
             * @throws std::runtime_error If tensor device type doesn't match module device type
             *
             * @note This is a runtime check that complements compile-time device type enforcement
             */
            void validateTensorDevice( const ITensor& tensor,
                const char* tensor_name = "tensor" ) const {
                if (tensor.getDeviceType() != TDeviceType) {
                    throw std::runtime_error( std::format(
                        "{} device type {} incompatible with {} module",
                        tensor_name,
                        deviceToString( tensor.getDeviceType() ),
                        deviceToString( TDeviceType )
                    ) );
                }
            }

            /**
             * @brief Helper method to convert parameters to string representation.
             *
             * Generates a formatted string showing all parameters, their shapes,
             * data types, and distinguishing between scalars and multi-dimensional tensors.
             *
             * @return std::string String representation of all parameters.
             */
            const std::string parametersToString() const {
                std::ostringstream oss;
                oss << "Parameters:\n";
                oss << "Total parameter count: " << parameterCount() << "\n";
                return oss.str();
            }

            /**
             * @brief Helper method to convert state tensors to string representation.
             *
             * Generates a formatted string showing all state tensors, their shapes,
             * and data types, distinguishing between scalars and multi-dimensional tensors.
             *
             * @return std::string String representation of all state tensors.
             */
            const std::string stateToString() const {
                std::ostringstream oss;
                return oss.str();
            }

        private:
            /**
             * @brief The execution context used for this module's computations.
             *
             * Manages the execution stream and library handles (cuBLAS, cuDNN) for
             * this module. Operations pass this context to TensorOps for execution
             * on the correct stream.
             */
            std::shared_ptr<ExecutionContextType> execution_context_;

            /** @brief Configuration including name, training mode, and precision policy */
            ConfigurationBase config_;

            /** @brief Whether the module is in training mode. Default is false */
            bool training_mode_{ false };
    };

    // ====================================================================
    // Type Aliases for Common Module Types
    // ====================================================================

    /**
     * @brief CPU module type alias
     */
    export template<typename ModuleImpl>
        using CpuModule = ModuleImpl;

    /**
     * @brief CUDA module type alias
     */
    export template<typename ModuleImpl>
        using CudaModule = ModuleImpl;
}