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
import Dnn.TensorTypeTraits;
import Dnn.ConfigurationBase;
import Compute.DeviceType;
import Compute.DeviceContext;
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
     * device context. Uses abstract TensorDataType enumeration for type-safe
     * operations across different precision formats.
     *
     * Key features:
     * - Abstract data type system using TensorDataType enumeration
     * - Device-agnostic design (CPU, CUDA)
     * - Support for scalar tensors (bias values, learning rates, etc.)
     * - Type-safe parameter and state management
     * - Serialization support for model persistence
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which the module will operate.
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda>
    class Module {
        public:

            //using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
            //using TensorType = Tensor<TDataType, MR>;

            /**
            * @brief Constructor with device name.
            *
            * Creates a module with a device context for the specified device name.
            * This allows modules to be created with a simple string identifier rather
            * than requiring manual construction of a DeviceContext.
            *
            * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
            *        Must be one of the names returned by DeviceRegistry::list_devices().
            * @param config Configuration for the module including training mode and precision.
            * @throws std::runtime_error If the specified device name is invalid or doesn't match TDeviceType.
            */
            explicit Module( const std::string& device_name, const ConfigurationBase& config )
                : device_context_( createContext( device_name ) ), config_( config ), training_mode_( config.isTraining() ) {

                config.validate();

                if (device_context_->getDevice()->getDeviceType() != TDeviceType) {
                    throw std::runtime_error( std::format(
                        "Device type mismatch: Module template requires {} but device name '{}' corresponds to {}",
                        deviceToString( TDeviceType ),
                        device_name,
                        deviceToString( device_context_->getDevice()->getDeviceType() )
                    ) );
                }
            }

            /**
                * @brief Constructor with a specific device context.
                *
                * @param context The device context to use for this module.
                * @param config Configuration for the module including training mode and precision.
                * @throws std::invalid_argument If the provided context is nullptr.
                * @throws std::runtime_error If the context device type doesn't match TDeviceType.
                */
            explicit Module( std::shared_ptr<DeviceContext> context, const ConfigurationBase& config )
                : config_( config ), training_mode_( config.isTraining() ) {
                if (!context) {
                    throw std::invalid_argument( "DeviceContext cannot be nullptr. Please provide a valid DeviceContext." );
                }

                if (context->getDevice()->getDeviceType() != TDeviceType) {
                    throw std::runtime_error( std::format(
                        "Device type mismatch: Module template requires {} but provided context is for {}",
                        deviceToString( TDeviceType ),
                        deviceToString( context->getDevice()->getDeviceType() )
                    ) );
                }

                device_context_ = context;
            }

            /**
                * @brief Virtual destructor for proper cleanup in derived classes.
                */
            virtual ~Module() = default;

            /**
                * @brief Forward pass of the module.
                *
                * Performs the forward computation of the module, transforming input
                * to output according to the module's learned parameters.
                *
                * @param input The input tensor to the module.
                * @param output The output tensor from the module (will be resized if needed).
                *
                * @note Input and output may have different shapes depending on the layer
                * @note Supports scalar tensors for special cases (e.g., loss values)
                */
            virtual void forward( const ITensor& input, ITensor& output ) = 0;

            /**
                * @brief Backward pass of the module.
                *
                * Computes gradients with respect to inputs given gradients with respect
                * to outputs. Updates parameter gradients internally.
                *
                * @param input The input tensor used in the forward pass.
                * @param output_grad The gradient of the loss with respect to the output.
                * @param input_grad The gradient of the loss with respect to the input (computed).
                *
                * @note Only called when training_mode_ is true
                * @note Scalar gradients supported for reduction operations
                */
            virtual void backward(
                const ITensor& input,
                const ITensor& output_grad,
                ITensor& input_grad ) = 0;

            /**
                * @brief Get the device context for this module.
                *
                * Returns a shared pointer to the device context that this module is currently using.
                *
                * @return std::shared_ptr<Compute::DeviceContext> The device context.
                */
            std::shared_ptr<Compute::DeviceContext> getDeviceContext() const {
                return device_context_;
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
            * @brief Get the parameter tensors of this module.
            *
            * Parameter tensors represent learnable weights that are updated during
            * training via gradient descent or other optimization algorithms.
            * This includes weights, biases (which may be scalars), and other trainable values.
            *
            * @return const std::unordered_map<std::string, std::shared_ptr<TensorType>>&
            *         Map of parameter names to tensor pointers.
            *
            * @note Parameters may include scalar tensors (rank 0)
            * @note All parameters use the module's TDataType and MR
            */
            /*const auto& getParameterTensors() const {
                return parameter_map_;
            }*/

            /**
            * @brief Get the state tensors of this module.
            *
            * State tensors represent non-trainable tensors that may be updated during
            * forward/backward passes (e.g., running mean and variance in batch normalization,
            * iteration counters, momentum buffers).
            *
            * @return const std::unordered_map<std::string, std::shared_ptr<TensorType>>&
            *         Map of state names to tensor pointers.
            *
            * @note State tensors may include scalars (e.g., iteration count)
            * @note State tensors are not updated by gradient descent
            */
            /*const auto& getStateTensors() const {
                return state_map_;
            }*/

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
                * @brief Get the device type of the current device context.
                *
                * @return Compute::DeviceType The device type (CPU or CUDA).
                */
            Compute::DeviceType getDeviceType() const {
                return device_context_->getDevice()->getDeviceType();
            }

            /**
            * @brief Get the abstract data type used by this module.
            *
            * @return TensorDataType The abstract data type enumeration value.
            */
            /*static constexpr TensorDataType getDataType() {
                return TDataType;
            }*/

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
            * @note Validates that loaded data types match TDataType
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
            /** @brief Map of parameter names to parameter tensors (may include scalars) */
            // FIXME: std::unordered_map<std::string, std::shared_ptr<TensorType>> parameter_map_ = {};

            /** @brief Map of state names to state tensors (may include scalars) */
            // FIXME: std::unordered_map<std::string, std::shared_ptr<TensorType>> state_map_ = {};

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
                /*for (const auto& [name, tensor] : getParameterTensors()) {
                    oss << "  " << name;
                    if (tensor->isScalar()) {
                        oss << " (scalar): ";
                    }
                    else {
                        oss << " (shape=[";
                        const auto& shape = tensor->shape();
                        for (size_t i = 0; i < shape.size(); ++i) {
                            oss << shape[i];
                            if (i < shape.size() - 1) oss << ", ";
                        }
                        oss << "]): ";
                    }
                    oss << "size=" << tensor->size()
                        << ", dtype=" << tensor->getDataTypeName() << "\n";
                }*/
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
                /*if (!state_map_.empty()) {
                    oss << "State tensors:\n";
                    for (const auto& [name, tensor] : getStateTensors()) {
                        oss << "  " << name;
                        if (tensor->isScalar()) {
                            oss << " (scalar): ";
                        }
                        else {
                            oss << " (shape=[";
                            const auto& shape = tensor->shape();
                            for (size_t i = 0; i < shape.size(); ++i) {
                                oss << shape[i];
                                if (i < shape.size() - 1) oss << ", ";
                            }
                            oss << "]): ";
                        }
                        oss << "size=" << tensor->size()
                            << ", dtype=" << tensor->getDataTypeName() << "\n";
                    }
                }*/
                return oss.str();
            }

        private:
            /** @brief The device context used for this module's computations */
            std::shared_ptr<Compute::DeviceContext> device_context_;

            /** @brief Configuration including name, training mode, and precision policy */
            ConfigurationBase config_;

            /** @brief Whether the module is in training mode. Default is false */
            bool training_mode_{ false };

            /**
                * @brief Helper method to create a DeviceContext from a device name.
                *
                * @param device_name Name of the device to create a context for.
                * @return std::shared_ptr<DeviceContext> The created device context.
                * @throws std::runtime_error If the device name is invalid.
                */
            static std::shared_ptr<Compute::DeviceContext> createContext( const std::string& device_name ) {
                return std::make_shared<Compute::DeviceContext>( device_name );
            }
    };

    /**
     * @brief Type alias for CPU-based modules with customizable data type.
     *
     * Convenient alias for creating CPU modules with specific precision requirements.
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration (default: FP32).
     *
     * Example:
     * @code
     * // FP32 CPU module
     * CpuModule<TensorDataType::FP32> layer("CPU", config);
     *
     * // FP16 CPU module for memory efficiency
     * CpuModule<TensorDataType::FP16> layer("CPU", config);
     * @endcode
     */
    /*export template<TensorDataType TDataType = TensorDataType::FP32>
        using CpuModule = Module<DeviceType::Cpu, TDataType>;*/

    /**
     * @brief Type alias for CUDA-based modules with customizable data type.
     *
     * Convenient alias for creating CUDA modules with specific precision requirements.
     *
     * @tparam TDataType Abstract tensor data type from TensorDataType enumeration (default: FP32).
     *
     * Example:
     * @code
     * // FP32 CUDA module
     * CudaModule<TensorDataType::FP32> layer("CUDA:0", config);
     *
     * // FP16 CUDA module for faster training
     * CudaModule<TensorDataType::FP16> layer("CUDA:0", config);
     * @endcode
     */
    /*export template<TensorDataType TDataType = TensorDataType::FP32>
        using CudaModule = Module<DeviceType::Cuda, TDataType>;*/
}