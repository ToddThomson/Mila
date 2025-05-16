/**
 * @file Module.ixx
 * @brief Defines the base Module class for the Mila deep neural network framework.
 */

module;
#include <miniz.h>  
#include <vector>  
#include <string>  
#include <memory>  
#include <unordered_map>
#include <stdexcept>  
#include <type_traits>  
#include <sstream>  
#include <format>  // Added for std::format in error messages

export module Dnn.Module;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Compute.Precision;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Abstract base class for all modules in the Mila DNN framework.
     *
     * The Module class provides a common interface for all neural network layers
     * and components, enabling consistent handling of parameters, state, and
     * device context. For container functionality that supports child modules,
     * use the Block class.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which the module will operate.
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     * @tparam TPrecision Data type used for internal calculations, defaults to TOutput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda,
        typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
        requires ValidFloatTensorTypes<TInput, TOutput>&& ValidTensorType<TInput>&& ValidPrecisionType<TPrecision>
    class Module {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;

        /**
        * @brief Constructor with device name.
        *
        * Creates a module with a device context for the specified device name.
        * This allows modules to be created with a simple string identifier rather
        * than requiring manual construction of a DeviceContext.
        *
        * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
        *        Must be one of the names returned by DeviceRegistry::list_devices().
        * @throws std::runtime_error If the specified device name is invalid or doesn't match TDeviceType.
        */
        explicit Module( const std::string& device_name )
            : device_context_( createContext( device_name ) ) {
            // Verify the created context's device type matches TDeviceType
            if ( device_context_->getDevice()->getDeviceType() != TDeviceType ) {
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
         * @throws std::invalid_argument If the provided context is nullptr.
         * @throws std::runtime_error If the context device type doesn't match TDeviceType.
         */
        explicit Module( std::shared_ptr<DeviceContext> context ) {
            if ( !context ) {
                throw std::invalid_argument( "DeviceContext cannot be nullptr. Please provide a valid DeviceContext." );
            }

            // Verify the provided context's device type matches TDeviceType
            if ( context->getDevice()->getDeviceType() != TDeviceType ) {
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
         * @brief Helper function to get a memory resource for the current device.
         *
         * This method returns a DynamicMemoryResource instance configured for the current device context.
         *
         * @return MR The DynamicMemoryResource appropriate for the current device context.
         */
         /*MR getMemoryResource() const {
             auto device_type = device_context_->getDevice()->getDeviceType();
             return MR( device_type );
         }*/

         /**
          * @brief Set the training mode of the module.
          *
          * Many modules behave differently during training versus inference
          * (e.g., dropout, batch normalization). This method only affects this
          * specific module.
          *
          * @param training True if the module is in training mode, false for inference mode.
          */
        void setTrainingMode( bool training ) {
            is_training_ = training;
        }

        /**
         * @brief Set the training mode of this module.
         *
         * @param is_training True if the module is in training mode, false for inference.
         */
        virtual void setTraining( bool is_training ) {
            is_training_ = is_training;
        }

        /**
         * @brief Check if the module is in training mode.
         *
         * @return true If the module is in training mode.
         * @return false If the module is in inference mode.
         */
        bool isTraining() const {
            return is_training_;
        }

        /**
         * @brief Get the number of trainable parameters in the module.
         *
         * This should count only the parameters in this specific module.
         *
         * @return size_t Total number of parameters.
         */
        virtual size_t parameterCount() const = 0;

        /**
         * @brief Get the parameter tensors of this module.
         *
         * Parameter tensors represent learnable weights that are updated during
         * training via gradient descent or other optimization algorithms.
         *
         * @return const std::unordered_map<std::string, std::shared_ptr<Tensor<TDataType, MR>>>&
         *         Map of parameter names to tensor pointers.
         */
        const auto& getParameterTensors() const {
            return parameter_map_;
        }

        /**
         * @brief Get the state tensors of this module.
         *
         * State tensors represent non-trainable tensors that may be updated during
         * forward/backward passes (e.g., running mean and variance in batch normalization).
         *
         * @tparam TMR Memory resource type (defaults to the module's MR type).
         * @return const std::unordered_map<std::string, std::shared_ptr<Tensor<TDataType, MR>>>&
         *         Map of state names to tensor pointers.
         */
        const auto& getStateTensors() const {
            return state_map_;
        }

        /**
         * @brief Get the name of the module.
         *
         * @return std::string Name of the module.
         */
        std::string getName() const {
            return name_;
        }

        /**
         * @brief Set the name of the module.
         *
         * @param name The name to set. Must not be empty and cannot contain a dot ('.').
         * @throws std::invalid_argument If the name is empty or contains a dot.
         */
        void setName( const std::string& name ) {
            if ( name.empty() ) {
                throw std::invalid_argument( "Name must not be empty and cannot contain a dot ('.')." );
            }
            name_ = name;
        }

        /**
        * @brief Save the module state to a zip archive.
        *
        * Serializes the module's parameters and state to the provided zip archive.
        * This enables model persistence for later reuse.
        *
        * @param zip The zip archive to save the state to.
        */
        virtual void save( mz_zip_archive& zip ) const = 0;

        /**
         * @brief Load the module state from a zip archive.
         *
         * Deserializes the module's parameters and state from the provided zip archive.
         * This enables loading pre-trained models for inference or continued training.
         *
         * @param zip The zip archive to load the state from.
         */
        virtual void load( mz_zip_archive& zip ) = 0;

        /**
         * @brief Convert the module to a string representation.
         *
         * This should include relevant information about the module structure,
         * parameters, and configuration for debugging and logging purposes.
         *
         * @return std::string String representation of the module.
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

        /**
         * @brief Get the device type of the current device context.
         *
         * @return Compute::DeviceType The device type (CPU or CUDA).
         */
        Compute::DeviceType getDeviceType() const {
            return device_context_->getDevice()->getDeviceType();
        }

    protected:
        /** @brief Map of parameter names to parameter tensors */
        std::unordered_map<std::string, std::shared_ptr<Tensor<TPrecision, MR>>> parameter_map_ = {};

        /** @brief Map of state names to state tensors */
        std::unordered_map<std::string, std::shared_ptr<Tensor<TPrecision, MR>>> state_map_ = {};

        /**
         * @brief Helper method to convert parameters to string representation.
         *
         * @return std::string String representation of all parameters.
         */
        const std::string parametersToString() const {
            std::ostringstream oss;
            for ( const auto& [name, tensor] : getParameterTensors() ) {
                oss << tensor->toString();
            }
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

        /**
         * @brief Helper method to convert state tensors to string representation.
         *
         * @return std::string String representation of all state tensors.
         */
        const std::string stateToString() const {
            std::ostringstream oss;
            for ( const auto& [name, tensor] : getStateTensors() ) {
                oss << tensor->toString();
            }
            return oss.str();
        }

    private:
        /** @brief The device context used for this module's computations */
        std::shared_ptr<Compute::DeviceContext> device_context_;

        /** @brief The name of the module. Cannot be empty and cannot contain a dot ('.') */
        std::string name_;

        /** @brief Whether the module is in training mode. Default is false */
        bool is_training_{ false };

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
     * @brief Type alias for CPU-based modules with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     * @tparam TCompute Data type used for internal calculations, defaults to TOutput.
     */
    export template<typename TInput = float, typename TOutput = TInput, typename TCompute = TOutput>
        using CpuModule = Module<DeviceType::Cpu, TInput, TOutput, TCompute>;

    /**
     * @brief Type alias for CUDA-based modules with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     * @tparam TCompute Data type used for internal calculations, defaults to TOutput.
     */
    export template<typename TInput = float, typename TOutput = TInput, typename TCompute = TOutput>
        using CudaModule = Module<DeviceType::Cuda, TInput, TOutput, TCompute>;
}