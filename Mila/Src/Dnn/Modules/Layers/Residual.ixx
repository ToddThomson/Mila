/**
 * @file Residual.ixx
 * @brief Implementation of the Residual connection module for neural networks.
 *
 * Provides a flexible implementation of residual connections which can be configured
 * with different connection types and scaling factors. Supports automatic dimension
 * matching via projection layers.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <stdexcept>

export module Dnn.Modules.Residual;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;
import Compute.Precision;
import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Serialization.ModelArchive;

import Dnn.Modules.Linear;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
	using namespace Mila::Dnn::Serialization;

    /**
     * @brief A class implementing a residual connection module.
     *
     * Residual connections help deep neural networks avoid vanishing gradients by
     * providing shortcut connections. The basic formula is y = x + F(x), where F
     * is a differentiable function (usually a sequence of neural network layers).
     *
     * This implementation supports three types of residual connections:
     * 1. Addition: y = x + F(x)
     * 2. Scaled Addition: y = x + alpha*F(x), where alpha is a scaling factor
     * 3. Gated: y = g*x + (1-g)*F(x), where g is a learnable parameter
     *
     * When input and output dimensions don't match, an optional projection layer
     * can be automatically added to make the dimensions compatible.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidFloatTensorTypes<TInput, TOutput>
    class Residual : public Module<TDeviceType, TInput, TOutput> {
    public:
        /**
         * @brief Memory resource type used for tensors, selected based on device type.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;

        /**
         * @brief Alias for base module type.
         */
        using ModuleBase = Module<TDeviceType, TInput, TOutput>;

        /**
         * @brief Constructs a new Residual module with a device name.
         *
         * Creates a new DeviceContext internally using the provided device name.
         * This constructor is useful for creating standalone modules without
         * pre-existing device contexts.
         *
         * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
         * @param config Configuration parameters for the Residual module.
         * @throws std::invalid_argument If the device name is invalid or the configuration is invalid
         * @throws std::runtime_error If device type doesn't match template parameter TDeviceType or inner module type mismatch
         */
        explicit Residual( const std::string& device_name, const ResidualConfig& config )
            : ModuleBase( std::make_shared<DeviceContext>( device_name ), config ), config_( config ) {

            config.validate();

            inner_module_ = config.getInnerModule<TDeviceType, TInput, TOutput>();
            createOperation();
        }

        /**
         * @brief Constructs a new Residual module with a provided device context.
         *
         * Uses a pre-existing DeviceContext instance. This constructor is useful when integrating
         * the module into a larger network that shares device contexts across modules.
         *
         * @param device_context The device context to use for this module.
         * @param config Configuration parameters for the Residual module.
         * @throws std::invalid_argument If device_context is null or configuration is invalid
         * @throws std::runtime_error If device context type doesn't match template parameter TDeviceType or inner module type mismatch
         */
        explicit Residual( std::shared_ptr<DeviceContext> device_context, const ResidualConfig& config )
            : ModuleBase( device_context, config ), config_( config ) {

            config.validate();

            inner_module_ = config.getInnerModule<TDeviceType, TInput, TOutput>();
            createOperation();
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * Counts the total number of trainable parameters in the residual module,
         * including the inner module, projection layer (if present), and gating
         * parameters (if using gated connections).
         *
         * @return size_t The total number of parameters.
         */
        size_t parameterCount() const override {
            size_t count = inner_module_->parameterCount();

            if ( projection_ ) {
                count += projection_->parameterCount();
            }

            if ( config_.getConnectionType() == ResidualConfig::ConnectionType::Gated ) {
                count += gate_weights_->size();
            }

            return count;
        }

        /**
         * @brief Performs the forward pass of the Residual connection.
         *
         * Applies the residual transformation based on the configured connection type:
         * - Addition: y = x + F(x)
         * - Scaled Addition: y = x + alpha*F(x)
         * - Gated: y = g*x + (1-g)*F(x)
         *
         * Handles projection when input and inner module dimensions don't match.
         *
         * @param input The input tensor to be processed.
         * @param output The output tensor where the results will be stored.
         * @throws std::runtime_error If dimensions don't match and projection is disabled.
         */
        void forward( const Tensor<TInput, MR>& input, Tensor<TOutput, MR>& output ) {
            inner_module_->forward( input, inner_output_ );

            bool input_dims_match = tensorShapesMatch( input, inner_output_ );

            if ( !input_dims_match && !config_.useProjection() ) {
                throw std::runtime_error( "Input and inner module output dimensions don't match, and projection is disabled" );
            }

            if ( !input_dims_match && !projection_ ) {
                createProjection( input.getShape(), inner_output_.getShape() );
            }

            switch ( config_.getConnectionType() ) {
                case ResidualConfig::ConnectionType::Addition:
                    if ( input_dims_match ) {
                        // y = x + F(x)
                        operation_->forward( input, inner_output_, properties_, output, output_state_ );
                    }
                    else {
                        // y = Wx + F(x), where W is projection matrix
                        projection_->forward( input, projection_output_ );
                        operation_->forward( projection_output_, inner_output_, properties_, output, output_state_ );
                    }
                    break;

                case ResidualConfig::ConnectionType::ScaledAddition:
                    // FIXME: properties_[ "scaling_factor" ] = config_.getScalingFactor();

                    if ( input_dims_match ) {
                        // y = x + alpha*F(x)
                        operation_->forward( input, inner_output_, properties_, output, output_state_ );
                    }
                    else {
                        // y = Wx + alpha*F(x)
                        projection_->forward( input, projection_output_ );
                        operation_->forward( projection_output_, inner_output_, properties_, output, output_state_ );
                    }
                    break;

                case ResidualConfig::ConnectionType::Gated:
                    if ( !gate_weights_ ) {
                        initializeGateParameters( inner_output_.getShape() );
                    }

                    // Compute gated connection: y = g*x + (1-g)*F(x)
                    if ( input_dims_match ) {
                        gated_operation_->forward( input, inner_output_, parameters_, output, output_state_ );
                    }
                    else {
                        projection_->forward( input, projection_output_ );
                        gated_operation_->forward( projection_output_, inner_output_, parameters_, output, output_state_ );
                    }
                    break;
            }
        }

        /**
         * @brief Performs the backward pass of the Residual connection.
         *
         * Computes gradients for the input tensor and parameters based on the output gradients.
         * Handles backpropagation through the inner module and projection layer (if present).
         *
         * @param input The input tensor from the forward pass.
         * @param output_grad The gradient of loss with respect to the output.
         * @param input_grad The tensor to store gradients with respect to input.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TOutput, MR>& output_grad,
            Tensor<TInput, MR>& input_grad ) {

            bool input_dims_match = tensorShapesMatch( input, inner_output_ );

            if ( config_.getConnectionType() != ResidualConfig::ConnectionType::Gated ) {
                // For standard addition and scaled addition
                if ( input_dims_match ) {
                    // Gradient flows both directly to input_grad and through inner module
                    output_grad.copyTo( input_grad );

                    // Also compute gradients for inner module
                    inner_module_->backward( input, output_grad, inner_input_grad_ );

                    // Add gradients from inner module
                    addTensors( input_grad, inner_input_grad_, input_grad );
                }
                else {
                    // Gradient flows through projection
                    Tensor<TOutput, MR> projection_grad( projection_output_.getShape() );
                    operation_->backward(
                        projection_output_,
                        inner_output_,
                        parameters_,
                        inner_parameter_grads_,
                        output_grad,
                        projection_grad,
                        inner_input_grad_,
                        properties_,
                        output_state_
                    );

                    // Backpropagate through projection
                    projection_->backward( input, projection_grad, input_grad );

                    // Backpropagate through inner module
                    inner_module_->backward( input, output_grad, inner_input_grad_ );

                    // Add gradients from inner module
                    addTensors( input_grad, inner_input_grad_, input_grad );
                }
            }
            else {
                // For gated connection, use the gated_operation_->backward
                if ( input_dims_match ) {
                    gated_operation_->backward(
                        input,
                        inner_output_,
                        parameters_,
                        parameter_grads_,
                        output_grad,
                        input_grad,
                        inner_input_grad_,
                        properties_,
                        output_state_
                    );

                    // Backpropagate through inner module
                    inner_module_->backward( input, inner_input_grad_, temp_grad_ );

                    // Add gradients
                    addTensors( input_grad, temp_grad_, input_grad );
                }
                else {
                    Tensor<TOutput, MR> projection_grad( projection_output_.getShape() );

                    gated_operation_->backward(
                        projection_output_,
                        inner_output_,
                        parameters_,
                        parameter_grads_,
                        output_grad,
                        projection_grad,
                        inner_input_grad_,
                        properties_,
                        output_state_
                    );

                    // Backpropagate through projection
                    projection_->backward( input, projection_grad, input_grad );

                    // Backpropagate through inner module
                    inner_module_->backward( input, inner_input_grad_, temp_grad_ );

                    // Add gradients
                    addTensors( input_grad, temp_grad_, input_grad );
                }
            }
        }

        /**
         * @brief Gets the inner module.
         *
         * Returns the inner module that implements the transformation F(x) in the
         * residual connection formula y = x + F(x).
         *
         * @return std::shared_ptr<Module<TDeviceType, TInput, TOutput>> The inner module
         */
        std::shared_ptr<Module<TDeviceType, TInput, TOutput>> getInnerModule() {
            return inner_module_;
        }

        /**
         * @brief Serializes the module state to a ZIP archive.
         *
         * Saves the state of the inner module, projection layer (if present),
         * and gating parameters (if used) to the provided ZIP archive.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( ModelArchive& zip ) const override {
            // TODO:
        }

        /**
         * @brief Deserializes the module state from a ZIP archive.
         *
         * Loads the state of the inner module, projection layer (if present),
         * and gating parameters (if used) from the provided ZIP archive.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( ModelArchive& archive ) override {
            inner_module_->load( archive );

            if ( projection_ ) {
                projection_->load( archive );
            }

            if ( config_.getConnectionType() == ResidualConfig::ConnectionType::Gated && gate_weights_ ) {
                // Load gate weights
                // Implementation depends on tensor deserialization
            }
        }

        /**
         * @brief Converts the module information to a human-readable string.
         *
         * Includes detailed information about the module configuration including:
         * - Module name
         * - Connection type
         * - Scaling factor (for scaled addition)
         * - Projection status
         * - Inner module information
         *
         * @return std::string A string representation of the module information.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Residual: " << this->getName() << std::endl;
            oss << "Connection Type: " << connectionTypeToString( config_.getConnectionType() ) << std::endl;

            if ( config_.getConnectionType() == ResidualConfig::ConnectionType::ScaledAddition ) {
                oss << "Scaling Factor: " << config_.getScalingFactor() << std::endl;
            }

            oss << "Has Projection: " << (projection_ ? "Yes" : "No") << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;
            oss << "Inner Module: " << std::endl;
            oss << inner_module_->toString() << std::endl;
            oss << "Parameter count: " << this->parameterCount() << std::endl;

            return oss.str();
        }

    private:
        /**
         * @brief Configuration for the Residual module.
         */
        ResidualConfig config_;

        /**
         * @brief The inner module implementing the transformation F(x).
         */
        std::shared_ptr<Module<TDeviceType, TInput, TOutput>> inner_module_;

        /**
         * @brief Optional projection layer for dimension matching.
         */
        std::shared_ptr<Linear<TDeviceType, TInput, TOutput>> projection_;

        /**
         * @brief Learnable gate weights for gated residual connections.
         */
        std::shared_ptr<Tensor<TOutput, MR>> gate_weights_;

        /**
         * @brief Collection of trainable parameters.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameters_;

        /**
         * @brief Gradients for trainable parameters.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameter_grads_;

        /**
         * @brief Gradients for inner parameters.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> inner_parameter_grads_;

        /**
         * @brief Operation-specific attributes.
         */
        OperationAttributes properties_;

        /**
         * @brief Output state tensors for backward pass.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> output_state_;

        /**
         * @brief Temporary tensor to store inner module output during forward pass.
         */
        Tensor<TOutput, MR> inner_output_{};

        /**
         * @brief Temporary tensor to store projection output during forward pass.
         */
        Tensor<TOutput, MR> projection_output_{};

        /**
         * @brief Temporary tensor to store inner module gradients during backward pass.
         */
        Tensor<TInput, MR> inner_input_grad_{};

        /**
         * @brief Temporary tensor for gradient accumulation.
         */
        Tensor<TInput, MR> temp_grad_{};

        /**
         * @brief Binary operation for standard and scaled residual connections.
         */
        std::shared_ptr<BinaryOperation<TDeviceType, TInput, TOutput, TOutput>> operation_;

        /**
         * @brief Binary operation for gated residual connections.
         */
        std::shared_ptr<BinaryOperation<TDeviceType, TInput, TOutput, TOutput>> gated_operation_;

        /**
         * @brief Converts connection type enum to string for display purposes.
         *
         * @param type The connection type enum value
         * @return std::string Human-readable representation of the connection type
         */
        static std::string connectionTypeToString( ResidualConfig::ConnectionType type ) {
            switch ( type ) {
                case ResidualConfig::ConnectionType::Addition:
                    return "Addition";
                case ResidualConfig::ConnectionType::ScaledAddition:
                    return "Scaled Addition";
                case ResidualConfig::ConnectionType::Gated:
                    return "Gated";
                default:
                    return "Unknown";
            }
        }

        /**
         * @brief Checks if two tensor shapes match for residual connection.
         *
         * @param a First tensor to compare
         * @param b Second tensor to compare
         * @return bool True if shapes match, false otherwise
         */
        bool tensorShapesMatch( const Tensor<TInput, MR>& a, const Tensor<TOutput, MR>& b ) {
            if ( a.getRank() != b.getRank() ) {
                return false;
            }

            auto a_shape = a.getShape();
            auto b_shape = b.getShape();

            for ( size_t i = 0; i < a_shape.size(); ++i ) {
                if ( a_shape[ i ] != b_shape[ i ] ) {
                    return false;
                }
            }

            return true;
        }

        /**
         * @brief Creates an appropriate operation based on the connection type.
         *
         * Instantiates the correct operation implementation based on the configured
         * connection type (Addition, ScaledAddition, or Gated) and device type.
         */
        void createOperation() {
            if ( config_.getConnectionType() != ResidualConfig::ConnectionType::Gated ) {
                std::string op_name;

                if ( config_.getConnectionType() == ResidualConfig::ConnectionType::Addition ) {
                    op_name = "ResidualAdd";
                }
                else {
                    op_name = "ScaledResidualAdd";
                }

                if constexpr ( TDeviceType == DeviceType::Cpu ) {
                    auto base_op = OperationRegistry::instance().createBinaryOperation<DeviceType::Cpu, TInput, TOutput, TOutput>(
                        "Cpu::" + op_name,
                        this->getDeviceContext(),
                        config_ );

                    operation_ = std::static_pointer_cast<BinaryOperation<DeviceType::Cpu, TInput, TOutput, TOutput>>(base_op);
                }
                else {
                    auto base_op = OperationRegistry::instance().createBinaryOperation<DeviceType::Cuda, TInput, TOutput, TOutput>(
                        "Cuda::" + op_name,
                        this->getDeviceContext(),
                        config_ );

                    operation_ = std::static_pointer_cast<BinaryOperation<DeviceType::Cuda, TInput, TOutput, TOutput>>(base_op);
                }
            }
            else {
                if constexpr ( TDeviceType == DeviceType::Cpu ) {
                    auto base_op = OperationRegistry::instance().createBinaryOperation<DeviceType::Cpu, TInput, TOutput, TOutput>(
                        "Cpu::GatedResidual",
                        this->getDeviceContext(),
                        config_ );

                    gated_operation_ = std::static_pointer_cast<BinaryOperation<DeviceType::Cpu, TInput, TOutput, TOutput>>(base_op);
                }
                else {
                    auto base_op = OperationRegistry::instance().createBinaryOperation<DeviceType::Cuda, TInput, TOutput, TOutput>(
                        "Cuda::GatedResidual",
                        this->getDeviceContext(),
                        config_ );

                    gated_operation_ = std::static_pointer_cast<BinaryOperation<DeviceType::Cuda, TInput, TOutput, TOutput>>(base_op);
                }
            }
        }

        /**
         * @brief Creates a projection layer when input and output dimensions don't match.
         *
         * Instantiates a Linear layer to project the input to the correct dimensions
         * to match the output of the inner module.
         *
         * @param input_shape Shape of the input tensor
         * @param output_shape Shape of the output tensor from inner module
         */
        void createProjection( const std::vector<size_t>& input_shape, const std::vector<size_t>& output_shape ) {
            size_t input_features = input_shape.back();
            size_t output_features = output_shape.back();

            auto projection_config = LinearConfig( input_features, output_features )
                .withName( this->getName() + ".projection" )
                .withTraining( this->isTraining() );

            projection_ = std::make_shared<Linear<TDeviceType, TInput, TOutput>>(
                this->getDeviceContext(),
                projection_config );

            projection_output_ = Tensor<TOutput, MR>( output_shape );
        }

        /**
         * @brief Initializes parameters for gated connections.
         *
         * Creates and initializes the learnable gate weights for gated residual connections.
         * The gate weights determine how much of the input vs. transformed output to use.
         *
         * @param shape Shape of the tensor for gate weights
         */
        void initializeGateParameters( const std::vector<size_t>& shape ) {
            gate_weights_ = std::make_shared<Tensor<TOutput, MR>>( shape );
            gate_weights_->setName( this->getName() + ".gate_weights" );

            parameters_.clear();
            parameters_.push_back( gate_weights_ );

            // Create parameter gradients if in training mode
            if ( this->isTraining() ) {
                parameter_grads_.clear();
                auto gate_weights_grad = std::make_shared<Tensor<TOutput, MR>>( shape );
                gate_weights_grad->setName( this->getName() + ".gate_weights_grad" );
                parameter_grads_.push_back( gate_weights_grad );
            }

            // Initialize gate weights with a reasonable default
            // (e.g., 0.5 for equal contribution from input and inner module)
            fill<TOutput, MR>( *gate_weights_, static_cast<TOutput>(0.5) );
        }

        /**
         * @brief Adds two tensors element-wise.
         *
         * Helper method to add gradients from different paths during backpropagation.
         *
         * @param a First input tensor
         * @param b Second input tensor
         * @param result Output tensor for the sum
         */
        void addTensors(
            const Tensor<TInput, MR>& a,
            const Tensor<TInput, MR>& b,
            Tensor<TInput, MR>& result ) {

            auto add_op = OperationRegistry::instance().createBinaryOperation<TDeviceType, TInput, TInput, TInput>(
                "ElementwiseAdd",
                this->getDeviceContext(),
                config_ );

            add_op->forward( a, b, {}, result, {} );
        }
    };

    /**
     * @brief Type alias for CPU-based residual module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CpuResidual = Residual<DeviceType::Cpu, TInput, TOutput>;

    /**
     * @brief Type alias for CUDA-based residual module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CudaResidual = Residual<DeviceType::Cuda, TInput, TOutput>;
}