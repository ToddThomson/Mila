/**
 * @file Residual.ixx
 * @brief Implementation of the Residual connection module for neural networks.
 *
 * Provides a flexible implementation of residual connections which can be configured
 * with different connection types and scaling factors. Supports automatic dimension
 * matching via projection layers.
 */

module;
#include <miniz.h>
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
import Compute.CpuDevice;
import Compute.CudaMemoryResource;
import Compute.CpuMemoryResource;

import Dnn.Modules.Linear;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief A class implementing a residual connection module.
     *
     * Residual connections help deep neural networks avoid vanishing gradients by
     * providing shortcut connections. The basic formula is y = x + F(x), where F
     * is a differentiable function (usually a sequence of neural network layers).
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidFloatTensorTypes<TInput, TOutput>
    class Residual : public Module<TDeviceType, TInput, TOutput> {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;
        using ModuleBase = Module<TDeviceType, TInput, TOutput>;

        /**
         * @brief Construct a new Residual module from configuration.
         *
         * @param config The configuration for this residual connection
         */
        explicit Residual( const ResidualConfig& config )
            : ModuleBase(
                config.getContext() ? config.getContext() : std::make_shared<DeviceContext>( config.getDeviceName() ),
                TDeviceType == DeviceType::Cpu ? ComputePrecision::Policy::Disabled : config.getPrecision() ),
            scaling_factor_( config.getScalingFactor() ),
            connection_type_( config.getConnectionType() ),
            use_projection_( config.useProjection() ) {

            config.validate();

            this->setName( config.getName() );
            this->setTraining( config.isTraining() );

            inner_module_ = config.getInnerModule<TDeviceType, TInput, TOutput>();
            createOperation();
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * @return size_t The total number of parameters (including inner module)
         */
        size_t parameterCount() const override {
            size_t count = inner_module_->parameterCount();

            if ( projection_ ) {
                count += projection_->parameterCount();
            }

            if ( connection_type_ == ResidualConfig::ConnectionType::Gated ) {
                count += gate_weights_->size();
            }

            return count;
        }

        /**
         * @brief Performs the forward pass of the Residual connection.
         *
         * @param input The input tensor
         * @param output The output tensor
         */
        void forward( const Tensor<TInput, MR>& input, Tensor<TOutput, MR>& output ) {
            inner_module_->forward( input, inner_output_ );

            bool input_dims_match = tensorShapesMatch( input, inner_output_ );

            if ( !input_dims_match && !use_projection_ ) {
                throw std::runtime_error( "Input and inner module output dimensions don't match, and projection is disabled" );
            }

            if ( !input_dims_match && !projection_ ) {
                createProjection( input.getShape(), inner_output_.getShape() );
            }

            switch ( connection_type_ ) {
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
                    properties_[ "scaling_factor" ] = scaling_factor_;

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
                    // This would use a custom operation for gated residual connections
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
         * @param input The input tensor from the forward pass
         * @param output_grad The gradient of loss with respect to the output
         * @param input_grad The tensor to store gradients with respect to input
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TOutput, MR>& output_grad,
            Tensor<TInput, MR>& input_grad ) {

            bool input_dims_match = tensorShapesMatch( input, inner_output_ );

            if ( connection_type_ != ResidualConfig::ConnectionType::Gated ) {
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
         * @return std::shared_ptr<Module<TDeviceType, TInput, TOutput>> The inner module
         */
        std::shared_ptr<Module<TDeviceType, TInput, TOutput>> getInnerModule() {
            return inner_module_;
        }

        /**
         * @brief Serializes the module state to a ZIP archive.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( mz_zip_archive& zip ) const override {
            inner_module_->save( zip );

            if ( projection_ ) {
                projection_->save( zip );
            }

            if ( connection_type_ == ResidualConfig::ConnectionType::Gated && gate_weights_ ) {
                // Save gate weights
                // Implementation depends on tensor serialization
            }
        }

        /**
         * @brief Deserializes the module state from a ZIP archive.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( mz_zip_archive& zip ) override {
            inner_module_->load( zip );

            if ( projection_ ) {
                projection_->load( zip );
            }

            if ( connection_type_ == ResidualConfig::ConnectionType::Gated && gate_weights_ ) {
                // Load gate weights
                // Implementation depends on tensor deserialization
            }
        }

        /**
         * @brief Converts the module information to a human-readable string.
         *
         * @return std::string A string representation of the module information.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Residual: " << this->getName() << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << "Connection Type: " << connectionTypeToString( connection_type_ ) << std::endl;

            if ( connection_type_ == ResidualConfig::ConnectionType::ScaledAddition ) {
                oss << "Scaling Factor: " << scaling_factor_ << std::endl;
            }

            oss << "Has Projection: " << (projection_ ? "Yes" : "No") << std::endl;
            oss << "Inner Module: " << std::endl;
            oss << inner_module_->toString() << std::endl;

            oss << "Parameter count: " << this->parameterCount() << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;

            return oss.str();
        }

    private:
        float scaling_factor_;
        ResidualConfig::ConnectionType connection_type_;
        bool use_projection_;

        std::shared_ptr<Module<TDeviceType, TInput, TOutput>> inner_module_;
        std::shared_ptr<Linear<TDeviceType, TInput, TOutput>> projection_;

        std::shared_ptr<Tensor<TOutput, MR>> gate_weights_;
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameters_;
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameter_grads_;
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> inner_parameter_grads_;

        OperationAttributes properties_;
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> output_state_;

        Tensor<TOutput, MR> inner_output_{};
        Tensor<TOutput, MR> projection_output_{};
        Tensor<TInput, MR> inner_input_grad_{};
        Tensor<TInput, MR> temp_grad_{};

        std::shared_ptr<BinaryOperation<TDeviceType, TInput, TOutput, TOutput>> operation_;
        std::shared_ptr<BinaryOperation<TDeviceType, TInput, TOutput, TOutput>> gated_operation_;

        /**
         * @brief Converts connection type enum to string.
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
         */
        void createOperation() {
            if ( connection_type_ != ResidualConfig::ConnectionType::Gated ) {
                std::string op_name;

                if ( connection_type_ == ResidualConfig::ConnectionType::Addition ) {
                    op_name = "ResidualAdd";
                }
                else {
                    op_name = "ScaledResidualAdd";
                }

                if constexpr ( TDeviceType == DeviceType::Cpu ) {
                    auto base_op = OperationRegistry::instance().createBinaryOperation<DeviceType::Cpu, TInput, TOutput, TOutput>(
                        "Cpu::" + op_name,
                        this->getDeviceContext(),
                        ComputePrecision::Policy::Disabled );

                    operation_ = std::static_pointer_cast<BinaryOperation<DeviceType::Cpu, TInput, TOutput, TOutput>>(base_op);
                }
                else {
                    auto base_op = OperationRegistry::instance().createBinaryOperation<DeviceType::Cuda, TInput, TOutput, TOutput>(
                        "Cuda::" + op_name,
                        this->getDeviceContext(),
                        this->getComputePrecision().getPolicy() );

                    operation_ = std::static_pointer_cast<BinaryOperation<DeviceType::Cuda, TInput, TOutput, TOutput>>(base_op);
                }
            }
            else {
                if constexpr ( TDeviceType == DeviceType::Cpu ) {
                    auto base_op = OperationRegistry::instance().createBinaryOperation<DeviceType::Cpu, TInput, TOutput, TOutput>(
                        "Cpu::GatedResidual",
                        this->getDeviceContext(),
                        ComputePrecision::Policy::Disabled );

                    gated_operation_ = std::static_pointer_cast<BinaryOperation<DeviceType::Cpu, TInput, TOutput, TOutput>>(base_op);
                }
                else {
                    auto base_op = OperationRegistry::instance().createBinaryOperation<DeviceType::Cuda, TInput, TOutput, TOutput>(
                        "Cuda::GatedResidual",
                        this->getDeviceContext(),
                        this->getComputePrecision().getPolicy() );

                    gated_operation_ = std::static_pointer_cast<BinaryOperation<DeviceType::Cuda, TInput, TOutput, TOutput>>(base_op);
                }
            }
        }

        /**
         * @brief Creates a projection layer when input and output dimensions don't match.
         */
        void createProjection( const std::vector<size_t>& input_shape, const std::vector<size_t>& output_shape ) {
            size_t input_features = input_shape.back();
            size_t output_features = output_shape.back();

            auto projection_config = LinearConfig( input_features, output_features )
                .withName( this->getName() + ".projection" )
                .withDeviceContext( this->getDeviceContext() )
                .withPrecision( this->getComputePrecision().getPolicy() )
                .training( this->isTraining() );

            projection_ = std::make_shared<Linear<TDeviceType, TInput, TOutput>>( projection_config );

            projection_output_ = Tensor<TOutput, MR>( output_shape );
        }

        /**
         * @brief Initializes parameters for gated connections.
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
         */
        void addTensors(
            const Tensor<TInput, MR>& a,
            const Tensor<TInput, MR>& b,
            Tensor<TInput, MR>& result ) {

            auto add_op = OperationRegistry::instance().createBinaryOperation<TDeviceType, TInput, TInput, TInput>(
                "ElementwiseAdd",
                this->getDeviceContext(),
                this->getComputePrecision().getPolicy() );

            add_op->forward( a, b, {}, result, {} );
        }
    };

    /**
     * @brief Type alias for CPU-based residual module with customizable tensor types.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CpuResidual = Residual<DeviceType::Cpu, TInput, TOutput>;

    /**
     * @brief Type alias for CUDA-based residual module with customizable tensor types.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CudaResidual = Residual<DeviceType::Cuda, TInput, TOutput>;
}