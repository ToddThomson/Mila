/**
 * @file LayerNorm.ixx
 * @brief Implementation of Layer Normalization module for neural networks.
 */

module;
#include <miniz.h>
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <type_traits>
#include <cstdint>
#include <stdexcept>

export module Dnn.Modules.LayerNorm;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;

import Compute.ComputeDevice;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Layer Normalization module.
     *
     * Layer Normalization is a technique used to normalize the inputs across features
     * for each data sample in a batch. It helps stabilize and accelerate deep neural
     * network training by reducing internal covariate shift.
     *
     * The operation can be expressed as:
     * y = ((x - mean) / sqrt(variance + epsilon)) * weight + bias
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     * @tparam TPrecision Data type used for internal calculations, defaults to TOutput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda,
        typename TInput = float,
        typename TOutput = TInput,
        typename TPrecision = TOutput>
        requires ValidTensorTypes<TInput, TOutput>&& ValidPrecisionType<TPrecision>
    class LayerNorm : public Module<TDeviceType, TInput, TOutput, TPrecision> {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>; ///< Memory resource type based on device type
        using ModuleBase = Module<TDeviceType, TInput, TOutput, TPrecision>; ///< Base class type for the module

        /**
         * @brief Constructs a new LayerNorm object with the default device context.
         *
         * @param name Name of the module for identification purposes.
         * @param device_name The name of the device to use for this module.
         * @param input_shape Shape of the input tensor, typically [batch_size, sequence_length, channels].
         * @param axis Axis for normalization. Default is -1 (last dimension).
         * @param has_bias Whether the module should use a bias tensor. Default is true.
         * @param is_training Whether the module is initially in training mode. Default is false.
         */
        LayerNorm(
            std::string name, std::string device_name, const std::vector<size_t>& input_shape,
            int64_t axis = -1, bool has_bias = true, bool is_training = false )
            : ModuleBase( device_name ), input_shape_{ input_shape }, axis_{ axis }, has_bias_{ has_bias } {
            this->setTraining( is_training );
            this->setName( name );
            initializeTensors();
            createOperation();
        }

        /**
         * @brief Constructs a new LayerNorm object with a specific device context.
         *
         * @param name Name of the module for identification purposes.
         * @param context The device context to use for this module.
         * @param input_shape Shape of the input tensor, typically [batch_size, sequence_length, channels].
         * @param axis Axis for normalization. Default is -1 (last dimension).
         * @param has_bias Whether the module should use a bias tensor. Default is true.
         * @param is_training Whether the module is initially in training mode. Default is false.
         */
        LayerNorm(
            std::string name, std::shared_ptr<DeviceContext> context,
            const std::vector<size_t>& input_shape, int64_t axis = -1, bool has_bias = true, bool is_training = false )
            : ModuleBase( context ), input_shape_{ input_shape }, axis_{ axis }, has_bias_{ has_bias } {
            this->setTraining( is_training );
            this->setName( name );
            initializeTensors();
            createOperation();
        }

        /**
         * @brief Gets the weight tensor used for scaling after normalization.
         *
         * The weight tensor is applied as a scale factor to the normalized values.
         *
         * @return std::shared_ptr<Tensor<TPrecision, MR>> Shared pointer to the weight tensor.
         */
        std::shared_ptr<Tensor<TPrecision, MR>> getWeight() {
            return std::static_pointer_cast<Tensor<TPrecision, MR>>(weight_);
        }

        /**
         * @brief Gets the bias tensor used after normalization and scaling.
         *
         * The bias tensor is added after normalization and scaling.
         *
         * @return std::shared_ptr<Tensor<TPrecision, MR>> Shared pointer to the bias tensor.
         */
        std::shared_ptr<Tensor<TPrecision, MR>> getBias() {
            return std::static_pointer_cast<Tensor<TPrecision, MR>>(bias_);
        }

        /**
        * @brief Gets whether the module has a bias tensor.
        *
        * @return bool True if the module has a bias tensor, false otherwise.
        */
        bool hasBias() const {
            return has_bias_;
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * Counts the total number of trainable parameters, which includes
         * the weight tensor and, if present, the bias tensor.
         *
         * @return size_t The total number of parameters.
         */
        size_t parameterCount() const override {
            size_t count = weight_->size();
            if ( has_bias_ ) {
                count += bias_->size();
            }
            return count;
        }

        /**
         * @brief Performs the forward pass of the Layer Normalization operation.
         *
         * Normalizes the input tensor across the specified axis, then scales and shifts
         * the result using the weight and bias tensors.
         *
         * @param input The input tensor to be normalized.
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TPrecision, MR>& input, Tensor<TPrecision, MR>& output ) {
            operation_->forward( input, parameters_, properties_, output, output_state_ );
        }

        /**
         * @brief Saves the module state to a ZIP archive.
         *
         * Serializes the trainable parameters (weight, bias) to the provided archive.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( mz_zip_archive& zip ) const override {
            // Save the state of the parameters
            for ( const auto& [name, tensor] : this->getParameterTensors() ) {
                // Save tensor data to zip archive
            }
        }

        /**
         * @brief Loads the module state from a ZIP archive.
         *
         * Deserializes the trainable parameters (weight, bias) from the provided archive.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( mz_zip_archive& zip ) override {
            for ( const auto& [name, tensor] : this->getParameterTensors() ) {
                // Load tensor data from zip archive
            }
        }

        /**
         * @brief Converts the module information to a human-readable string.
         *
         * Includes detailed information about the module configuration,
         * parameters, and state tensors.
         *
         * @return std::string A string representation of the module information.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "LayerNorm: " << this->getName();
            oss << ", Normalization Axis: " << axis_;
            oss << ", Input shape: (";
            for ( size_t i = 0; i < input_shape_.size(); ++i ) {
                oss << input_shape_[ i ];
                if ( i != input_shape_.size() - 1 ) {
                    oss << ",";
                }
            }
            oss << ")";
            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;

            oss << "Parameter Tensors..." << std::endl;
            for ( const auto& [name, tensor] : this->getParameterTensors() ) {
                oss << tensor->toString();
            }
            oss << "Parameter count: " << parameterCount() << std::endl;

            oss << "State Tensors..." << std::endl;
            for ( const auto& [name, tensor] : this->getStateTensors() ) {
                oss << tensor->toString();
            }

            return oss.str();
        }

    private:
        /**
         * @brief The shape of the input tensor to be normalized.
         */
        std::vector<size_t> input_shape_;

        /**
         * @brief Small constant added to variance for numerical stability.
         */
        float epsilon_{ 1e-05f };

        /**
         * @brief The axis along which to normalize. Default is -1 for last dimension.
         */
        int64_t axis_{ -1 };

        /**
         * @brief Whether the module has a bias tensor. Default is true.
         */
        bool has_bias_{ true };

        /**
         * @brief The weight tensor for scaling after normalization.
         */
        std::shared_ptr<Tensor<TPrecision, MR>> weight_{ nullptr };

        /**
         * @brief The bias tensor added after normalization and scaling.
         */
        std::shared_ptr<Tensor<TPrecision, MR>> bias_{ nullptr };

        /**
         * @brief The mean tensor used for normalization.
         *
         * Stores the mean values computed during the forward pass.
         */
        std::shared_ptr<Tensor<TPrecision, MR>> mean_{ nullptr };

        /**
         * @brief The reciprocal standard deviation tensor.
         *
         * Stores the reciprocal of the standard deviation values (1/sqrt(variance + epsilon))
         * computed during the forward pass.
         */
        std::shared_ptr<Tensor<TPrecision, MR>> rstd_{ nullptr };

        /**
         * @brief The trainable parameters for this module (weight and bias).
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> parameters_;

        /**
         * @brief Cache of intermediate tensors needed for backward pass.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> output_state_;

        /**
         * @brief The operation attributes, including epsilon and axis information.
         */
        OperationAttributes properties_;

        /**
         * @brief The underlying operation that implements Layer Normalization.
         */
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput, TPrecision>> operation_{ nullptr };

        /**
         * @brief Initializes the tensors needed for the Layer Normalization operation.
         *
         * Creates and initializes:
         * - weight tensor (initialized to ones)
         * - bias tensor (initialized to zeros)
         * - mean tensor (for storing means during forward pass)
         * - reciprocal standard deviation tensor (for storing 1/std during forward pass)
         */
        void initializeTensors() {
            // Clear existing parameters and state
            parameters_.clear();
            output_state_.clear();
            this->parameter_map_.clear();
            this->state_map_.clear();

            // Get device type for proper tensor creation
            auto device_type = this->getDeviceContext()->getDevice()->getDeviceType();

            auto batch_size = input_shape_[ 0 ];
            auto sequence_length = input_shape_[ 1 ];
            auto channels = input_shape_[ 2 ];

            weight_ = std::make_shared<Tensor<TPrecision, MR>>(
                std::vector<size_t>{channels}, static_cast<TPrecision>(1.0f) );
            weight_->setName( this->getName() + ".weight" );

            if ( has_bias_ ) {
                bias_ = std::make_shared<Tensor<TPrecision, MR>>(
                    std::vector<size_t>{channels} );
                bias_->setName( this->getName() + ".bias" );
            }

            mean_ = std::make_shared<Tensor<TPrecision, MR>>(
                std::vector<size_t>{batch_size, sequence_length} );
            mean_->setName( this->getName() + ".mean" );

            rstd_ = std::make_shared<Tensor<TPrecision, MR>>(
                std::vector<size_t>{batch_size, sequence_length} );
            rstd_->setName( this->getName() + ".rstd" );

            parameters_.emplace_back( weight_ );
            this->parameter_map_[ "weight" ] = weight_;

            if ( has_bias_ ) {
                parameters_.emplace_back( bias_ );
                this->parameter_map_[ "bias" ] = bias_;
            }

            // Add state tensors
            output_state_.emplace_back( mean_ );
            output_state_.emplace_back( rstd_ );

            this->state_map_[ "mean" ] = mean_;
            this->state_map_[ "rstd" ] = rstd_;

            // Set epsilon and axis in the properties
            properties_.set( "epsilon", epsilon_ );
            properties_.set( "axis", axis_ );
        }

        /**
         * @brief Creates the appropriate Layer Normalization operation based on the current device context.
         *
         * This method initializes the operation_ member with the appropriate implementation
         * of Layer Normalization for either CPU or CUDA, based on the current device context.
         */
        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<TInput, TOutput, TPrecision, DeviceType::Cpu>(
                    "Cpu::LayerNormOp",
                    this->getDeviceContext() );

                operation_ = std::static_pointer_cast<UnaryOperation<TInput, TOutput, TPrecision, TDeviceType>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<TInput, TOutput, TPrecision, DeviceType::Cuda>(
                    "Cuda::LayerNormOp",
                    this->getDeviceContext() );

                operation_ = std::static_pointer_cast<UnaryOperation<TInput, TOutput, TPrecision, TDeviceType>>(base_op);
            }
        }
    };

    /**
     * @brief Type alias for CPU-based layer normalization module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     * @tparam TPrecision Data type used for internal calculations, defaults to TOutput.
     */
    export template<typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
        using CpuLayerNorm = LayerNorm<DeviceType::Cpu, TInput, TOutput, TPrecision>;

    /**
     * @brief Type alias for CUDA-based layer normalization module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     * @tparam TPrecision Data type used for internal calculations, defaults to TOutput.
     */
    export template<typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
        using CudaLayerNorm = LayerNorm<DeviceType::Cuda, TInput, TOutput, TPrecision>;
}