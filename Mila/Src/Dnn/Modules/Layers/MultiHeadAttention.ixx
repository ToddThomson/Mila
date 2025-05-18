module;
#include <miniz.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>

export module Dnn.Modules.Attention;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;

import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.CpuDevice;
import Compute.CudaDevice;

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
     * @brief Multi-head attention module for transformer architectures.
     *
     * This module implements the multi-head attention mechanism, which allows different
     * parts of the input to attend to different parts of the sequence. This is a core
     * component of transformer architectures.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput.
     * @tparam TPrecision The data type used for internal calculations, defaults to TOutput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
        requires ValidTensorTypes<TInput, TOutput>
    class MultiHeadAttention : public Module<TDeviceType, TInput, TOutput> {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>; ///< Memory resource type based on device type
        using ModuleBase = Module<TDeviceType, TInput, TOutput>; ///< Base class type for the module

        /**
         * @brief Constructs a new MultiHeadAttention module with the default device context.
         *
         * @param name The name of the module for identification purposes.
         * @param device_name The name of the device to use for this module.
         * @param input_shape The shape of the input tensor.
         * @param num_heads The number of attention heads.
         * @param is_training Whether the module is initially in training mode. Default is false.
         */
        MultiHeadAttention( std::string name, std::string device_name,
            const std::vector<size_t>& input_shape, size_t num_heads, bool is_training = false )
            : ModuleBase( device_name ), input_shape_{ input_shape }, num_heads_{ num_heads } {
            this->setTraining( is_training );
            this->setName( name );
            initializeTensors();
            createOperation();
        }

        /**
         * @brief Constructs a new MultiHeadAttention module with a specific device context.
         *
         * @param name The name of the module for identification purposes.
         * @param context The device context to use for this module.
         * @param input_shape The shape of the input tensor.
         * @param num_heads The number of attention heads.
         * @param is_training Whether the module is initially in training mode. Default is false.
         */
        MultiHeadAttention( std::string name, std::shared_ptr<DeviceContext> context,
            const std::vector<size_t>& input_shape, size_t num_heads, bool is_training = false )
            : ModuleBase( context ), input_shape_{ input_shape }, num_heads_{ num_heads } {
            this->setTraining( is_training );
            this->setName( name );
            initializeTensors();
            createOperation();
        }

        /**
        * @brief Get the number of parameters.
        *
        * @return size_t The number of parameters.
        */
        size_t parameterCount() const override {
            return 0;
        }

        /**
         * @brief Performs the forward pass of the MultiHeadAttention operation.
         *
         * @param input The input tensor to be processed.
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TPrecision, MR>& input, Tensor<TPrecision, MR>& output ) {
            operation_->forward( input, parameters_, properties_, output, output_state_ );
        }

        /**
         * @brief Saves the module state to a ZIP archive.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( mz_zip_archive& zip ) const override {
            // MultiHeadAttention has no parameters to save
        }

        /**
         * @brief Loads the module state from a ZIP archive.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( mz_zip_archive& zip ) override {
            // MultiHeadAttention has no parameters to load
        }

        /**
        * @brief Convert the module information to string.
        *
        * @return std::string Module information as string.
        */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "MultiHeadAttention: " << this->getName();
            oss << ", Number of heads: " << num_heads_;
            oss << ", Input shape: (";
            for ( size_t i = 0; i < input_shape_.size(); ++i ) {
                oss << input_shape_[ i ];
                if ( i != input_shape_.size() - 1 ) {
                    oss << ",";
                }
            }
            oss << ")";
            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->stateToString();

            return oss.str();
        }

    private:
        /**
         * @brief The shape of the input tensor.
         *
         * Typically in the format [batch_size, sequence_length, embedding_dim].
         */
        std::vector<size_t> input_shape_;

        /**
         * @brief The number of attention heads.
         *
         * The embedding dimension is divided into this many heads, allowing the
         * model to attend to different parts of the sequence simultaneously.
         */
        size_t num_heads_{ 0 };

        /**
         * @brief Tensor storing the attention weights after softmax.
         *
         * Shape: [batch_size, num_heads_, sequence_length, sequence_length]
         */
        std::shared_ptr<Tensor<TPrecision, MR>> attn_{ nullptr };

        /**
         * @brief Tensor storing the pre-softmax attention values.
         *
         * Shape: [batch_size, num_heads_, sequence_length, sequence_length]
         */
        std::shared_ptr<Tensor<TPrecision, MR>> pre_attn_{ nullptr };

        /**
         * @brief The parameters for this module.
         *
         * MultiHeadAttention typically has no parameters at this level,
         * as the query, key, and value projections are handled by separate
         * linear layers in practice.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> parameters_;

        /**
         * @brief Cache for intermediate tensors needed for backward pass.
         */
        std::vector<std::shared_ptr<Tensor<TPrecision, MR>>> output_state_;

        /**
         * @brief Properties that configure the attention operation.
         */
        OperationAttributes properties_;

        /**
         * @brief The underlying operation that implements the multi-head attention.
         */
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>> operation_{ nullptr };

        /**
         * @brief Initializes the tensors needed for the MultiHeadAttention operation.
         *
         * Creates intermediate tensors for storing attention matrices and sets up
         * operation properties.
         */
        void initializeTensors() {
            // Clear existing state
            output_state_.clear();

            auto batch_size = input_shape_[ 0 ];
            auto sequence_length = input_shape_[ 1 ];

            pre_attn_ = std::make_shared<Tensor<TPrecision, MR>>(
                std::vector<size_t>{batch_size, num_heads_, sequence_length, sequence_length} );
            pre_attn_->setName( this->getName() + ".pre_attn" );

            attn_ = std::make_shared<Tensor<TPrecision, MR>>(
                std::vector<size_t>{batch_size, num_heads_, sequence_length, sequence_length} );
            attn_->setName( this->getName() + ".attn" );

            // Add state tensors
            output_state_.emplace_back( pre_attn_ );
            output_state_.emplace_back( attn_ );

            // Set number of heads in properties
            properties_.set( "num_heads", num_heads_ );
        }

        /**
         * @brief Creates the appropriate MultiHeadAttention operation based on the current device context.
         *
         * This method selects and initializes the correct implementation (CPU or CUDA)
         * of the attention operation based on the device type.
         */
        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TInput, TOutput, TPrecision>(
                    "Cpu::MultiHeadAttentionOp",
                    this->getDeviceContext() );
                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TInput, TOutput, TPrecision>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TInput, TOutput, TPrecision>(
                    "Cuda::MultiHeadAttentionOp",
                    this->getDeviceContext() );
                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TInput, TOutput, TPrecision>>(base_op);
            }
        }
    };

    /**
     * @brief Type alias for CPU-based multi-head attention module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     * @tparam TPrecision Data type used for internal calculations, defaults to TOutput.
     */
    export template<typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
        using CpuMultiHeadAttention = MultiHeadAttention<DeviceType::Cpu, TInput, TOutput, TPrecision>;

    /**
     * @brief Type alias for CUDA-based multi-head attention module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     * @tparam TPrecision Data type used for internal calculations, defaults to TOutput.
     */
    export template<typename TInput = float, typename TOutput = TInput, typename TPrecision = TOutput>
        using CudaMultiHeadAttention = MultiHeadAttention<DeviceType::Cuda, TInput, TOutput, TPrecision>;
}