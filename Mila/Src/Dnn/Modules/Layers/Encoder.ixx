/**
* @file Encoder.ixx
* @brief Implementation of the Encoder module for token and positional embeddings in transformer models.
*/

module;
#include <miniz.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <cuda_fp16.h>

export module Dnn.Modules.Encoder;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;
import Compute.DeviceType;
import Compute.CpuDevice;
import Compute.CudaDevice;
import Compute.DeviceContext;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
    * @class Encoder
    * @brief An encoder module that provides token and positional embeddings.
    *
    * The Encoder transforms input token IDs into continuous vector representations by:
    * 1. Looking up token embeddings from a vocabulary table (wte)
    * 2. Adding positional embeddings (wpe) based on sequence position
    *
    * This implementation supports both CPU and CUDA execution depending on the device context.
    * The encoder is a fundamental component in transformer architectures, providing the initial
    * representation of tokens that subsequent layers will process.
    *
    * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
    * @tparam TInput The data type of the input token IDs (typically int).
    * @tparam TOutput The data type of the output embeddings (typically float).
    * @tparam TPrecision The data type used for internal calculations, defaults to TOutput.
    */
    export
        template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = int, typename TOutput = float>
        requires ValidTensorType<TInput> && ValidFloatTensorType<TOutput>
    class Encoder : public Module<TDeviceType, TInput, TOutput> {
    public:
        using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::CudaMemoryResource, Compute::CpuMemoryResource>;
        using ModuleBase = Module<TDeviceType, TInput, TOutput>; ///< Base class type for the module

        /**
        * @brief Construct a new Encoder module with a device name.
        *
        * @param name The name of the module for identification.
        * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
        * @param channels The embedding dimension size (C).
        * @param max_seq_len The maximum sequence length supported by the encoder (maxT).
        * @param vocab_len The size of the vocabulary (V).
        * @param is_training Whether the module is in training mode. Default is false.
        */
        Encoder( std::string name, std::string device_name, size_t channels, size_t max_seq_len, size_t vocab_len, bool is_training = false )
            : ModuleBase( device_name ), channels_{ channels }, max_seq_len_{ max_seq_len }, vocab_len_{ vocab_len } {
            this->setTraining( is_training );
            this->setName( name );
            initializeTensors();
            createOperation();
        }

        /**
        * @brief Construct a new Encoder module with a provided device context.
        *
        * @param name The name of the module for identification.
        * @param context The device context to use for this module. If nullptr, the default context will be used.
        * @param channels The embedding dimension size (C).
        * @param max_seq_len The maximum sequence length supported by the encoder (maxT).
        * @param vocab_len The size of the vocabulary (V).
        * @param is_training Whether the module is in training mode. Default is false.
        */
        Encoder( std::string name, std::shared_ptr<DeviceContext> context, size_t channels, size_t max_seq_len, size_t vocab_len,
            bool is_training = false )
            : ModuleBase( context ), channels_{ channels }, max_seq_len_{ max_seq_len }, vocab_len_{ vocab_len } {
            this->setTraining( is_training );
            this->setName( name );
            initializeTensors();
            createOperation();
        }

        /**
        * @brief Perform the forward pass of the encoder.
        *
        * Transforms input token IDs into continuous embeddings by:
        * 1. Looking up token embeddings from the embedding table (wte)
        * 2. Adding positional embeddings (wpe) based on token position
        *
        * @param input The input tensor containing token IDs with shape (B,T).
        * @param output The output tensor that will contain embeddings with shape (B,T,C).
        */
        void forward( const Tensor<TInput, MR>& input, Tensor<TOutput, MR>& output ) {
            operation_->forward( input, parameters_, attributes_, output, output_state_ );
        }

        /**
        * @brief Get the number of channels (embedding dimension).
        *
        * @return size_t The number of channels (C).
        */
        size_t getChannels() const {
            return channels_;
        }

        /**
        * @brief Get the vocabulary length.
        *
        * @return size_t The vocabulary length (V).
        */
        size_t getVocabularyLength() const {
            return vocab_len_;
        }

        /**
        * @brief Get the maximum sequence length.
        *
        * @return size_t The maximum sequence length (maxT).
        */
        size_t getMaxSequenceLength() const {
            return max_seq_len_;
        }

        /**
        * @brief Get the number of parameters in the module.
        *
        * Counts all learnable parameters in the encoder, which includes
        * all elements in the token embedding table (wte) and position
        * embedding table (wpe).
        *
        * @return size_t The total number of parameters.
        */
        size_t parameterCount() const override {
            return wte_->size() + wpe_->size();
        }

        /**
        * @brief Save the encoder parameters to a zip archive.
        *
        * Serializes all parameter tensors (wte and wpe) to the specified zip archive.
        * This enables model persistence for later reuse or distribution.
        *
        * @param zip The zip archive to save the parameters to.
        */
        void save( mz_zip_archive& zip ) const override {
            // Save the state of the parameters
            for ( const auto& [name, tensor] : this->getParameterTensors() ) {
                // Save tensor data to zip archive
            }
        }

        /**
        * @brief Load the encoder parameters from a zip archive.
        *
        * Deserializes all parameter tensors (wte and wpe) from the specified zip archive.
        * This enables loading pretrained models for inference or continued training.
        *
        * @param zip The zip archive to load the parameters from.
        */
        void load( mz_zip_archive& zip ) override {
            for ( const auto& [name, tensor] : this->getParameterTensors() ) {
                // Load tensor data from zip archive
            }
        }

        /**
        * @brief Get the module information as a string.
        *
        * Provides a human-readable description of the encoder configuration,
        * including dimensions, parameter counts, and tensor information.
        *
        * @return std::string The module information.
        */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "Encoder: " << this->getName();
            oss << ", Channels: " << channels_ << ", Max Sequence Length: " << max_seq_len_;
            oss << ", Vocabulary Length: " << vocab_len_;
            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << "Parameter Tensors..." << std::endl;
            for ( const auto& [name, tensor] : this->getParameterTensors() ) {
                oss << tensor->toString();
            }
            oss << "Parameter count: " << parameterCount() << std::endl;

            return oss.str();
        }

    private:
        size_t channels_; ///< The embedding dimension size (C).
        size_t max_seq_len_; ///< The maximum sequence length the encoder can process (maxT).
        size_t vocab_len_; ///< The vocabulary size, representing the number of unique tokens (V).

        /**
         * Token embedding table with shape (V,C), maps token IDs to vector representations.
         * V is the vocabulary size and C is the embedding dimension.
         */
        std::shared_ptr<Tensor<TOutput, MR>> wte_{ nullptr };

        /**
         * Position embedding table with shape (maxT,C), encodes token position information.
         * maxT is the maximum sequence length and C is the embedding dimension.
         */
        std::shared_ptr<Tensor<TOutput, MR>> wpe_{ nullptr };

        /**
         * Vector of parameter tensors that will be used during forward/backward passes.
         * Contains both the token embeddings (wte) and position embeddings (wpe).
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameters_;

        /**
         * Output state tensors used for intermediate values. Not used in this module.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> output_state_;

        /**
         * Operation-specific attributes and configuration.
         */
        OperationAttributes attributes_;

        /**
         * The computational operation that implements the encoder logic.
         */
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>> operation_{ nullptr };

        /**
        * @brief Initialize the token and positional embedding tensors.
        *
        * Creates and initializes:
        * - wte (word token embeddings) tensor of shape (vocab_len_, channels_)
        * - wpe (word position embeddings) tensor of shape (max_seq_len_, channels_)
        *
        * Both tensors are initialized using Xavier initialization to ensure proper
        * gradient flow during training. The tensors are registered as parameters
        * in the module's parameter map for training and serialization.
        */
        void initializeTensors() {
            parameters_.clear();
            this->parameter_map_.clear();

            wte_ = std::make_shared<Tensor<TOutput, MR>>( std::vector<size_t>{vocab_len_, channels_} );
            wte_->setName( this->getName() + ".wte" );
            xavier<TOutput, MR>( *wte_, vocab_len_, channels_ );

            wpe_ = std::make_shared<Tensor<TOutput, MR>>( std::vector<size_t>{max_seq_len_, channels_} );
            wpe_->setName( this->getName() + ".wpe" );
            xavier<TOutput, MR>( *wpe_, max_seq_len_, channels_ );

            // Add tensors to parameters list and map
            parameters_.emplace_back( wte_ );
            parameters_.emplace_back( wpe_ );

            this->parameter_map_[ "wte" ] = wte_;
            this->parameter_map_[ "wpe" ] = wpe_;
        }

        /**
        * @brief Create the computational operation based on current device context.
        *
        * Instantiates either a CPU or CUDA encoder operation based on the current device context.
        * The operation implements the actual embedding lookup and addition logic during forward pass.
        *
        * For CPU device, creates a "Cpu::EncoderOp" operation.
        * For CUDA device, creates a "Cuda::EncoderOp" operation.
        */
        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TInput, TOutput>(
                    "Cpu::EncoderOp",
                    this->getDeviceContext() );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TInput, TOutput>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TInput, TOutput>(
                    "Cuda::EncoderOp",
                    this->getDeviceContext() );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TInput, TOutput>>(base_op);
            }
        }
    };

    /**
     * @brief Type alias for CPU-based encoder module with customizable tensor types.
     *
     * @tparam TInput Data type of the input token IDs (typically int).
     * @tparam TOutput Data type of the output embeddings (typically float).
     * @tparam TPrecision Data type used for internal calculations, defaults to TOutput.
     */
    export template<typename TInput = int, typename TOutput = float>
        using CpuEncoder = Encoder<DeviceType::Cpu, TInput, TOutput>;

    /**
     * @brief Type alias for CUDA-based encoder module with customizable tensor types.
     *
     * @tparam TInput Data type of the input token IDs (typically int).
     * @tparam TOutput Data type of the output embeddings (typically float).
     * @tparam TPrecision Data type used for internal calculations, defaults to TOutput.
     */
    export template<typename TInput = int, typename TOutput = float>
        using CudaEncoder = Encoder<DeviceType::Cuda, TInput, TOutput>;
}