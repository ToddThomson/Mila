/**
 * @file Dropout.ixx
 * @brief Implementation of Dropout regularization module for neural networks.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <random>

export module Dnn.Modules.Dropout;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.Precision;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.ComputeDevice;
import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
	using namespace Mila::Dnn::Serialization;

    /**
     * @brief Dropout regularization module for neural networks.
     *
     * Dropout is a regularization technique that randomly zeroes elements of an input
     * tensor during training with probability p, and optionally scales the remaining
     * elements by 1/(1-p). During inference, dropout is typically disabled.
     *
     * Dropout helps prevent overfitting by preventing co-adaptation of feature detectors.
     * The technique effectively trains an ensemble of multiple networks sharing parameters,
     * which improves generalization.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which the module will operate.
     * @tparam TInput The data type of the input tensor elements.
     * @tparam TOutput The data type of the output tensor elements, defaults to TInput.
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidFloatTensorTypes<TInput, TOutput>
    class Dropout : public Module<TDeviceType, TInput, TOutput> {
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
         * @brief Constructs a new Dropout module with a device name.
         *
         * Creates a new DeviceContext internally using the provided device name.
         * This constructor is useful for creating standalone modules without
         * pre-existing device contexts.
         *
         * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
         * @param config Configuration parameters for the Dropout module.
         * @throws std::invalid_argument If the device name is invalid or the configuration is invalid
         * @throws std::runtime_error If device type doesn't match template parameter TDeviceType
         */
        explicit Dropout( const std::string& device_name, const DropoutConfig& config )
            : ModuleBase( std::make_shared<DeviceContext>( device_name ), config ), config_( config ) {

            config.validate();

            // Initialize random number generator with a random seed
            std::random_device rd;
            seed_ = rd();
            rng_.seed( seed_ );

            createOperation();
        }

        /**
         * @brief Constructs a new Dropout module with a provided device context.
         *
         * Uses a pre-existing DeviceContext instance. This constructor is useful when integrating
         * the module into a larger network that shares device contexts across modules.
         *
         * @param device_context The device context to use for this module.
         * @param config Configuration parameters for the Dropout module.
         * @throws std::invalid_argument If device_context is null or configuration is invalid
         * @throws std::runtime_error If device context type doesn't match template parameter TDeviceType
         */
        explicit Dropout( std::shared_ptr<DeviceContext> device_context, const DropoutConfig& config )
            : ModuleBase( device_context, config ), config_( config ) {

            config.validate();

            // Initialize random number generator with a random seed
            std::random_device rd;
            seed_ = rd();
            rng_.seed( seed_ );

            createOperation();
        }

        /**
         * @brief Gets the dropout probability used by this module.
         *
         * @return float The probability of zeroing elements
         */
        float getProbability() const {
            return config_.getProbability();
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * The Dropout module has no trainable parameters.
         *
         * @return size_t Always returns 0 for Dropout.
         */
        size_t parameterCount() const override {
            return 0;
        }

        /**
         * @brief Performs the forward pass of the Dropout operation.
         *
         * During training, randomly zeroes elements with probability p and scales
         * remaining elements by 1/(1-p). During inference, performs identity operation
         * or scaling based on configuration.
         *
         * @param input The input tensor.
         * @param output The output tensor where the results will be stored.
         */
        void forward( const Tensor<TInput, MR>& input, Tensor<TOutput, MR>& output ) {
            // Update mask if in training mode
            if ( this->isTraining() && config_.getProbability() > 0.0f ) {
                // Create or resize mask tensor if needed
                if ( !mask_ || mask_->getShape() != input.getShape() ) {
                    mask_ = std::make_shared<Tensor<TOutput, MR>>( input.getShape() );
                    mask_->setName( this->getName() + ".mask" );
                    // Add to state map for better debugging
                    this->state_map_[ "mask" ] = mask_;
                }

                // Generate new mask for this forward pass
                generateMask( *mask_, input.getShape() );

                // Update state tensors for operation
                if ( output_state_.empty() ) {
                    output_state_.push_back( mask_ );
                }
                else {
                    output_state_[ 0 ] = mask_;
                }
            }

            // Set operation properties
            properties_.set( "probability", config_.getProbability() );
            properties_.set( "training", this->isTraining() );
            properties_.set( "scale_during_inference", config_.scalesDuringInference() );

            // Perform the forward operation
            operation_->forward( input, parameters_, properties_, output, output_state_ );
        }

        /**
         * @brief Performs the backward pass of the Dropout operation.
         *
         * Computes the gradient of the Dropout function with respect to its input.
         * During training, gradients are only propagated for non-zeroed elements.
         * The gradient computation is straightforward: multiply the output gradient
         * by the same mask used in the forward pass.
         *
         * @param input The input tensor from the forward pass.
         * @param output_grad The gradient of loss with respect to the output.
         * @param input_grad The tensor to store gradients with respect to input.
         */
        void backward(
            const Tensor<TInput, MR>& input,
            const Tensor<TOutput, MR>& output_grad,
            Tensor<TInput, MR>& input_grad ) {

            operation_->backward(
                input,           // Input tensor
                output_grad,     // Gradient from next layer
                parameters_,     // Empty for Dropout
                {},              // No parameter gradients for Dropout
                input_grad,      // Gradient to propagate to previous layer
                properties_,     // Operation properties
                output_state_    // Cached mask tensor from forward pass
            );
        }

        /**
         * @brief Sets the random seed for dropout mask generation.
         *
         * This allows for reproducible dropout patterns when needed.
         *
         * @param seed The random seed value
         */
        void setSeed( unsigned int seed ) {
            seed_ = seed;
            rng_.seed( seed_ );
        }

        /**
         * @brief Gets the current random seed.
         *
         * @return unsigned int The current random seed
         */
        unsigned int getSeed() const {
            return seed_;
        }

        /**
         * @brief Serializes the module state to a ZIP archive.
         *
         * Implementation of the Module interface for serialization. Since Dropout has no
         * learnable parameters, this is a no-op implementation.
         *
         * @param zip ZIP archive for serialization
         */
        void save( ModelArchive& archive ) const override {
            // No-op: Dropout is a stateless module with no parameters to persist
        }

        /**
         * @brief Deserializes the module state from a ZIP archive.
         *
         * Implementation of the Module interface for deserialization. Since Dropout has no
         * learnable parameters, this is a no-op implementation.
         *
         * @param zip ZIP archive for deserialization
         */
        void load( ModelArchive& archive ) override {
            // No-op: Dropout is a stateless module with no parameters to load
        }

        /**
         * @brief Generates a string representation of this module's configuration.
         *
         * @return std::string A formatted string with module information
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "Dropout: " << this->getName() << std::endl;
            oss << "Probability: " << config_.getProbability() << std::endl;
            oss << "Mode: " << (this->isTraining() ? "Training" : "Inference") << std::endl;
            oss << "Scale during inference: " << (config_.scalesDuringInference() ? "Yes" : "No") << std::endl;
            oss << "Same mask per batch: " << (config_.usesSameMaskPerBatch() ? "Yes" : "No") << std::endl;
            oss << "Random seed: " << seed_ << std::endl;
            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;

            return oss.str();
        }

    private:
        /**
         * @brief Configuration for the Dropout module.
         */
        DropoutConfig config_;

        /**
         * @brief Random seed for reproducible dropout patterns.
         */
        unsigned int seed_{ 0 };

        /**
         * @brief Random number generator for mask generation.
         */
        std::mt19937 rng_;

        /**
         * @brief The binary mask tensor for element selection.
         *
         * Contains 0 for dropped elements and scale factor for kept elements.
         */
        std::shared_ptr<Tensor<TOutput, MR>> mask_{ nullptr };

        /**
         * @brief Collection of parameters for this module (empty for Dropout).
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> parameters_;

        /**
         * @brief Collection of output state tensors for caching.
         */
        std::vector<std::shared_ptr<Tensor<TOutput, MR>>> output_state_;

        /**
         * @brief Operation attributes and configuration.
         */
        OperationAttributes properties_;

        /**
         * @brief The operation that implements the dropout calculation.
         */
        std::shared_ptr<UnaryOperation<TDeviceType, TInput, TOutput>> operation_{ nullptr };

        /**
         * @brief Generates a new dropout mask for the given shape.
         *
         * Creates a binary mask with elements set to 0 (drop) with probability p,
         * and scale factor 1/(1-p) (keep) with probability (1-p).
         *
         * @param mask The tensor to populate with the mask values
         * @param shape The shape of the input/mask tensor
         */
        void generateMask( Tensor<TOutput, MR>& mask, const std::vector<size_t>& shape ) {
            // CPU-side implementations for mask generation
            // In practice, the actual implementation would offload this to GPU for CUDA

            // Calculate the scaling factor
            float scale = 1.0f / (1.0f - config_.getProbability());

            // Reset mask values
            auto* mask_data = mask.data();
            size_t total_elements = mask.size();

            if ( TDeviceType == DeviceType::Cpu ) {
                // CPU implementation
                std::uniform_real_distribution<float> dist( 0.0f, 1.0f );

                if ( config_.usesSameMaskPerBatch() ) {
                    // Same mask for all elements in batch
                    // Calculate elements per batch item
                    size_t batch_size = shape[ 0 ];
                    size_t elements_per_batch = total_elements / batch_size;

                    // Generate pattern for one batch item
                    std::vector<float> pattern( elements_per_batch );
                    for ( size_t i = 0; i < elements_per_batch; ++i ) {
                        pattern[ i ] = dist( rng_ ) >= config_.getProbability() ? scale : 0.0f;
                    }

                    // Replicate pattern across all batch items
                    for ( size_t b = 0; b < batch_size; ++b ) {
                        for ( size_t i = 0; i < elements_per_batch; ++i ) {
                            mask_data[ b * elements_per_batch + i ] = static_cast<TOutput>( pattern[ i ] );
                        }
                    }
                }
                else {
                    // Independent mask for each element
                    for ( size_t i = 0; i < total_elements; ++i ) {
                        mask_data[ i ] = static_cast<TOutput>( dist( rng_ ) >= config_.getProbability() ? scale : 0.0f );
                    }
                }
            }
            else {
                // For CUDA, we'd prepare the mask on CPU and transfer to device
                // or use a CUDA kernel to generate the mask directly on the device
                // This is just a placeholder - actual implementation would depend on CUDA kernel
                std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
                for ( size_t i = 0; i < total_elements; ++i ) {
                    mask_data[ i ] = static_cast<TOutput>( dist( rng_ ) >= config_.getProbability() ? scale : 0.0f );
                }

                // Transfer mask to device would happen here
            }
        }

        /**
         * @brief Creates the appropriate Dropout operation based on the current device context.
         *
         * This method initializes the operation_ member with the appropriate implementation
         * of the Dropout operation for either CPU or CUDA, based on the current device context.
         * It also passes the config object to the operation.
         */
        void createOperation() {
            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TInput, TOutput>(
                    "Cpu::DropoutOp",
                    this->getDeviceContext(),
                    config_ );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TInput, TOutput>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TInput, TOutput>(
                    "Cuda::DropoutOp",
                    this->getDeviceContext(),
                    config_ );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TInput, TOutput>>(base_op);
            }
        }
    };

    /**
     * @brief Type alias for CPU-based dropout module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CpuDropout = Dropout<DeviceType::Cpu, TInput, TOutput>;

    /**
     * @brief Type alias for CUDA-based dropout module with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CudaDropout = Dropout<DeviceType::Cuda, TInput, TOutput>;
}