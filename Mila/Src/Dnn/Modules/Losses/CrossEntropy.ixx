/**
 * @file CrossEntropy.ixx
 * @brief Implementation of the CrossEntropy loss function module.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <cstdint>

export module Dnn.Modules.CrossEntropy;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;
import Compute.Precision;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.ComputeDevice;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationAttributes;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
	using namespace Mila::Dnn::Serialization;

    /**
     * @brief CrossEntropy loss module for neural networks.
     *
     * This class implements the cross entropy loss function, which is commonly used in classification tasks.
     * It computes the negative log likelihood of the correct class given the predicted probabilities.
     *
     * The cross entropy loss for a single example is calculated as:
     * -log(p_i) where p_i is the predicted probability for the correct class i.
     *
     * For multi-class problems with K classes, the formula is:
     * L(y, ?) = -?(y_k * log(?_k)) for k=1 to K
     * where y is the ground truth (one-hot) and ? is the predicted probability distribution.
     *
     * Features supported by this implementation:
     * - Class weighting for imbalanced datasets
     * - Padding index ignoring for variable-length sequences
     * - Label smoothing for regularization
     * - Optional reduction (mean or none)
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TLogits The data type of the predicted probabilities (typically float).
     * @tparam TTargets The data type of the target class indices (typically int).
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TLogits = float, typename TTargets = int>
        requires ValidFloatTensorType<TLogits>
    class CrossEntropy : public Module<TDeviceType, TLogits, TTargets> {
    public:
        /**
         * @brief Memory resource type used for tensors, selected based on device type.
         */
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;

        /**
         * @brief Alias for base module type.
         */
        using ModuleBase = Module<TDeviceType, TLogits, TTargets>;

        /**
         * @brief Constructs a new CrossEntropy module with a device name.
         *
         * Creates a new DeviceContext internally using the provided device name.
         * This constructor is useful for creating standalone modules without
         * pre-existing device contexts.
         *
         * @param device_name The name of the device to use (e.g., "CPU", "CUDA:0").
         * @param config Configuration parameters for the CrossEntropy module.
         * @throws std::invalid_argument If the device name is invalid or the configuration is invalid
         * @throws std::runtime_error If device type doesn't match template parameter TDeviceType
         */
        explicit CrossEntropy( const std::string& device_name, const CrossEntropyConfig& config )
            : ModuleBase( std::make_shared<DeviceContext>( device_name ), config ), config_( config ) {

            config.validate();

            const auto& weights = config.getClassWeights();
            if ( !weights.empty() ) {
                initializeClassWeights( weights );
            }

            createOperation();
        }

        /**
         * @brief Constructs a new CrossEntropy module with a provided device context.
         *
         * Uses a pre-existing DeviceContext instance. This constructor is useful when integrating
         * the module into a larger network that shares device contexts across modules.
         *
         * @param device_context The device context to use for this module.
         * @param config Configuration parameters for the CrossEntropy module.
         * @throws std::invalid_argument If device_context is null or configuration is invalid
         * @throws std::runtime_error If device context type doesn't match template parameter TDeviceType
         */
        explicit CrossEntropy( std::shared_ptr<DeviceContext> device_context, const CrossEntropyConfig& config )
            : ModuleBase( device_context, config ), config_( config ) {

            config.validate();

            const auto& weights = config.getClassWeights();
            if ( !weights.empty() ) {
                initializeClassWeights( weights );
            }

            createOperation();
        }

        /**
         * @brief Gets the number of trainable parameters in this module.
         *
         * Returns the number of elements in the class weights tensor if present,
         * otherwise returns 0 since CrossEntropy doesn't have trainable parameters.
         *
         * @return size_t The total number of parameters.
         */
        size_t parameterCount() const override {
            return class_weights_ ? class_weights_->size() : 0;
        }

        /**
         * @brief Performs the forward pass of the cross entropy operation.
         *
         * Computes the cross entropy loss between the predicted logits and target indices.
         * The operation applies any configured options such as class weighting, padding
         * index ignoring, and label smoothing.
         *
         * @param input The input tensor containing predicted logits.
         * @param targets The target tensor containing class indices.
         * @param output The output tensor that will contain the loss value(s).
         */
        void forward( const Tensor<TLogits, MR>& input, const Tensor<TTargets, MR>& targets, Tensor<TLogits, MR>& output ) {
            operation_->forward( input, targets, parameters_, output, output_state_ );
        }

        /**
         * @brief Calculates gradients for the backward pass.
         *
         * Computes the gradient of the cross entropy loss with respect to the input logits.
         * This gradient is used during backpropagation to update network weights.
         *
         * @param input The input tensor from the forward pass (logits).
         * @param targets The target tensor containing class indices.
         * @param output_grad The gradient of loss with respect to the output.
         * @param input_grad The tensor to store gradients with respect to input.
         */
        void backward(
            const Tensor<TLogits, MR>& input,
            const Tensor<TTargets, MR>& targets,
            const Tensor<TLogits, MR>& output_grad,
            Tensor<TLogits, MR>& input_grad ) {

            operation_->backward(
                input,           // Input tensor (logits)
                targets,         // Target classes
                parameters_,     // Optional class weights
                {},              // No parameter gradients needed
                output_grad,     // Gradient from next layer
                input_grad,      // Gradient to propagate to logits
                output_state_    // Cached tensors from forward pass
            );
        }

        /**
         * @brief Gets the vocabulary size.
         *
         * @return int64_t The number of possible classes.
         */
        int64_t getVocabSize() const {
            return config_.getVocabSize();
        }

        /**
         * @brief Checks if padding is ignored.
         *
         * @return bool True if padding indices are ignored, false otherwise.
         */
        bool ignorePadding() const {
            return config_.ignorePadding();
        }

        /**
         * @brief Gets the padding index.
         *
         * @return int64_t The index value that represents padding.
         */
        int64_t getPaddingIndex() const {
            return config_.getPaddingIndex();
        }

        /**
         * @brief Checks if loss is reduced.
         *
         * @return bool True if loss is reduced (mean), false if per-sample losses are returned.
         */
        bool isReduced() const {
            return config_.getReduction();
        }

        /**
         * @brief Gets the label smoothing factor.
         *
         * @return float The label smoothing factor (between 0 and 1).
         */
        float getLabelSmoothing() const {
            return config_.getLabelSmoothing();
        }

        /**
         * @brief Gets the class weights tensor.
         *
         * @return std::shared_ptr<Tensor<TLogits, MR>> The class weights tensor or nullptr if not used.
         */
        std::shared_ptr<Tensor<TLogits, MR>> getClassWeights() const {
            return class_weights_;
        }

        /**
         * @brief Serializes the module state to a ZIP archive.
         *
         * Saves the class weights tensor (if present) to the provided ZIP archive.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( ModelArchive& archive ) const override {
            if ( class_weights_ ) {
                // Save class weights if present
                // Implementation depends on tensor serialization
            }
        }

        /**
         * @brief Deserializes the module state from a ZIP archive.
         *
         * Loads the class weights tensor (if expected) from the provided ZIP archive.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( ModelArchive& archive ) override {
            if ( class_weights_ ) {
                // Load class weights if present
                // Implementation depends on tensor deserialization
            }
        }

        /**
         * @brief Generates a string representation of this module's configuration.
         *
         * @return std::string A formatted string with module information
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "CrossEntropy: " << this->getName() << std::endl;
            oss << "Vocabulary Size: " << config_.getVocabSize() << std::endl;

            if ( config_.ignorePadding() ) {
                oss << "Ignoring padding at index: " << config_.getPaddingIndex() << std::endl;
            }

            if ( !config_.getClassWeights().empty() ) {
                oss << "Using class weights" << std::endl;
            }

            oss << "Reduction: " << (config_.getReduction() ? "Mean" : "None") << std::endl;

            if ( config_.getLabelSmoothing() > 0.0f ) {
                oss << "Label smoothing: " << config_.getLabelSmoothing() << std::endl;
            }

            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;

            return oss.str();
        }

    private:
        /**
         * @brief Configuration for the CrossEntropy module.
         */
        CrossEntropyConfig config_;

        /**
         * @brief Optional tensor containing weights for each class.
         *
         * Used to address class imbalance by giving different importance to different classes.
         */
        std::shared_ptr<Tensor<TLogits, MR>> class_weights_{ nullptr };

        /**
         * @brief Collection of parameters for this module.
         *
         * Only contains class_weights_ if present, otherwise empty.
         */
        std::vector<std::shared_ptr<ITensor>> parameters_;

        /**
         * @brief Collection of output state tensors for caching.
         *
         * Stores intermediate results from forward pass needed for backward pass.
         */
        std::vector<std::shared_ptr<Tensor<TLogits, MR>>> output_state_;

        /**
         * @brief The operation that implements the cross entropy calculation.
         */
        std::shared_ptr<UnaryOperation<TDeviceType, TLogits, TTargets>> operation_{ nullptr };

        /**
         * @brief Initializes the class weights tensor from a vector of weights.
         *
         * Creates and populates a tensor with class weights for weighted cross entropy loss.
         * This is useful for handling imbalanced datasets where some classes are underrepresented.
         *
         * @param weights Vector of weight values, one per class
         */
        void initializeClassWeights( const std::vector<float>& weights ) {
            class_weights_ = std::make_shared<Tensor<TLogits, MR>>(
                std::vector<size_t>{static_cast<size_t>(config_.getVocabSize())} );
            class_weights_->setName( this->getName() + ".class_weights" );

            // Copy the weights into the tensor
            // This is a simplified placeholder - actual implementation would copy the data to device

            parameters_.clear();
            parameters_.push_back( class_weights_ );
        }

        /**
         * @brief Creates the appropriate cross entropy operation for the current device.
         *
         * Instantiates either a CPU or CUDA cross entropy operation based on the device type.
         * Sets operation attributes from the configuration object.
         */
        void createOperation() {
            // Set operation attributes from config
            /*attributes_.set( "vocab_size", config_.getVocabSize() );

            if ( config_.ignorePadding() ) {
                attributes_.set( "ignore_padding", true );
                attributes_.set( "padding_idx", config_.getPaddingIndex() );
            }

            attributes_.set( "reduce", config_.getReduction() );

            if ( config_.getLabelSmoothing() > 0.0f ) {
                attributes_.set( "label_smoothing", config_.getLabelSmoothing() );
            }*/

            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TLogits, TTargets>(
                    "Cpu::CrossEntropyOp",
                    this->getDeviceContext(),
                    config_ );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TLogits, TTargets>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TLogits, TTargets>(
                    "Cuda::CrossEntropyOp",
                    this->getDeviceContext(),
                    config_ );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TLogits, TTargets>>(base_op);
            }
        }
    };

    /**
     * @brief Type alias for CPU-based cross entropy module with customizable tensor types.
     *
     * @tparam TLogits Data type of the input logits tensor elements.
     * @tparam TTargets Data type of the target indices, typically int.
     */
    export template<typename TLogits = float, typename TTargets = int>
        using CpuCrossEntropy = CrossEntropy<DeviceType::Cpu, TLogits, TTargets>;

    /**
     * @brief Type alias for CUDA-based cross entropy module with customizable tensor types.
     *
     * @tparam TLogits Data type of the input logits tensor elements.
     * @tparam TTargets Data type of the target indices, typically int.
     */
    export template<typename TLogits = float, typename TTargets = int>
        using CudaCrossEntropy = CrossEntropy<DeviceType::Cuda, TLogits, TTargets>;
}