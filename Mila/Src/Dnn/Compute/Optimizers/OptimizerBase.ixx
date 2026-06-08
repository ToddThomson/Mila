/**
 * @file OptimizerBase.ixx
 * @brief Base interface for neural network parameter optimizers.
 *
 * Defines the abstract Optimizer class template that provides a uniform interface
 * for parameter update algorithms (SGD, Adam, AdamW, etc.) across different devices
 * and precision levels.
 */

module;
#include <memory>

export module Compute.OptimizerBase;

import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;

// TJT: TODO: Optimizer should be in Mila::Dnn . It is a core DNN concept, not just Compute.
namespace Mila::Dnn::Compute
{
    /**
     * @brief Abstract base class for parameter optimizers.
     *
     * Optimizers update model parameters using computed gradients according to
     * specific update rules (SGD, Adam, AdamW, etc.). The optimizer:
     * - Maintains internal state per parameter (momentum, velocity, etc.)
     * - Performs parameter updates via step()
     *
     * Template Parameters:
     * @tparam TDeviceType Device where optimization occurs (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType::FP32, FP16, BF16)
     *
     * Typical usage pattern:
     * @code
     * // Create optimizer
     * auto optimizer = std::make_shared<AdamWOptimizer<DeviceType::Cuda, TensorDataType::FP32>>(
     *     learning_rate, beta1, beta2, epsilon, weight_decay);
     *
     * // Register parameters
     * auto params = model->getParameters();
     * auto grads = model->getGradients();
     * for (size_t i = 0; i < params.size(); ++i) {
     *     optimizer->addParameter(params[i], grads[i]);
     * }
     *
     * // Training loop
     * for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
     *     model->zeroGradients();           // Clear model-owned gradients (activation + parameter grads)
     *     model->forward(input, output);    // Forward pass
     *     model->backward(input, grad);     // Compute gradients
     *     optimizer->step();                // Update parameters
     * }
     * @endcode
     *
     * Implementation Requirements:
     * - Derived classes must handle device-specific memory and execution
     * - State tensors (momentum, variance) must reside on same device as parameters
     * - Thread-safety is not guaranteed; synchronize externally if needed
     * - Parameters must remain valid for optimizer lifetime
     *
     * @note Parameters and gradients are stored as weak references; the Module
     *       retains ownership and is responsible for parameter lifetime.
     * @note Implementations should support asynchronous execution where possible
     *       (e.g., CUDA streams) without requiring explicit synchronization in step().
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
    class Optimizer
    {
    public:
        virtual ~Optimizer() = default;

        /**
         * @brief Register a parameter tensor for optimization.
         *
         * Adds a parameter-gradient pair to the optimizer's update list. The
         * optimizer will allocate internal state tensors (momentum, variance, etc.)
         * matching the parameter shape and device placement.
         *
         * @param param Shared pointer to parameter tensor to be optimized
         * @param grad Shared pointer to gradient tensor (must match param shape)
         *
         * @throws std::invalid_argument if param or grad is nullptr
         * @throws std::invalid_argument if param and grad shapes don't match
         * @throws std::invalid_argument if param and grad are on different devices
         * @throws std::runtime_error if state allocation fails
         *
         * @note Must be called after model->build() when parameter shapes are known
         * @note Parameter and gradient must persist for the optimizer's lifetime
         * @note Calling multiple times with same parameter updates the gradient reference
         * @note State tensors are initialized to zero on first registration
         *
         * @see step()
         *
         * Example:
         * @code
         * auto params = model->getParameters();
         * auto grads = model->getGradients();
         *
         * for (size_t i = 0; i < params.size(); ++i) {
         *     optimizer->addParameter(params[i], grads[i]);
         * }
         * @endcode
         */
        virtual void addParameter( ITensor* param, ITensor* grad ) = 0;

        /**
         * @brief Perform one optimization step.
         *
         * Updates all registered parameters using their accumulated gradients
         * according to the optimizer's update rule (SGD, Adam, AdamW, etc.). 
         * This is the HOT PATH method called every training iteration.
         *
         * For algorithms with state (Adam, AdamW):
         * - Updates first and second moment estimates
         * - Applies bias correction if needed
         * - Computes parameter update
         * - Writes updated parameters back to tensors
         *
         * @throws std::runtime_error if no parameters have been registered
         * @throws std::runtime_error if gradient data is invalid or null
         *
         * @note Gradients should be computed via backward() before calling step()
         * @note For CUDA implementations, may be asynchronous (uses device stream)
         * @note Increments internal step counter for algorithms requiring it (Adam, AdamW)
         *
         * @see addParameter()
         * @see backward()
         *
         * Typical sequence:
         * @code
         * model->zeroGradients();              // Clear previous gradients (model-managed)
         * model->forward(input, output);       // Forward pass
         * loss = computeLoss(output, target);
         * model->backward(input, loss_grad);   // Compute gradients
         * optimizer->step();                   // Update parameters
         * @endcode
         */
        virtual void step() = 0;

        /**
         * @brief Get the current learning rate.
         *
         * Returns the base learning rate used for parameter updates. Some optimizers
         * may apply adaptive per-parameter learning rates internally (Adam, AdamW),
         * but this method returns the global scaling factor.
         *
         * @return Current learning rate as a float
         *
         * @note For adaptive optimizers, actual effective learning rate per parameter
         *       may differ due to momentum and variance scaling
         *
         * @see setLearningRate()
         */
        virtual float getLearningRate() const = 0;

        /**
         * @brief Set the learning rate for future updates.
         *
         * Updates the base learning rate used by the optimizer. Typically used for
         * learning rate schedules (decay, warmup, cyclic, etc.).
         *
         * @param learning_rate New learning rate (must be positive)
         *
         * @throws std::invalid_argument if learning_rate <= 0
         *
         * @note Takes effect immediately for the next step() call
         * @note Does not affect optimizer state (momentum, variance)
         * @note For learning rate schedules, call this at epoch or iteration boundaries
         *
         * @see getLearningRate()
         *
         * Example with learning rate decay:
         * @code
         * float initial_lr = 0.001f;
         * optimizer->setLearningRate(initial_lr);
         *
         * for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
         *     // Training loop...
         *
         *     // Decay learning rate every 10 epochs
         *     if (epoch > 0 && epoch % 10 == 0) {
         *         float new_lr = optimizer->getLearningRate() * 0.5f;
         *         optimizer->setLearningRate(new_lr);
         *         std::cout << "Learning rate: " << new_lr << std::endl;
         *     }
         * }
         * @endcode
         */
        virtual void setLearningRate( float learning_rate ) = 0;
    };
}