/**
 * @file MnistClassifier.ixx
 * @brief MNIST digit classifier using feedforward neural network architecture.
 *
 * Device-templated network implementing a three-layer neural network
 * for MNIST handwritten digit classification (784 -> 128 -> 64 -> 10).
 */

module;
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <ostream>
#include <format>
#include <utility>
#include <optional>

export module Mnist.Classifier;

import Mila;
import Mnist.DataLoader;
import Dnn.Network;

namespace Mila::Mnist
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief MNIST handwritten digit classifier.
     *
     * Three-layer feed-forward neural network architecture:
     *   Input (784) -> Linear (128) -> GELU -> Linear (64) -> GELU -> Linear (10) -> Output
     *
     * Construction Pattern:
     * 1. Constructor creates and owns ExecutionContext
     * 2. Component graph is built via createGraph() (context-independent)
     * 3. Context is propagated to self and all children via setExecutionContext()
     * 4. onBuilding() hook builds children with shapes and allocates buffers
     *
     * Serialization Contract:
     * - Implements save_() override to write type identifier and configuration
     * - Provides static Load() factory method for type-safe deserialization
     * - Base Network class handles component graph topology serialization
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class MnistClassifier : public Network<TDeviceType, TPrecision>
    {
    public:

        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using NetworkBase = Network<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using GeluType = Gelu<TDeviceType, TPrecision>;
        using ComponentPtr = typename NetworkBase::ComponentPtr;

        /**
         * @brief Construct MNIST classifier network.
         *
         * Follows the concrete network construction pattern:
         * 1. Create and own ExecutionContext for specified device
         * 2. Build component graph (context-independent)
         * 3. Propagate context to self and all children
         *
         * @param name Classifier name for identification and serialization
         * @param batch_size Batch size for training/inference
         * @param device_id Device identifier for network execution
         *
         * @throws std::invalid_argument if batch_size <= 0
         * @throws std::invalid_argument if device_id.type does not match TDeviceType
         * @throws std::runtime_error if ExecutionContext creation fails
         */
        explicit MnistClassifier(const std::string& name,
                                 int64_t batch_size,
                                 DeviceId device_id)
            : NetworkBase(name),
              owned_context_(createExecutionContext(device_id)),
              batch_size_(batch_size)
        {
            if (batch_size_ <= 0)
            {
                throw std::invalid_argument("MnistClassifier: batch size must be positive");
            }

            if (device_id.type != TDeviceType)
            {
                throw std::invalid_argument(
                    std::format("MnistClassifier: device type mismatch: expected {}, got {}",
                                deviceTypeToString(TDeviceType),
                                deviceTypeToString(device_id.type)));
            }

            // Build component graph (context-independent)
            createGraph();

            // Propagate context to self and all children
            this->setExecutionContext(owned_context_.get());
        }

        ~MnistClassifier() override = default;

        /**
         * @brief Load MnistClassifier from archive.
         *
         * Static factory method for type-safe deserialization. Reconstructs
         * the classifier by:
         * 1. Reading configuration from archive metadata
         * 2. Constructing via normal constructor (creates graph + context)
         * 3. Building with saved input shape
         * 4. Loading component weights into built components
         *
         * @param archive Archive containing serialized classifier
         * @param device_id Device for execution (may differ from saved device)
         * @return Unique pointer to reconstructed MnistClassifier
         *
         * @throws std::runtime_error if archive is malformed
         * @throws std::runtime_error if configuration is invalid
         */
        static std::unique_ptr<MnistClassifier> Load(ModelArchive& archive, DeviceId device_id)
        {
            // Read classifier-specific metadata
            //json meta = archive.readJson("network/classifier_meta.json");

            std::string name = "name";// meta.at( "name" );
            int64_t batch_size = 4; // meta.at( "batch_size" );

            //// Validate architecture constants match
            //int64_t saved_hidden1 = meta.at("hidden1_size");
            //int64_t saved_hidden2 = meta.at("hidden2_size");
            //int64_t saved_classes = meta.at("num_classes");

            //if (saved_hidden1 != HIDDEN1_SIZE || saved_hidden2 != HIDDEN2_SIZE || 
            //    saved_classes != MNIST_NUM_CLASSES)
            //{
            //    throw std::runtime_error(
            //        std::format("MnistClassifier::Load: architecture mismatch. "
            //                    "Archive has [{}, {}, {}], expected [{}, {}, {}]",
            //                    saved_hidden1, saved_hidden2, saved_classes,
            //                    HIDDEN1_SIZE, HIDDEN2_SIZE, MNIST_NUM_CLASSES));
            //}

            // Construct via normal path (creates graph + context)

            auto classifier = std::make_unique<MnistClassifier>(name, batch_size, device_id);

            // Build with saved input shape
            //shape_t input_shape = meta.at("input_shape");
            //classifier->build(input_shape);

            // Load component weights (base class handles graph traversal)
            //loadComponentWeights(archive, classifier.get());

            return classifier;
        }

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to child components.
         *
         * All setup and validation was done in onBuilding(). This method chains
         * forward calls through the classifier structure using pre-allocated buffers.
         *
         * @param input Input tensor containing flattened MNIST images (batch_size, 784)
         * @param output Output tensor for class logits (batch_size, 10)
         *
         * @throws std::runtime_error if classifier has not been built
         */
        void forward(const ITensor& input, ITensor& output)
        {
            if (!this->isBuilt())
            {
                throw std::runtime_error("MnistClassifier: must be built before forward pass");
            }

            fc1_->forward(input, *hidden1_pre_act_);
            gelu1_->forward(*hidden1_pre_act_, *hidden1_);
            fc2_->forward(*hidden1_, *hidden2_pre_act_);
            gelu2_->forward(*hidden2_pre_act_, *hidden2_);
            output_fc_->forward(*hidden2_, output);
        }

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to child components.
         *
         * Chains backward calls through the classifier structure in reverse order.
         *
         * @param input Original forward input tensor
         * @param output_grad Gradient w.r.t. classifier output
         * @param input_grad Gradient w.r.t. classifier input (output)
         *
         * @throws std::runtime_error if classifier has not been built
         */
        void backward(const ITensor& input, const ITensor& output_grad, ITensor& input_grad)
        {
            if (!this->isBuilt())
            {
                throw std::runtime_error("MnistClassifier: must be built before backward pass");
            }

            auto device = this->getDeviceId();

            TensorType hidden2_grad(device, hidden2_shape_);
            zeros(hidden2_grad);
            output_fc_->backward(*hidden2_, output_grad, hidden2_grad);

            TensorType hidden2_pre_grad(device, hidden2_shape_);
            zeros(hidden2_pre_grad);
            gelu2_->backward(*hidden2_pre_act_, hidden2_grad, hidden2_pre_grad);

            TensorType hidden1_grad(device, hidden1_shape_);
            zeros(hidden1_grad);
            fc2_->backward(*hidden1_, hidden2_pre_grad, hidden1_grad);

            TensorType hidden1_pre_grad(device, hidden1_shape_);
            zeros(hidden1_pre_grad);
            gelu1_->backward(*hidden1_pre_act_, hidden1_grad, hidden1_pre_grad);

            fc1_->backward(input, hidden1_pre_grad, input_grad);
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        DeviceId getDeviceId() const override
        {
            return NetworkBase::getDeviceId();
        }

        void synchronize() override
        {
            NetworkBase::synchronize();
        }

        size_t parameterCount() const override
        {
            return NetworkBase::parameterCount();
        }

        std::vector<ITensor*> getParameters() const override
        {
            return NetworkBase::getParameters();
        }

        std::vector<ITensor*> getGradients() const override
        {
            return NetworkBase::getGradients();
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << std::endl;
            oss << "MNIST Classifier: " << this->getName() << std::endl;
            oss << "Device: " << getDeviceId().toString() << std::endl;

            oss << "Architecture:" << std::endl;
            oss << "  Input:   784 features (28x28 flattened)" << std::endl;
            oss << "  Layer 1: 784 -> " << HIDDEN1_SIZE << " + GELU" << std::endl;
            oss << "  Layer 2: " << HIDDEN1_SIZE << " -> " << HIDDEN2_SIZE << " + GELU" << std::endl;
            oss << "  Output:  " << HIDDEN2_SIZE << " -> " << MNIST_NUM_CLASSES << " classes" << std::endl;

            if (this->isBuilt())
            {
                oss << "Parameters: " << parameterCount() << std::endl;
            }

            oss << "Batch size: " << batch_size_ << std::endl;

            if (this->isBuilt())
            {
                oss << "Input shape: (";
                for (size_t i = 0; i < input_shape_.size(); ++i)
                {
                    oss << input_shape_[i];
                    if (i != input_shape_.size() - 1)
                        oss << ", ";
                }
                oss << ")" << std::endl;

                oss << "Output shape: (";
                for (size_t i = 0; i < output_shape_.size(); ++i)
                {
                    oss << output_shape_[i];
                    if (i != output_shape_.size() - 1)
                        oss << ", ";
                }
                oss << ")" << std::endl;
            }

            oss << "Components: " << std::endl;

            if (fc1_)
            {
                oss << "  - fc1: " << fc1_->getName() << std::endl;
            }

            if (gelu1_)
            {
                oss << "  - gelu1: " << gelu1_->getName() << std::endl;
            }

            if (fc2_)
            {
                oss << "  - fc2: " << fc2_->getName() << std::endl;
            }

            if (gelu2_)
            {
                oss << "  - gelu2: " << gelu2_->getName() << std::endl;
            }

            if (output_fc_)
            {
                oss << "  - output: " << output_fc_->getName() << std::endl;
            }

            oss << std::endl;

            return oss.str();
        }

        IExecutionContext* getExecutionContext() const
        {
            return NetworkBase::getExecutionContext();
        }

    protected:

        /**
         * @brief Save classifier-specific configuration (required by Network base).
         *
         * Implements the serialization contract by writing type identifier
         * and configuration metadata to enable reconstruction via Load().
         *
         * @param archive Archive to write to
         * @param mode Serialization mode (passed from Network::save())
         */
        void save_(ModelArchive& archive, SerializationMode /*mode*/) const override
        {
            //meta;
            //meta["type"] = "MnistClassifier";  // Type identifier for runtime dispatch
            //meta["version"] = 1;
            //meta["name"] = this->getName();
            //meta["batch_size"] = batch_size_;

            //// Save shape metadata for validation
            //if (this->isBuilt())
            //{
            //    meta["input_shape"] = input_shape_;
            //    meta["output_shape"] = output_shape_;
            //    meta["hidden1_shape"] = hidden1_shape_;
            //    meta["hidden2_shape"] = hidden2_shape_;
            //}

            //// Save architecture constants for validation
            //meta["hidden1_size"] = HIDDEN1_SIZE;
            //meta["hidden2_size"] = HIDDEN2_SIZE;
            //meta["num_classes"] = MNIST_NUM_CLASSES;

            //archive.writeJson("network/classifier_meta.json", meta);
        }

        /**
         * @brief Create the MNIST classifier network graph (context-independent).
         *
         * Defines the computational graph:
         *   fc1 -> gelu1 -> fc2 -> gelu2 -> output
         *
         * Components are created without ExecutionContext (shared mode).
         * Context will be propagated after this method returns via setExecutionContext().
         */
        void createGraph()
        {
            addLinear("fc1", MNIST_IMAGE_SIZE, HIDDEN1_SIZE);
            addActivation("gelu1");
            addLinear("fc2", HIDDEN1_SIZE, HIDDEN2_SIZE);
            addActivation("gelu2");
            addLinear("output", HIDDEN2_SIZE, MNIST_NUM_CLASSES);
        }

        /**
         * @brief Hook invoked during build() to initialize classifier with input shape.
         *
         * Validates input shape, computes per-layer shapes, caches typed pointers to children,
         * builds all child components with appropriate shapes, and allocates intermediate buffers.
         *
         * All children have ExecutionContext at this point (propagated in constructor).
         *
         * @param input_shape Expected input tensor shape
         *
         * @throws std::invalid_argument if input_shape is invalid for MNIST
         */
        void onBuilding(const shape_t& input_shape) override
        {
            validateInputShape(input_shape);

            input_shape_ = input_shape;

            hidden1_shape_ = input_shape;
            hidden1_shape_.back() = HIDDEN1_SIZE;

            hidden2_shape_ = input_shape;
            hidden2_shape_.back() = HIDDEN2_SIZE;

            output_shape_ = input_shape;
            output_shape_.back() = MNIST_NUM_CLASSES;

            // Cache typed pointers to children
            fc1_ = this->template getComponentAs<LinearType>(this->getName() + ".fc1");
            fc1_->build(input_shape);

            gelu1_ = this->template getComponentAs<GeluType>(this->getName() + ".gelu1");
            gelu1_->build(hidden1_shape_);

            fc2_ = this->template getComponentAs<LinearType>(this->getName() + ".fc2");
            fc2_->build(hidden1_shape_);

            gelu2_ = this->template getComponentAs<GeluType>(this->getName() + ".gelu2");
            gelu2_->build(hidden2_shape_);

            output_fc_ = this->template getComponentAs<LinearType>(this->getName() + ".output");
            output_fc_->build(hidden2_shape_);

            // Allocate intermediate activation buffers
            auto device = this->getDeviceId();

            hidden1_pre_act_ = std::make_shared<TensorType>(device, hidden1_shape_);
            hidden1_pre_act_->setName(this->getName() + ".hidden1_pre_act");

            hidden1_ = std::make_shared<TensorType>(device, hidden1_shape_);
            hidden1_->setName(this->getName() + ".hidden1");

            hidden2_pre_act_ = std::make_shared<TensorType>(device, hidden2_shape_);
            hidden2_pre_act_->setName(this->getName() + ".hidden2_pre_act");

            hidden2_ = std::make_shared<TensorType>(device, hidden2_shape_);
            hidden2_->setName(this->getName() + ".hidden2");
        }

    private:

        static constexpr int64_t HIDDEN1_SIZE = 128;
        static constexpr int64_t HIDDEN2_SIZE = 64;

        // Owned ExecutionContext (concrete class responsibility)
        std::unique_ptr<IExecutionContext> owned_context_{nullptr};

        // Configuration
        int64_t batch_size_;

        // Computed shapes (cached during build)
        shape_t input_shape_;
        shape_t hidden1_shape_;
        shape_t hidden2_shape_;
        shape_t output_shape_;

        // Typed component pointers (cached during onBuilding)
        std::shared_ptr<LinearType> fc1_{nullptr};
        std::shared_ptr<GeluType> gelu1_{nullptr};
        std::shared_ptr<LinearType> fc2_{nullptr};
        std::shared_ptr<GeluType> gelu2_{nullptr};
        std::shared_ptr<LinearType> output_fc_{nullptr};

        // Activation buffers (allocated during onBuilding)
        std::shared_ptr<TensorType> hidden1_pre_act_{nullptr};
        std::shared_ptr<TensorType> hidden1_{nullptr};
        std::shared_ptr<TensorType> hidden2_pre_act_{nullptr};
        std::shared_ptr<TensorType> hidden2_{nullptr};

        /**
         * @brief Helper to create and register a linear layer child component.
         *
         * @param suffix Component name suffix (will be prefixed with classifier name)
         * @param in_features Input feature dimension
         * @param out_features Output feature dimension
         */
        void addLinear(const std::string& suffix, dim_t in_features, dim_t out_features)
        {
            auto cfg = LinearConfig(in_features, out_features).withBias(false);

            auto component = std::make_shared<LinearType>(
                this->getName() + "." + suffix, cfg, std::nullopt);

            this->addComponent(component);
        }

        /**
         * @brief Helper to create and register an activation layer child component.
         *
         * @param suffix Component name suffix
         */
        void addActivation(const std::string& suffix)
        {
            auto cfg = GeluConfig();

            auto component = std::make_shared<GeluType>(
                this->getName() + "." + suffix, cfg, std::nullopt);

            this->addComponent(component);
        }

        /**
         * @brief Validate input shape for MNIST classifier.
         *
         * @throws std::invalid_argument if shape is invalid
         */
        void validateInputShape(const shape_t& input_shape) const
        {
            if (input_shape.empty())
            {
                throw std::invalid_argument("MnistClassifier: input must have rank >= 1");
            }

            int64_t input_features = input_shape.back();

            if (input_features != MNIST_IMAGE_SIZE)
            {
                throw std::invalid_argument(
                    std::format("MnistClassifier: input feature dimension mismatch. "
                                "Expected {} (28x28 flattened), got {}",
                                MNIST_IMAGE_SIZE, input_features));
            }
        }

        /**
         * @brief Load component weights from archive.
         *
         * Helper method for Load() to populate weights into already-built components.
         * Base Network class handles component graph traversal; this method loads
         * weights into each component.
         *
         * @param archive Archive containing serialized weights
         * @param classifier Classifier instance with built components
         *
         * @note This is a placeholder - actual implementation will iterate over
         *       components and load their weights from the archive.
         */
        static void loadComponentWeights(ModelArchive& /*archive*/, 
                                         MnistClassifier* /*classifier*/)
        {
            // TODO: Implement weight loading
            // For each component in classifier->getComponents():
            //   - Read component weights from archive
            //   - Set weights on component via setParameters()
        }
    };
}