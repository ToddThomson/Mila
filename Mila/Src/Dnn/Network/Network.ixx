/**
 * @file Network.ixx
 * @brief Root composite network container.
 *
 * Provides a CompositeComponent-derived container representing a complete neural
 * network model. Network serves as the top-level entry point and delegates
 * ExecutionContext ownership and deserialization to concrete subclasses.
 */

module;
#include <memory>
#include <string>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <chrono>
#include <algorithm>
#include <vector>
#include <exception>
#include <format>

export module Dnn.Network;

import Dnn.CompositeComponent;
import Dnn.TensorDataType;
import Compute.DeviceType;
import Compute.DeviceId;
import Compute.OptimizerBase;
import Serialization.ModelArchive;
import Serialization.Mode;
import Serialization.Metadata;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Root composite network container.
     *
     * Network is a specialized CompositeComponent that represents a complete neural
     * network model and serves as the top-level entry point. It provides high-level
     * serialization semantics while delegating lifecycle management to concrete subclasses.
     *
     * Ownership Model:
     * - **Concrete subclasses own ExecutionContext** (flexibility for construction ordering)
     * - Network receives non-owning context pointer via setExecutionContext()
     * - All child components share the Network's ExecutionContext (efficient resource pooling)
     *
     * Construction Pattern (Concrete Subclass):
     * @code
     * class MnistClassifier : public Network<DeviceType::Cpu, TensorDataType::FP32>
     * {
     * public:
     *     explicit MnistClassifier(const std::string& name, int64_t batch_size, DeviceId device_id)
     *         : Network(name),
     *           owned_context_(createExecutionContext(device_id)),
     *           batch_size_(batch_size)
     *     {
     *         // 1. Create component graph (context-independent)
     *         createGraph();
     *         
     *         // 2. Propagate context to self and children
     *         this->setExecutionContext(owned_context_.get());
     *     }
     * 
     * private:
     *     std::unique_ptr<IExecutionContext> owned_context_;  // Concrete class owns context
     * };
     * @endcode
     *
     * Serialization Contract:
     * - **Base class (Network)**: Saves component graph topology and generic metadata
     * - **Concrete class**: MUST override save_() to write type identifier and configuration
     * - **Concrete class**: MUST provide static Load() factory method for deserialization
     *
     * Deserialization Pattern (Concrete Subclass):
     * @code
     * // REQUIRED: Static factory method for type-safe deserialization
     * static std::unique_ptr<MnistClassifier> Load(ModelArchive& archive, DeviceId device_id)
     * {
     *     // 1. Read concrete-specific metadata
     *     json meta = archive.readJson("network/classifier_meta.json");
     *     std::string name = meta.at("name");
     *     int64_t batch_size = meta.at("batch_size");
     *     
     *     // 2. Construct via normal constructor path
     *     auto classifier = std::make_unique<MnistClassifier>(name, batch_size, device_id);
     *     
     *     // 3. Build with saved input shape
     *     shape_t input_shape = meta.at("input_shape");
     *     classifier->build(input_shape);
     *     
     *     // 4. Load component weights
     *     // (Base class handles graph traversal; weights loaded into already-built components)
     *     
     *     return classifier;
     * }
     * @endcode
     *
     * Design Rationale:
     * - Concrete classes control infrastructure (context) lifecycle
     * - Network base class focuses on container semantics and serialization
     * - Clear initialization order: create context ? pass to base ? build graph
     * - Enables future flexibility (custom contexts, multi-device, etc.)
     * - Type-safe deserialization via concrete class Load() methods
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
    class Network : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using CompositeBase = CompositeComponent<TDeviceType, TPrecision>;
        using ComponentPtr = typename CompositeBase::ComponentPtr;

        /**
         * @brief Construct network (context managed by derived class).
         *
         * Base constructor for concrete network classes. Derived classes are
         * responsible for:
         * 1. Creating and owning ExecutionContext
         * 2. Building the component graph via createGraph()
         * 3. Calling setExecutionContext() to propagate context to children
         *
         * @param name Network name for identification and serialization
         *
         * @throws std::invalid_argument if name is not a valid identifier
         */
        explicit Network(const std::string& name)
            : CompositeBase(name)
        {}

        ~Network() override = default;

        /**
         * @brief Synchronize all child components.
         *
         * Waits for outstanding device operations on all children.
         */
        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        // ====================================================================
        // Training Setup
        // ====================================================================

        /**
         * @brief Create and configure an optimizer for this network's parameters.
         *
         * Factory method that creates an optimizer, enables training mode on the network,
         * and registers all network parameters and gradients in a single atomic operation.
         *
         * Lifecycle:
         * 1. Enables training mode (allocates gradients for all components)
         * 2. Creates optimizer using network's ExecutionContext
         * 3. Registers all network parameters and gradients
         * 4. Returns ready-to-use optimizer
         *
         * Usage Pattern:
         * @code
         * // Build network
         * mnist_net->build(input_shape);
         * 
         * // Create optimizer in one step
         * auto optimizer = mnist_net->createOptimizer<AdamWOptimizer<DeviceType::Cuda, TensorDataType::FP32>>(
         *     AdamWConfig()
         *         .withLearningRate(0.001f)
         *         .withWeightDecay(0.01f)
         * );
         * 
         * // Optimizer is ready to use
         * optimizer->step();
         * @endcode
         *
         * @tparam TOptimizer Optimizer type (e.g., AdamWOptimizer, SGD)
         * @tparam TConfig Optimizer configuration type
         * @param config Optimizer configuration
         * @return Shared pointer to configured and ready-to-use optimizer
         *
         * @throws std::runtime_error if network is not built
         * @throws std::runtime_error if parameter/gradient count mismatch
         *
         * @note This method automatically calls setTraining(true), so explicit
         *       training mode activation is not required.
         */
        template<typename TOptimizer, typename TConfig>
        std::shared_ptr<TOptimizer> createOptimizer( const TConfig& config )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error(
                    std::format(
                        "Network::createOptimizer: Network '{}' must be built before creating optimizer",
                        this->getName() )
                );
            }

            this->setTraining( true );

            auto optimizer = std::make_shared<TOptimizer>(
                this->getExecutionContext(), config );

            auto params = this->getParameters();
            auto grads = this->getGradients();

            if ( params.size() != grads.size() )
            {
                throw std::runtime_error(
                    std::format(
                        "Network::createOptimizer: Parameter/gradient count mismatch for network '{}'. "
                        "Parameters: {}, Gradients: {}",
                        this->getName(), params.size(), grads.size() )
                );
            }

            for ( size_t i = 0; i < params.size(); ++i )
            {
                optimizer->addParameter( params[i], grads[i] );
            }

            return optimizer;
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Save network to archive.
         *
         * Saves component graph structure and delegates to concrete class
         * via save_() hook for type-specific configuration.
         *
         * Archive structure produced:
         * - network/meta.json: Base metadata (name, version, num_components, timestamp)
         * - network/architecture.json: Component topology (names, paths, ordering)
         * - components/<name>/...: Child component state (recursive)
         * - Concrete class writes additional files via save_() override
         *
         * @param archive Archive to write to
         * @param mode Serialization mode (Checkpoint, WeightsOnly, Architecture)
         *
         * @throws std::runtime_error if save_() is not overridden by concrete class
         */
        void save(ModelArchive& archive, SerializationMode mode) const
        {
            saveNetworkMetadata(archive, mode);
            saveComponentGraph(archive, mode);
            
            // Hook: concrete class saves type-specific metadata
            save_(archive, mode);
        }

        /**
         * @brief Generate a human-readable description.
         *
         * @return String representation showing network name and children
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Network: " << this->getName() << " " << CompositeBase::toString();

            return oss.str();
        }

    protected:
        /**
         * @brief Hook for concrete classes to save type-specific state.
         *
         * REQUIRED override for concrete networks. Must write:
         * - Type identifier (e.g., "type": "MnistClassifier")
         * - Configuration parameters (batch_size, architecture constants)
         * - Shape metadata (for validation during Load())
         *
         * This metadata enables the concrete class's Load() method to
         * reconstruct the network.
         *
         * Example implementation:
         * @code
         * void save_(ModelArchive& archive, SerializationMode mode) const override
         * {
         *     json meta;
         *     meta["type"] = "MnistClassifier";  // Type identifier for runtime dispatch
         *     meta["batch_size"] = batch_size_;
         *     meta["input_shape"] = input_shape_;
         *     // ... other configuration
         *     archive.writeJson("network/classifier_meta.json", meta);
         * }
         * @endcode
         *
         * @param archive Archive to write to
         * @param mode Serialization mode (passed from save())
         */
        virtual void save_(ModelArchive& archive, SerializationMode mode) const = 0;

    private:
        /**
         * @brief Save base network metadata.
         *
         * Writes generic metadata that applies to all networks:
         * - format_version: Archive format version (for compatibility checking)
         * - name: Network name
         * - num_components: Component count (for validation)
         * - mode: Serialization mode (Checkpoint/WeightsOnly/Architecture)
         * - export_time: Unix timestamp of serialization
         *
         * @param archive Archive to write to
         * @param mode Serialization mode
         */
        void saveNetworkMetadata( ModelArchive& archive, SerializationMode mode ) const
        {
            SerializationMetadata net_meta;
            net_meta.set( "format_version", int64_t( 1 ) )
                .set( "name", this->getName() )
                .set( "num_components", static_cast<int64_t>(this->childCount()) )
                .set( "mode", serializationModeToString( mode ) );

            auto now = std::chrono::system_clock::now();
            net_meta.set( "export_time", static_cast<int64_t>(
                std::chrono::system_clock::to_time_t( now )) );

            archive.writeMetadata( "network/meta.json", net_meta );
        }
        
        /**
         * @brief Save component graph topology.
         *
         * Writes the component manifest (list of child components) and
         * recursively saves each component's state with scoped namespacing.
         *
         * Archive structure:
         * - network/architecture.json: Component manifest metadata
         * - network/components_list.json: Array of component names (for ordering)
         * - network/component_<name>.json: Individual component descriptor
         * - components/<name>/...: Component state (via recursive save_)
         *
         * Components are saved in deterministic (sorted by name) order for
         * reproducible archives.
         *
         * @param archive Archive to write to
         * @param mode Serialization mode (passed to children)
         */
        void saveComponentGraph( ModelArchive& archive, SerializationMode mode ) const
        {
            const auto& named_map = this->getNamedComponents();
            std::vector<std::string> names;
            names.reserve( named_map.size() );

            for ( const auto& pair : named_map )
            {
                names.push_back( pair.first );
            }

            std::sort( names.begin(), names.end() );

            // Save architecture metadata (component count)
            SerializationMetadata arch_meta;
            arch_meta.set( "num_components", static_cast<int64_t>(names.size()) );
            archive.writeMetadata( "network/architecture.json", arch_meta );

            // Save component descriptors individually
            for ( size_t i = 0; i < names.size(); ++i )
            {
                const auto& nm = names[ i ];

                SerializationMetadata comp_desc;
                comp_desc.set( "name", nm )
                    .set( "path", "components/" + nm )
                    .set( "index", static_cast<int64_t>( i ) );

                archive.writeMetadata( "network/component_" + nm + ".json", comp_desc );
            }

            // Recursively save each child component
            for ( const auto& nm : names )
            {
                auto it = named_map.find( nm );

                if ( it == named_map.end() )
                {
                    throw std::runtime_error(
                        "Network::save: inconsistent component map for '" + nm + "'" );
                }

                const auto& component = it->second;

                try
                {
                    ModelArchive::ScopedScope scope( archive, std::string( "components/" ) + nm );
                    component->save_( archive, mode );
                }
                catch ( const std::exception& e )
                {
                    throw std::runtime_error(
                        std::format(
                            "Network::save: failed saving component '{}' into archive '{}': {}",
                            nm,
                            archive.getFilepath(),
                            e.what()
                        )
                    );
                }
            }
        }
    };
}