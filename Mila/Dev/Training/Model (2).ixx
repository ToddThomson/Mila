/**
 * @file Model.ixx
 * @brief Training and persistence orchestration for device-templated Models.
 *
 * Provides the Model class which coordinates training/evaluation loops,
 * checkpointing, and model export. Model is an optional convenience layer
 * that wraps Network + Optimizer for high-level training workflows.
 */

module;
#include <memory>
#include <filesystem>
#include <chrono>
#include <utility>
#include <stdexcept>
#include <optional>
#include <string>
#include <format>

export module Dnn.Model;

import Dnn.TensorDataType;
import Dnn.TensorTypes;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.DeviceId;
import Compute.OptimizerBase;
import Compute.ExecutionContext;
import Data.DataLoader;
import Dnn.Network;
import Dnn.NetworkFactory;

import Dnn.Optimizers.AdamW;
import Dnn.Optimizers.AdamWConfig;

import Dnn.ModelConfig;
import Modeling.CheckpointManager;
import Modeling.CheckpointMetaData;
import Modeling.TrainingHistory;
import Serialization.ModelArchive;
import Serialization.Metadata;
import Serialization.ZipSerializer;
import Serialization.OpenMode;
import Serialization.Mode;
import Utils.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief High-level training orchestration wrapper for Network + Optimizer.
     *
     * Model is an **optional convenience layer** that coordinates training loops,
     * checkpointing, and evaluation. It is not required for inference or low-level
     * training - Network can be used standalone.
     *
     * Design Philosophy:
     * - **Network is primary** - self-contained, reusable for inference
     * - **Model is optional** - training convenience, not infrastructure manager
     * - **Network owns ExecutionContext** - self-contained resource management
     * - **Model coordinates training** - orchestrates training/eval/checkpointing
     *
     * Ownership Model:
     * - Model takes ownership of Network and Optimizer via unique_ptr
     * - Network already owns its ExecutionContext (created in Network constructor)
     * - Optimizer references Network's parameters (registered via Network::createOptimizer())
     *
     * Construction Pattern:
     * @code
     * // Build network
     * auto mnist_net = std::make_unique<MnistClassifier<DeviceType::Cuda, TensorDataType::FP32>>(
     *     "Mnist", batch_size, device_id );
     * mnist_net->build( input_shape );
     * 
     * // Create model (Network creates optimizer internally)
     * auto model = Model<DeviceType::Cuda, TensorDataType::FP32>::create(
     *     std::move( mnist_net ),
     *     AdamWConfig().withLearningRate( 0.001f ).withWeightDecay( 0.01f ),
     *     ModelConfig().withEpochs( 20 ).withBatchSize( 128 )
     * );
     * 
     * // High-level training
     * auto history = model->train( train_loader, val_loader );
     * model->save( "trained_model.mila" );
     * @endcode
     *
     * Threading / Safety:
     * - Not thread-safe. Callers must serialize access to a Model instance.
     *
     * Template parameters:
     * - TDeviceType: compile-time compute device (CPU, CUDA, ...)
     * - TPrecision: compile-time tensor precision enum
     */
    export template<Compute::DeviceType TDeviceType, Mila::Dnn::TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Model
    {
    public:

        // ====================================================================
        // Factory Methods (Preferred Construction Pattern)
        // ====================================================================

        /**
         * @brief Create a Model with Network and Optimizer in one clean step.
         *
         * Factory method that creates a Model by:
         * 1. Taking ownership of the already-built Network
         * 2. Using Network::createOptimizer() to create and register optimizer
         * 3. Constructing Model with Network + Optimizer + Config
         *
         * This is the **preferred** way to create a Model as it leverages
         * Network::createOptimizer() for automatic training mode setup and
         * parameter registration.
         *
         * Preconditions:
         * - Network must be non-null and already built via network->build()
         * - Network owns its ExecutionContext (created in Network constructor)
         *
         * Lifecycle:
         * 1. Network created and built by caller
         * 2. Model::create() calls Network::createOptimizer()
         * 3. Network::createOptimizer() enables training mode and registers parameters
         * 4. Model constructed with Network + Optimizer
         *
         * @tparam TOptimizer Optimizer type (e.g., AdamWOptimizer)
         * @tparam TOptimizerConfig Optimizer configuration type
         * @param network Unique pointer to built network
         * @param optimizer_config Optimizer configuration
         * @param model_config Model training configuration
         * @return Unique pointer to constructed Model
         *
         * @throws std::invalid_argument if network is null or not built
         *
         * @example
         * auto model = Model<DeviceType::Cuda, TensorDataType::FP32>::create(
         *     std::move( mnist_net ),
         *     AdamWConfig().withLearningRate( 0.001f ),
         *     ModelConfig().withEpochs( 20 )
         * );
         */
        template<typename TOptimizer, typename TOptimizerConfig>
        static std::unique_ptr<Model> create(
            std::unique_ptr<Network<TDeviceType, TPrecision>> network,
            const TOptimizerConfig& optimizer_config,
            const ModelConfig& model_config )
        {
            if ( !network )
            {
                throw std::invalid_argument( "Model::create: network cannot be null" );
            }

            if ( !network->isBuilt() )
            {
                throw std::invalid_argument(
                    std::format( "Model::create: network '{}' must be built before creating Model",
                        network->getName() ) );
            }

            auto optimizer = network->template createOptimizer<TOptimizer>( optimizer_config );

            return std::unique_ptr<Model>( new Model(
                std::move( network ),
                std::move( optimizer ),
                model_config ) );
        }

        /**
         * @brief Load a training checkpoint and reconstruct a Model instance.
         *
         * This factory method reads checkpoint metadata and uses NetworkFactory
         * to reconstruct the network, then loads optimizer state.
         *
         * Preconditions:
         * - Checkpoint archive must be valid and contain required metadata
         *
         * @param filepath Filesystem path to the checkpoint archive
         * @param device_id Device to load network on (may differ from saved device)
         * @return Unique pointer to reconstructed Model instance
         *
         * @throws std::runtime_error if metadata validation fails
         * @throws std::runtime_error if network reconstruction fails
         *
         * @example
         * auto model = Model<DeviceType::Cuda, TensorDataType::FP32>::fromCheckpoint(
         *     "checkpoint_epoch_10.mila",
         *     Device::Cuda( 0 ) );
         */
        static std::unique_ptr<Model> fromCheckpoint(
            const std::string& filepath,
            Mila::Dnn:: DeviceId device_id )
        {
            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath, std::move( serializer ), OpenMode::Read );

            SerializationMetadata model_meta = archive.readMetadata( "model/meta.json" );
            validateModelMetadata<TDeviceType, TPrecision>( model_meta );

            // TODO: NetworkFactory needs to be updated to accept DeviceId instead of ExecutionContext
            // For now, create a temporary context (Network will create its own)
            auto temp_context = std::make_shared<Compute::ExecutionContext<TDeviceType>>( device_id );
            auto network = NetworkFactory::create<TDeviceType, TPrecision>( archive, temp_context );

            // TODO: Read optimizer config from checkpoint
            // For now, create default AdamW config
            auto optimizer_config = Optimizers::AdamWConfig();
            
            // Use Network::createOptimizer() to create and register optimizer
            auto optimizer = network->template createOptimizer<Optimizers::AdamWOptimizer<TDeviceType, TPrecision>>(
                optimizer_config );

            // Load optimizer state from checkpoint
            optimizer->load( archive, "optimizer/" );

            // FIXME: Deserialize config when ModelConfig has serialization support
            ModelConfig config;

            return std::unique_ptr<Model>( new Model(
                std::move( network ),
                std::move( optimizer ),
                config ) );
        }

        // ====================================================================
        // Training and Evaluation
        // ====================================================================

        /**
         * @brief Run the configured training loop.
         *
         * Implements a simple epoch loop driven by `config_`. This method:
         * - Calls trainEpoch() for each epoch
         * - Optionally runs validation and saves checkpoints at configured intervals
         * - Honors early stopping settings in config_
         *
         * Postconditions:
         * - Checkpoints may be written to disk via saveTrainingCheckpoint
         *
         * @param training_loader Dataset reader for training data
         * @param eval_loader Optional dataset reader for validation data
         * @return TrainingHistory that accumulates per-epoch statistics
         *
         * @note Current implementation is a placeholder - trainEpoch() and
         *       validateEpoch() need to be implemented with actual training logic
         */
        template<TensorDataType TInputType, TensorDataType TTargetType, typename TMemoryResource>
        TrainingHistory train(
            Data::DataLoader<TInputType, TTargetType, TMemoryResource>& training_loader,
            std::optional<Data::DataLoader<TInputType, TTargetType, TMemoryResource>*> eval_loader = std::nullopt )
        {
            TrainingHistory history;

            if ( config_.getVerbose() )
            {
                Utils::Logger::info_fmt( "Starting training for {} epochs...", config_.getEpochs() );
            }

            for ( std::size_t epoch = 0; epoch < config_.getEpochs(); ++epoch )
            {
                history.current_epoch = epoch;

                network_->setTraining( true );
                double train_loss = trainEpoch();
                history.train_losses.push_back( train_loss );

                double val_loss = 0.0;
                if ( eval_loader.has_value() )
                {
                    network_->setTraining( false );
                    val_loss = validateEpoch();
                    history.val_losses.push_back( val_loss );

                    if ( val_loss < history.best_val_loss )
                    {
                        history.best_val_loss = val_loss;
                        history.epochs_without_improvement = 0;
                    }
                    else
                    {
                        history.epochs_without_improvement++;
                    }
                }

                if ( config_.getVerbose() )
                {
                    if ( eval_loader.has_value() )
                    {
                        Utils::Logger::info_fmt( "Epoch {}/{}: loss = {:.6f}, val_loss = {:.6f}",
                            epoch + 1, config_.getEpochs(), train_loss, val_loss );
                    }
                    else
                    {
                        Utils::Logger::info_fmt( "Epoch {}/{}: loss = {:.6f}",
                            epoch + 1, config_.getEpochs(), train_loss );
                    }
                }

                if ( (epoch + 1) % config_.getCheckpointFrequency() == 0 )
                {
                    Modeling::CheckpointMetadata metadata{
                        .epoch = epoch,
                        .train_loss = train_loss,
                        .val_loss = val_loss,
                        .timestamp = std::chrono::system_clock::now(),
                        .filepath = {}
                    };

                    saveTrainingCheckpoint( metadata );
                }

                if ( config_.getEarlyStoppingEnabled() &&
                    history.epochs_without_improvement >= config_.getEarlyStoppingPatience() )
                {
                    if ( config_.getVerbose() )
                    {
                        Utils::Logger::info_fmt( "Early stopping triggered after {} epochs without improvement",
                            history.epochs_without_improvement );
                    }
                    break;
                }
            }

            return history;
        }

        /**
         * @brief Evaluate the model on a dataset.
         *
         * Placeholder: current implementation returns 0.0. Replace with dataset-driven
         * evaluation logic when integrating DatasetReader/Loader.
         *
         * @return Computed evaluation metric (loss) as double
         */
        double evaluate( /* DatasetReader& test_data */ )
        {
            // TODO: Implement evaluation loop
            return 0.0;
        }

        // ====================================================================
        // Serialization and Checkpointing
        // ====================================================================

        /**
         * @brief Save a full training checkpoint to the provided filepath.
         *
         * Writes a ZIP archive with:
         * - model/meta.json (using SerializationMetadata)
         * - network/* (delegated to Network::save)
         * - optimizer/* (delegated to optimizer->save)
         * - model/config.json (using SerializationMetadata)
         *
         * Preconditions:
         * - The archive path must be writable
         *
         * @param filepath Filesystem path where checkpoint is written
         */
        void saveCheckpoint( const std::string& filepath ) const
        {
            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath, std::move( serializer ), OpenMode::Write );

            SerializationMetadata model_meta;
            model_meta.set( "model_version", int64_t( 1 ) )
                      .set( "device", deviceTypeToString( TDeviceType ) )
                      .set( "precision", "FP32" ) // FIXME: precisionToString( TPrecision )
                      .set( "framework_version", int64_t( 1 ) ); // MILA_VERSION
            
            archive.writeMetadata( "model/meta.json", model_meta );

            network_->save( archive, SerializationMode::Checkpoint );

            optimizer_->save( archive, "optimizer/" );

            // FIXME: Serialize config when ModelConfig has serialization support
            // SerializationMetadata cfg = config_.toMetadata();
            // archive.writeMetadata( "model/config.json", cfg );

            archive.close();
        }

        /**
         * @brief Export a model artifact intended for inference.
         *
         * Produces a compact archive containing model metadata and weights only.
         * The exported archive is validated by loadModel() via the export_mode
         * metadata flag.
         *
         * @param filepath Path to write exported model file
         */
        void save( const std::string& filepath ) const
        {
            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath, std::move( serializer ), OpenMode::Write );

            SerializationMetadata model_meta;
            model_meta.set( "model_version", int64_t( 1 ) )
                      .set( "device", deviceTypeToString( TDeviceType ) )
                      .set( "precision", "FP32" ) // FIXME: precisionToString( TPrecision )
                      .set( "export_mode", true );
            
            archive.writeMetadata( "model/meta.json", model_meta );

            network_->save( archive, SerializationMode::WeightsOnly );

            archive.close();
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        /**
         * @brief Access the owned network (const).
         *
         * @return Reference to the internal Network instance
         */
        const Network<TDeviceType, TPrecision>& network() const
        {
            return *network_;
        }

        /**
         * @brief Access the owned network (mutable).
         *
         * @return Reference to the internal Network instance
         */
        Network<TDeviceType, TPrecision>& network()
        {
            return *network_;
        }

        /**
         * @brief Access the owned optimizer (const).
         *
         * @return Reference to the internal Optimizer instance
         */
        const Compute::Optimizer<TDeviceType, TPrecision>& optimizer() const
        {
            return *optimizer_;
        }

        /**
         * @brief Access the owned optimizer (mutable).
         *
         * @return Reference to the internal Optimizer instance
         */
        Compute::Optimizer<TDeviceType, TPrecision>& optimizer()
        {
            return *optimizer_;
        }

    private:

        /**
         * @brief Private constructor (use factory methods for construction).
         *
         * Constructs a Model from an already-built network and optimizer.
         * This constructor is private to enforce the factory pattern via
         * Model::create() which properly uses Network::createOptimizer().
         *
         * @param network Unique pointer to built network
         * @param optimizer Unique pointer to configured optimizer
         * @param config Model configuration for training
         */
        Model(
            std::unique_ptr<Network<TDeviceType, TPrecision>> network,
            std::unique_ptr<Compute::Optimizer<TDeviceType, TPrecision>> optimizer,
            const ModelConfig& config )
            : network_( std::move( network ) ), optimizer_( std::move( optimizer ) ), config_( config )
        {
            if ( !network_ )
            {
                throw std::invalid_argument( "Model: network cannot be null" );
            }
            
            if ( !optimizer_ )
            {
                throw std::invalid_argument( "Model: optimizer cannot be null" );
            }

            checkpoint_manager_ = std::make_unique<Modeling::CheckpointManager>( config_ );
        }

        // Member variables
        std::unique_ptr<Network<TDeviceType, TPrecision>> network_;
        std::unique_ptr<Compute::Optimizer<TDeviceType, TPrecision>> optimizer_;
        const ModelConfig config_;
        std::unique_ptr<Modeling::CheckpointManager> checkpoint_manager_;

        // ====================================================================
        // Internal Helper Methods
        // ====================================================================

        /**
         * @internal
         * @brief Validate that model metadata in an archive matches requested device/precision.
         *
         * @param meta Metadata to validate
         * @throws std::runtime_error if device or precision mismatch
         */
        template<Compute::DeviceType D, TensorDataType P>
        static void validateModelMetadata( const SerializationMetadata& meta )
        {
            std::string file_device = meta.tryGetString( "device" ).value_or( "" );
            std::string file_precision = meta.tryGetString( "precision" ).value_or( "" );

            if ( file_device != deviceTypeToString( D ) )
            {
                throw std::runtime_error(
                    std::format( "Device mismatch: file='{}', requested='{}'",
                        file_device, deviceTypeToString( D ) ) );
            }

            if ( file_precision != "FP32" /* precisionToString(P) */ )
            {
                throw std::runtime_error(
                    std::format( "Precision mismatch: file='{}', requested='{}'",
                        file_precision, "FP32" /* precisionToString(P) */ ) );
            }
        }

        /**
         * @internal
         * @brief Create and persist a training checkpoint using the CheckpointManager.
         *
         * @param metadata Checkpoint metadata used to name and register the checkpoint
         */
        void saveTrainingCheckpoint( const Modeling::CheckpointMetadata& metadata )
        {
            auto filename = checkpoint_manager_->generateCheckpointFilename( metadata.epoch );
            auto filepath = config_.getCheckpointDir() / filename;

            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath.string(), std::move( serializer ), OpenMode::Write );

            SerializationMetadata model_meta;
            model_meta.set( "model_version", int64_t( 1 ) )
                      .set( "device", deviceTypeToString( TDeviceType ) )
                      .set( "precision", "FP32" ) // FIXME: precisionToString( TPrecision )
                      .set( "framework_version", int64_t( 1 ) ) // MILA_VERSION
                      .set( "epoch", static_cast<int64_t>( metadata.epoch ) )
                      .set( "train_loss", metadata.train_loss )
                      .set( "val_loss", metadata.val_loss )
                      .set( "timestamp", static_cast<int64_t>(
                          std::chrono::system_clock::to_time_t( metadata.timestamp ) ) );
            
            archive.writeMetadata( "model/meta.json", model_meta );

            network_->save( archive, SerializationMode::Checkpoint );

            optimizer_->save( archive, "optimizer/" );

            // FIXME: Serialize config when ModelConfig has serialization support
            // SerializationMetadata cfg = config_.toMetadata();
            // archive.writeMetadata( "model/config.json", cfg );

            archive.close();

            Modeling::CheckpointMetadata saved_metadata = metadata;
            saved_metadata.filepath = filepath;
            checkpoint_manager_->addCheckpoint( saved_metadata );

            if ( config_.getVerbose() )
            {
                Utils::Logger::info_fmt(
                    "Checkpoint saved: {} (epoch {}, train_loss: {:.6f}, val_loss: {:.6f})",
                    filename, metadata.epoch, metadata.train_loss, metadata.val_loss );
            }
        }

        /**
         * @internal
         * @brief Load a checkpoint archive from a given path and restore state.
         *
         * @param filepath Filesystem path to the checkpoint archive
         */
        void loadCheckpointFromPath( const std::filesystem::path& filepath )
        {
            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath.string(), std::move( serializer ), OpenMode::Read );

            SerializationMetadata model_meta = archive.readMetadata( "model/meta.json" );
            validateModelMetadata<TDeviceType, TPrecision>( model_meta );

            network_->load( archive, SerializationMode::Checkpoint );

            optimizer_->load( archive, "optimizer/" );

            // FIXME: Deserialize config when ModelConfig has serialization support
            // auto cfg = archive.readMetadata( "model/config.json" );
            // config_ = ModelConfig::fromMetadata( cfg );
        }

        /**
         * @internal
         * @brief Single-epoch training implementation.
         *
         * TODO: Replace with dataset-driven training logic
         * 
         * @return Epoch training loss
         */
        double trainEpoch()
        {
            // TODO: Implement training epoch loop
            // - Iterate over batches
            // - Forward pass
            // - Compute loss
            // - Backward pass
            // - Optimizer step
            return 0.0;
        }

        /**
         * @internal
         * @brief Single-epoch validation implementation.
         *
         * TODO: Replace with dataset-driven validation logic
         * 
         * @return Validation loss
         */
        double validateEpoch()
        {
            // TODO: Implement validation epoch loop
            // - Iterate over batches
            // - Forward pass (no backward)
            // - Compute loss
            return 0.0;
        }
    };
}