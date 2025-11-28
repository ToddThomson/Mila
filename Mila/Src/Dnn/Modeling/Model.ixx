/**
 * @file Model.ixx
 * @brief Training and persistence orchestration for device-templated Models.
 *
 * Provides the Model class which owns a network and optimizer, coordinates
 * training/evaluation loops, and exposes import/export and checkpointing helpers.
 */

module;
#include <memory>
#include <filesystem>
#include <chrono>
#include <utility>
#include <stdexcept>
#include <optional>
#include <string>

export module Dnn.Model;

import Dnn.TensorDataType;
import Dnn.TensorTypes;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Compute.OptimizerBase;
import Compute.ExecutionContext;
import Data.DatasetReader;
import Dnn.Network;
import Dnn.NetworkFactory;

import Dnn.Loss;

import Dnn.Optimizers.AdamW;
import Dnn.Optimizers.AdamWConfig;

import Dnn.ModelConfig;
import Modeling.CheckpointManager;
import Modeling.CheckpointMetaData;
import Modeling.TrainingHistory;
import Serialization.ModelArchive;
import Serialization.ZipSerializer;
import Serialization.OpenMode;
import Serialization.Mode;
import Utils.Logger;
import nlohmann.json;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Serialization;
    using json = nlohmann::json;

    /**
     * @brief High-level model wrapper that owns network and optimizer.
     *
     * The Model class coordinates training/evaluation, checkpointing and
     * model export. It is parameterized by compile-time device and precision
     * so all contained modules and optimizers are device/precision-consistent.
     *
     * Ownership:
     *  - The Model takes ownership of a `Network` and an `Optimizer` via
     *    unique_ptr at construction.
     *
     * Threading / Safety:
     *  - Not thread-safe. Callers must serialize access to a Model instance.
     *
     * Template parameters:
     *  - TDeviceType: compile-time compute device (CPU, CUDA, ...).
     *  - TPrecision: compile-time tensor precision enum.
     */
    export template<Compute::DeviceType TDeviceType, Mila::Dnn::TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Model
    {
    public:

        /**
         * @brief Construct a Model from an already-constructed network and optimizer.
         *
         * Preconditions:
         *  - `network` must be non-null and already configured for the target device.
         *  - `optimizer` must be non-null and compatible with the network parameter layout.
         *
         * Ownership:
         *  - Model takes ownership of both `network` and `optimizer`.
         *
         * @param network Unique pointer to the network implementation.
         * @param optimizer Unique pointer to the optimizer instance.
         */
        Model(
            std::unique_ptr<Network<TDeviceType, TPrecision>> network,
            std::unique_ptr<Compute::Optimizer<TDeviceType, TPrecision>> optimizer,
			//std::unique_ptr<Loss<TDeviceType, TPrecision>> loss_fn,
            const ModelConfig& config )
			: network_( std::move( network ) ), optimizer_( std::move( optimizer ) ), config_( config )
        {
            if (!network_)
            {
                throw std::invalid_argument( "Model: network cannot be null" );
            }
            
            if (!optimizer_)
            {
                throw std::invalid_argument( "Model: optimizer cannot be null" );
            }

            checkpoint_manager_ = std::make_unique<Modeling::CheckpointManager>( config_ );
        }

        /**
         * @brief Run the configured training loop.
         *
         * Implements a simple epoch loop driven by `config_`. This method:
         *  - Recreates the checkpoint manager if needed.
         *  - Calls `trainEpoch()` for each epoch.
         *  - Optionally runs validation and saves checkpoints at configured intervals.
         *  - Honors early stopping settings in `config_`.
         *
         * Postconditions:
         *  - Checkpoints may be written to disk via `saveTrainingCheckpoint`.
         *
         * @return TrainingHistory that accumulates per-epoch statistics.
         */
		template<TensorDataType TInputType, TensorDataType TTargetType, typename TMemoryResource>
        TrainingHistory train(
            Data::DatasetReader<TInputType, TTargetType, TMemoryResource> train_reader,
			std::optional<Data::DatasetReader<TInputType, TTargetType, TMemoryResource>> val_reader = std::nullopt )
        {
            TrainingHistory history;

            if (config_.getVerbose())
            {
                Utils::Logger::info_fmt( "Starting training for {} epochs...", config_.getEpochs() );
            }

            for (std::size_t epoch = 0; epoch < config_.getEpochs(); ++epoch)
            {
                history.current_epoch = epoch;

                double train_loss = trainEpoch();
                history.train_losses.push_back( train_loss );

                double val_loss = 0.0;
                if (config_.getValidationSplit() > 0.0)
                {
                    val_loss = validateEpoch();
                    history.val_losses.push_back( val_loss );

                    if (val_loss < history.best_val_loss)
                    {
                        history.best_val_loss = val_loss;
                        history.epochs_without_improvement = 0;
                    }
                    else
                    {
                        history.epochs_without_improvement++;
                    }
                }

                if (config_.getVerbose())
                {
                    if (config_.getValidationSplit() > 0.0)
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

                if ((epoch + 1) % config_.getCheckpointFrequency() == 0)
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

                if (config_.getEarlyStoppingEnabled() &&
                    history.epochs_without_improvement >= config_.getEarlyStoppingPatience())
                {
                    if (config_.getVerbose())
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
         * @brief Resume training from the latest checkpoint and continue for additional epochs.
         *
         * This method:
         *  - Loads the latest checkpoint,
         *  - Adjusts the configured epoch count if additional_epochs > 0,
         *  - Continues training from the epoch after the checkpoint.
         *
         * @param additional_epochs Optional number of epochs to append to the resumed schedule.
         * @return TrainingHistory aggregated from resumed training.
         *
         * @throws std::runtime_error if no checkpoint exists to resume from.
         */
        /*TrainingHistory resumeTraining( std::size_t additional_epochs = 0 )
        {
            if (!checkpoint_manager_)
            {
                checkpoint_manager_ = std::make_unique<Modeling::CheckpointManager>( config_ );
                checkpoint_manager_->scanCheckpoints();
            }

            auto latest = loadLatestCheckpoint();
            if (!latest)
            {
                throw std::runtime_error( "No checkpoint found to resume from" );
            }

            if (config_.getVerbose())
            {
                Utils::Logger::info_fmt( "Resuming training from epoch {}", latest->epoch + 1 );
            }

            std::size_t original_epochs = config_.getEpochs();
            if (additional_epochs > 0)
            {
                config_.setEpochs( latest->epoch + additional_epochs );
            }

            TrainingHistory history = trainFromEpoch( latest->epoch + 1 );

            if (additional_epochs > 0)
            {
                config_.setEpochs( original_epochs );
            }

            return history;
        }*/

        /**
         * @brief Evaluate the model on a dataset.
         *
         * Placeholder: current implementation returns 0.0. Replace with dataset-driven
         * evaluation logic when integrating DatasetReader/Loader.
         *
         * @return Computed evaluation metric (loss) as double.
         */
        double evaluate( /* DatasetReader& test_data */ )
        {
            return 0.0;
        }

        /**
         * @brief Save a full training checkpoint to the provided filepath.
         *
         * Writes a ZIP archive with:
         *  - model/meta.json
         *  - network/* (delegated to Network::save)
         *  - optimizer/* (delegated to optimizer->save)
         *  - model/config.json
         *
         * Preconditions:
         *  - The archive path must be writable.
         *
         * @param filepath Filesystem path where checkpoint is written.
         */
        void saveCheckpoint( const std::string& filepath ) const
        {
            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath, std::move( serializer ), OpenMode::Write );

            json model_meta;
            model_meta["model_version"] = 1;
            model_meta["device"] = deviceTypeToString( TDeviceType );
            model_meta["precision"] = "FP32"; // FIXME: precisionToString( TPrecision );
            model_meta["framework_version"] = 1; // MILA_VERSION;
            archive.writeJson( "model/meta.json", model_meta );

            network_->save( archive, SerializationMode::Checkpoint );

            optimizer_->save( archive, "optimizer/" );

			// FIXME:
            //json cfg = config_.toJson();
            //archive.writeJson( "model/config.json", cfg );

            archive.close();
        }

        /**
         * @brief Load a training checkpoint and reconstruct a Model instance.
         *
         * This factory method reads checkpoint metadata and uses NetworkFactory
         * to reconstruct the network. Optimizer reconstruction is currently
         * simplified and a device/precision-compatible optimizer is created.
         *
         * Preconditions:
         *  - `exec_context` must be valid for the requested device.
         *
         * @param filepath Filesystem path to the checkpoint archive.
         * @param exec_context Execution context to attach to reconstructed network.
         * @return Unique pointer to a reconstructed Model instance.
         */
        static std::unique_ptr<Model> fromCheckpoint(
            const std::string& filepath,
            std::shared_ptr<Compute::ExecutionContext<TDeviceType>> exec_context )
        {
            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath, std::move( serializer ), OpenMode::Read );

            json model_meta = archive.readJson( "model/meta.json" );
            validateModelMetadata<TDeviceType, TPrecision>( model_meta );

            auto network = NetworkFactory::create<TDeviceType>( archive, exec_context );

            //auto optimizer = OptimizerFactory::create<TDeviceType, TPrecision>(
            //    archive, "optimizer/", network->getParameters() );

            auto adamw_config = Optimizers::AdamWConfig();
                /*.withLearningRate( config.learning_rate )
                .withBeta1( config.beta1 )
                .withBeta2( config.beta2 )
                .withEpsilon( config.epsilon )
                .withWeightDecay( config.weight_decay )
                .withName( "AdamW" );*/

            auto optimizer = std::make_shared<Optimizers::AdamWOptimizer<TDeviceType, dtype_t::FP32>>(
                exec_context, adamw_config );

            json cfg = archive.readJson( "model/config.json" );
            ModelConfig config;
            //config.fromJson( cfg );

            auto model = std::make_unique<Model>(
                std::move( network ),
                std::move( optimizer )
            );
            //model->config_ = std::move( config );

            return model;
        }

        /**
         * @brief Export a model artifact intended for inference.
         *
         * Produces a compact archive containing model metadata and weights only.
         * The exported archive is validated by `loadModel()` via the `export_mode`
         * metadata flag.
         *
         * @param filepath Path to write exported model file.
         */
        void save( const std::string& filepath ) const
        {
            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath, std::move( serializer ), OpenMode::Write );

            json model_meta;
            model_meta["model_version"] = 1;
            model_meta["device"] = deviceTypeToString( TDeviceType );
            model_meta["precision"] = "FP32"; // FIXME: precisionToString( TPrecision );
            model_meta["export_mode"] = true;
            archive.writeJson( "model/meta.json", model_meta );

            network_->save( archive, SerializationMode::WeightsOnly );

            archive.close();
        }

        ///**
        // * @brief Load a model exported for inference.
        // *
        // * Validates that the archive was exported (not a training checkpoint)
        // * and then reconstructs the network via NetworkFactory.
        // *
        // * @param filepath Path to exported model archive.
        // * @param exec_context Execution context for device resources.
        // * @return Unique pointer to reconstructed Network instance.
        // *
        // * @throws std::runtime_error if archive is not an exported model.
        // */
        //static std::unique_ptr<Network<TDeviceType>> loadModel(
        //    const std::string& filepath,
        //    std::shared_ptr<Compute::ExecutionContext<TDeviceType>> exec_context )
        //{
        //    auto serializer = std::make_unique<ZipSerializer>();
        //    ModelArchive archive( filepath, std::move( serializer ), OpenMode::Read );

        //    json model_meta = archive.readJson( "model/meta.json" );

        //    if (!model_meta.value( "export_mode", false ))
        //    {
        //        throw std::runtime_error(
        //            "File is not an exported model. Use loadCheckpoint() for training checkpoints." );
        //    }

        //    validateModelMetadata<TDeviceType, TPrecision>( model_meta );

        //    return NetworkFactory::create<TDeviceType>( archive, exec_context );
        //}


        /**
         * @brief Access the owned network (const).
         *
         * @return Reference to the internal Network instance.
         */
        const Network<TDeviceType,TPrecision>& network() const
        {
            return *network_;
        }

        /**
         * @brief Access the owned optimizer (const).
         *
         * @return Reference to the internal Optimizer instance.
         */
        const Compute::Optimizer<TDeviceType, TPrecision>& optimizer() const
        {
            return *optimizer_;
        }

    private:
        std::unique_ptr<Network<TDeviceType, TPrecision>> network_;
        std::unique_ptr<Compute::Optimizer<TDeviceType, TPrecision>> optimizer_;
        const ModelConfig config_;
        std::unique_ptr<Modeling::CheckpointManager> checkpoint_manager_;

        /**
         * @internal
         * @brief Validate that model metadata in an archive matches requested device/precision.
         *
         * Throws on mismatch.
         */
        template<Compute::DeviceType D, TensorDataType P>
        static void validateModelMetadata( const json& meta )
        {
            std::string file_device = meta.at( "device" ).get<std::string>();
            std::string file_precision = meta.at( "precision" ).get<std::string>();

            if (file_device != deviceTypeToString( D ))
            {
                throw std::runtime_error(
                    std::format( "Device mismatch: file='{}', requested='{}'",
                        file_device, deviceTypeToString( D ) ) );
            }

            if (file_precision != "FP32" /* precisionToString(P) */ )
            {
                throw std::runtime_error(
                    std::format( "Precision mismatch: file='{}', requested='{}'",
                        file_precision, "FP32"/*precisionToString(P) */));
            }
        }

        /**
         * @internal
         * @brief Create and persist a training checkpoint using the CheckpointManager.
         *
         * Writes model/meta.json, network data, optimizer state and model/config.json.
         *
         * @param metadata Checkpoint metadata used to name and register the checkpoint.
         */
        void saveTrainingCheckpoint( const Modeling::CheckpointMetadata& metadata )
        {
            auto filename = checkpoint_manager_->generateCheckpointFilename( metadata.epoch );
            auto filepath = config_.getCheckpointDir() / filename;

            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath.string(), std::move( serializer ), OpenMode::Write );

            json model_meta;
            model_meta["model_version"] = 1;
            model_meta["device"] = deviceTypeToString( TDeviceType );
            model_meta["precision"] = "FP32"; // FIXME: precisionToString( TPrecision );
            model_meta["framework_version"] = 1;// MILA_VERSION;
            model_meta["epoch"] = metadata.epoch;
            model_meta["train_loss"] = metadata.train_loss;
            model_meta["val_loss"] = metadata.val_loss;
            model_meta["timestamp"] = std::chrono::system_clock::to_time_t( metadata.timestamp );
            archive.writeJson( "model/meta.json", model_meta );

            network_->save( archive, SerializationMode::Checkpoint );

            optimizer_->save( archive, "optimizer/" );

            // FIXME:
            //json cfg = config_.toJson();
            //archive.writeJson( "model/config.json", cfg );

            archive.close();

            Modeling::CheckpointMetadata saved_metadata = metadata;
            saved_metadata.filepath = filepath;
            checkpoint_manager_->addCheckpoint( saved_metadata );

            if (config_.getVerbose())
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
         * Reads model/meta.json and validates, then delegates to network->load
         * and optimizer->load before applying configuration from the archive.
         *
         * @param filepath Filesystem path to the checkpoint archive.
         */
        void loadCheckpointFromPath( const std::filesystem::path& filepath )
        {
            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath.string(), std::move( serializer ), OpenMode::Read );

            json model_meta = archive.readJson( "model/meta.json" );
            validateModelMetadata<TDeviceType, TPrecision>( model_meta );

            network_->load( archive, SerializationMode::Checkpoint );

            optimizer_->load( archive, "optimizer/" );

            json cfg = archive.readJson( "model/config.json" );
            // FIXME: config_.fromJson( cfg );
        }

        /**
         * @internal
         * @brief Single-epoch training implementation.
         *
         * Replace with dataset-driven training logic. Returns epoch training loss.
         */
        double trainEpoch()
        {
            return 0.0;
        }

        /**
         * @internal
         * @brief Single-epoch validation implementation.
         *
         * Replace with dataset-driven validation logic. Returns validation loss.
         */
        double validateEpoch()
        {
            return 0.0;
        }

        /**
         * @internal
         * @brief Continue training from a specific starting epoch.
         *
         * Called by resumeTraining after determining the resume point.
         *
         * @param start_epoch Epoch index to begin training from (inclusive).
         * @return Aggregated TrainingHistory for resumed runs.
         */
        TrainingHistory trainFromEpoch( std::size_t start_epoch )
        {
            if (!checkpoint_manager_)
            {
                checkpoint_manager_ = std::make_unique<Modeling::CheckpointManager>( config_ );
            }

            TrainingHistory history;

            if (config_.getVerbose())
            {
                Utils::Logger::info_fmt( "Continuing training from epoch {} to {}...",
                    start_epoch, config_.getEpochs() );
            }

            for ( std::size_t epoch = start_epoch; epoch < config_.getEpochs(); ++epoch )
            {
                history.current_epoch = epoch;

                double train_loss = trainEpoch();
                history.train_losses.push_back( train_loss );

                double val_loss = 0.0;
                if ( config_.getValidationSplit() > 0.0 )
                {
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
                    if ( config_.getValidationSplit() > 0.0 )
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
    };
}