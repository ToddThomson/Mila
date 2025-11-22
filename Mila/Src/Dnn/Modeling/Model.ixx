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
import Data.DatasetLoader;
import Dnn.Network;
import Dnn.NetworkFactory;

import Dnn.Optimizers.AdamW;
import Dnn.Optimizers.AdamWConfig;

import Modeling.ModelConfig;
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

    export template<Compute::DeviceType TDeviceType, Mila::Dnn::TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Model
    {
    public:

        Model(
            std::unique_ptr<Network<TDeviceType>> network,
            std::unique_ptr<Compute::Optimizer<TDeviceType, TPrecision>> optimizer )
            : network_( std::move( network ) ), optimizer_( std::move( optimizer ) )
        {
        }

        auto& configure( const ModelConfig& config )
        {
            config_ = config;
            checkpoint_manager_ = std::make_unique<Modeling::CheckpointManager>( config_ );

            return *this;
        }

        ModelConfig& config()
        {
            return config_;
        }

        TrainingHistory train()
        {
            if (!checkpoint_manager_)
            {
                checkpoint_manager_ = std::make_unique<Modeling::CheckpointManager>( config_ );
            }

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

        double evaluate( /* DatasetReader& test_data */ )
        {
            return 0.0;
        }

        void saveCheckpoint( const std::string& filepath ) const
        {
            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath, std::move( serializer ), OpenMode::Write );

            json model_meta;
            model_meta["model_version"] = 1;
            model_meta["device"] = deviceTypeToString( TDeviceType );
            model_meta["precision"] = precisionToString( TPrecision );
            model_meta["framework_version"] = 1; // MILA_VERSION;
            archive.writeJson( "model/meta.json", model_meta );

            network_->save( archive, SerializationMode::Checkpoint );

            optimizer_->save( archive, "optimizer/" );

            json cfg = config_.toJson();
            archive.writeJson( "model/config.json", cfg );

            archive.close();
        }

        static std::unique_ptr<Model> loadCheckpoint(
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

            auto adamw_config = Optimizers::AdamWConfig()
                .withLearningRate( config.learning_rate )
                .withBeta1( config.beta1 )
                .withBeta2( config.beta2 )
                .withEpsilon( config.epsilon )
                .withWeightDecay( config.weight_decay )
                .withName( "AdamW" );

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

        void exportModel( const std::string& filepath ) const
        {
            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath, std::move( serializer ), OpenMode::Write );

            json model_meta;
            model_meta["model_version"] = 1;
            model_meta["device"] = deviceTypeToString( TDeviceType );
            model_meta["precision"] = precisionToString( TPrecision );
            model_meta["export_mode"] = true;
            archive.writeJson( "model/meta.json", model_meta );

            network_->save( archive, SerializationMode::WeightsOnly );

            archive.close();
        }

        static std::unique_ptr<Network<TDeviceType>> loadModel(
            const std::string& filepath,
            std::shared_ptr<Compute::ExecutionContext<TDeviceType>> exec_context )
        {
            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath, std::move( serializer ), OpenMode::Read );

            json model_meta = archive.readJson( "model/meta.json" );

            if (!model_meta.value( "export_mode", false ))
            {
                throw std::runtime_error(
                    "File is not an exported model. Use loadCheckpoint() for training checkpoints." );
            }

            validateModelMetadata<TDeviceType, TPrecision>( model_meta );

            return NetworkFactory::create<TDeviceType>( archive, exec_context );
        }

        std::optional<Modeling::CheckpointMetadata> loadLatestCheckpoint()
        {
            if (!checkpoint_manager_)
            {
                checkpoint_manager_ = std::make_unique<Modeling::CheckpointManager>( config_ );
                checkpoint_manager_->scanCheckpoints();
            }

            auto latest = checkpoint_manager_->getLatestCheckpoint();
            if (!latest)
            {
                return std::nullopt;
            }

            loadCheckpointFromPath( latest->filepath );

            if (config_.getVerbose())
            {
                Utils::Logger::info_fmt( "Model state restored from epoch {}", latest->epoch );
            }

            return latest;
        }

        std::optional<Modeling::CheckpointMetadata> loadBestCheckpoint()
        {
            if (!checkpoint_manager_)
            {
                checkpoint_manager_ = std::make_unique<Modeling::CheckpointManager>( config_ );
                checkpoint_manager_->scanCheckpoints();
            }

            auto best = checkpoint_manager_->getBestCheckpoint();
            if (!best)
            {
                return std::nullopt;
            }

            loadCheckpointFromPath( best->filepath );

            if (config_.getVerbose())
            {
                Utils::Logger::info_fmt( "Best model restored from epoch {} (val_loss: {:.6f})",
                    best->epoch, best->val_loss );
            }

            return best;
        }

        TrainingHistory resumeTraining( std::size_t additional_epochs = 0 )
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
        }

        const Network<TDeviceType>& network() const
        {
            return *network_;
        }

        const Compute::Optimizer<TDeviceType, TPrecision>& optimizer() const
        {
            return *optimizer_;
        }

    private:
        std::unique_ptr<Network<TDeviceType>> network_;
        std::unique_ptr<Compute::Optimizer<TDeviceType, TPrecision>> optimizer_;
        ModelConfig config_;
        std::unique_ptr<Modeling::CheckpointManager> checkpoint_manager_;

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

            if (file_precision != precisionToString( P ))
            {
                throw std::runtime_error(
                    std::format( "Precision mismatch: file='{}', requested='{}'",
                        file_precision, precisionToString( P ) ) );
            }
        }

        void saveTrainingCheckpoint( const Modeling::CheckpointMetadata& metadata )
        {
            auto filename = checkpoint_manager_->generateCheckpointFilename( metadata.epoch );
            auto filepath = config_.getCheckpointDir() / filename;

            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath.string(), std::move( serializer ), OpenMode::Write );

            json model_meta;
            model_meta["model_version"] = 1;
            model_meta["device"] = deviceTypeToString( TDeviceType );
            model_meta["precision"] = precisionToString( TPrecision );
            model_meta["framework_version"] = 1;// MILA_VERSION;
            model_meta["epoch"] = metadata.epoch;
            model_meta["train_loss"] = metadata.train_loss;
            model_meta["val_loss"] = metadata.val_loss;
            model_meta["timestamp"] = std::chrono::system_clock::to_time_t( metadata.timestamp );
            archive.writeJson( "model/meta.json", model_meta );

            network_->save( archive, SerializationMode::Checkpoint );

            optimizer_->save( archive, "optimizer/" );

            json cfg = config_.toJson();
            archive.writeJson( "model/config.json", cfg );

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

        void loadCheckpointFromPath( const std::filesystem::path& filepath )
        {
            auto serializer = std::make_unique<ZipSerializer>();
            ModelArchive archive( filepath.string(), std::move( serializer ), OpenMode::Read );

            json model_meta = archive.readJson( "model/meta.json" );
            validateModelMetadata<TDeviceType, TPrecision>( model_meta );

            network_->load( archive, SerializationMode::Checkpoint );

            optimizer_->load( archive, "optimizer/" );

            json cfg = archive.readJson( "model/config.json" );
            config_.fromJson( cfg );
        }

        double trainEpoch()
        {
            return 0.0;
        }

        double validateEpoch()
        {
            return 0.0;
        }

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

            for (std::size_t epoch = start_epoch; epoch < config_.getEpochs(); ++epoch)
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
    };
}