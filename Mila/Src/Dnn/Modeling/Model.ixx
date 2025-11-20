module;
#include <memory>
#include <filesystem>
#include <chrono>
#include <utility>
#include <stdexcept>
#include <optional>

export module Dnn.Model;

import Dnn.TensorDataType;
import Dnn.TensorTypes;
import Dnn.CompositeModule;
import Compute.DeviceType;
import Compute.OptimizerBase;
import Data.DataLoader;
import Modeling.ModelConfig;
//import Model.LossFunction;
import Modeling.CheckpointManager;
import Modeling.CheckpointMetaData;
import Modeling.TrainingHistory;
import Serialization.ModelSerializer;
import Utils.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Serialization;

    export template<Compute::DeviceType TDeviceType, Mila::Dnn::TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Model
    {
    public:

        Model(
            std::unique_ptr<Network<TDeviceType>> network,
            std::unique_ptr<Mila::Dnn::Compute::Optimizer<TDeviceType, TPrecision>> optimizer
            //std::unique_ptr<LossFunction> loss,
            /* std::unique_ptr<Mila::Dnn::Data::DataLoader> dataset */ )
            : network_( std::move( network ) )
            , optimizer_( std::move( optimizer ) )
            //, loss_( std::move( loss ) )
            //, dataset_( std::move( dataset ) )
            , config_()
            , checkpoint_manager_( nullptr )
        {
        }

        // Configure the model
        auto& configure( const ModelConfig& config )
        {
            config_ = config;
            checkpoint_manager_ = std::make_unique<CheckpointManager>( config_ );

            return *this;
        }

        // Get mutable config for fluent API
        ModelConfig& config()
        {
            return config_;
        }

        // Train the model
        TrainingHistory train()
        {
            if (!checkpoint_manager_)
            {
                checkpoint_manager_ = std::make_unique<CheckpointManager>( config_ );
            }

            TrainingHistory history;

            if (config_.getVerbose())
            {
                Utils::Logger::info_fmt( "Starting training for {} epochs...", config_.getEpochs() );
            }

            for (std::size_t epoch = 0; epoch < config_.getEpochs(); ++epoch)
            {
                history.current_epoch = epoch;

                // Training phase
                double train_loss = trainEpoch();
                history.train_losses.push_back( train_loss );

                // Validation phase (if validation split specified)
                double val_loss = 0.0;
                if (config_.getValidationSplit() > 0.0)
                {
                    val_loss = validateEpoch();
                    history.val_losses.push_back( val_loss );

                    // Check for improvement
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

                // Logging
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

                // Checkpoint saving
                if ((epoch + 1) % config_.getCheckpointFrequency() == 0)
                {
                    CheckpointMetadata metadata{
                        .epoch = epoch,
                        .train_loss = train_loss,
                        .val_loss = val_loss,
                        .timestamp = std::chrono::system_clock::now(),
                        .filepath = {}
                    };

                    //checkpoint_manager_->saveCheckpoint<TDeviceType>( *network_, metadata );

                    /* FIXME: checkpoint_manager_->saveCheckpoint(
                        [this]( Serialization::ModelSerializer& ser ) {
                            network_->serialize( ser, "network/" );
                            optimizer_->serialize( ser, "optimizer/" );
                        },
                        metadata
                    );*/
                }

                // Early stopping
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

        // Evaluate on test data
        double evaluate( /* DatasetReader& test_data */ )
        {
            // TODO: Implement evaluation logic

            return 0.0;
        }
        
        // Load from checkpoint with architecture reconstruction
        static Model fromCheckpoint(
            const std::filesystem::path& checkpoint_path,
            std::unique_ptr<Optimizer> optimizer,
            std::unique_ptr<LossFunction> loss,
            std::unique_ptr<DatasetReader> dataset )
        {
            ZipSerializer serializer;
            if (!serializer.openForRead( checkpoint_path.string() ))
            {
                throw std::runtime_error( "Failed to open checkpoint" );
            }

            // Check if architecture is stored in checkpoint
            if (!serializer.hasFile( "architecture.json" ))
            {
                throw std::runtime_error(
                    "Checkpoint does not contain architecture. "
                    "Please reconstruct the network manually."
                );
            }

            // Load and reconstruct architecture
            auto arch_size = serializer.getFileSize( "architecture.json" );
            std::string arch_json( arch_size, '\0' );
            serializer.extractData( "architecture.json", arch_json.data(), arch_size );

            auto network = Network::from_json( arch_json );

            serializer.close();

            // Create model with reconstructed network
            Model model(
                std::move( network ),
                std::move( optimizer ),
                std::move( loss ),
                std::move( dataset )
            );

            // Load the checkpoint state
            model.loadCheckpoint( checkpoint_path );

            return model;
        }
        /**
         * @brief Loads model state from a checkpoint file
         * @param checkpoint_path Path to the checkpoint file
         * @return Metadata of the loaded checkpoint
         */
        CheckpointMetadata loadCheckpoint( const std::filesystem::path& checkpoint_path )
        {
            if (!checkpoint_manager_)
            {
                checkpoint_manager_ = std::make_unique<CheckpointManager>( config_ );
            }

            auto metadata = checkpoint_manager_->load_checkpoint(
                checkpoint_path,
                [this]( ModelSerializer& serializer ) {
                    // Deserialize network weights
                    network_->deserialize( serializer, "network/" );

                    // Deserialize optimizer state
                    optimizer_->deserialize( serializer, "optimizer/" );

                    // Deserialize any other state you've saved
                }
            );

            if (config_.getVerbose())
            {
                Utils::Logger::info_fmt( "Model state restored from epoch {}", metadata.epoch );
            }

            return metadata;
        }

        /**
         * @brief Loads the latest checkpoint
         * @return Metadata of the loaded checkpoint, or nullopt if no checkpoints exist
         */
        std::optional<CheckpointMetadata> loadLatestCheckpoint()
        {
            if (!checkpoint_manager_)
            {
                checkpoint_manager_ = std::make_unique<CheckpointManager>( config_ );
                checkpoint_manager_->scan_checkpoints();
            }

            return checkpoint_manager_->load_latest_checkpoint(
                [this]( Serialization::ModelSerializer& serializer ) {
                    network_->deserialize( serializer, "network/" );
                    optimizer_->deserialize( serializer, "optimizer/" );
                }
            );
        }

        TrainingHistory resume_training( std::size_t additional_epochs = 0 )
        {
            if (!checkpoint_manager_)
            {
                checkpoint_manager_ = std::make_unique<CheckpointManager>( config_ );
                //checkpoint_manager_->scan_checkpoints();
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

            // Adjust epochs if additional_epochs specified
            std::size_t original_epochs = config_.getEpochs();
            if (additional_epochs > 0)
            {
                config_.set_epochs( latest->epoch + additional_epochs );
            }

            // Continue training
            TrainingHistory history = trainFromEpoch( latest->epoch + 1 );

            // Restore original config if we modified it
            if (additional_epochs > 0)
            {
                config_.set_epochs( original_epochs );
            }

            return history;
        }

        // Accessors
        const CompositeModule<TDeviceType>& network() const
        {
            return *network_;
        }

        const Compute::Optimizer<TDeviceType, TPrecision>& optimizer() const
        {
            return *optimizer_;
        }

        const CheckpointManager* checkpointManager() const
        {
            return checkpoint_manager_.get();
        }

    private:
        std::unique_ptr<Network<TDeviceType>> network_;
        std::unique_ptr<Compute::Optimizer<TDeviceType, TPrecision>> optimizer_;
        //std::unique_ptr<LossFunction> loss_;
        //std::unique_ptr<DataLoader> dataset_;
        ModelConfig config_;
        std::unique_ptr<CheckpointManager> checkpoint_manager_;

        double trainEpoch()
        {
            // TODO: Implement training loop for one epoch
            // - Iterate through batches
            // - Forward pass
            // - Compute loss
            // - Backward pass
            // - Update weights with optimizer

            return 0.0;
        }

        double validateEpoch()
        {
            // TODO: Implement validation loop
            // - Iterate through validation batches
            // - Forward pass only (no gradient computation)
            // - Compute loss

            return 0.0;
        }

        /**
         * @brief Internal fit method that starts from a specific epoch
         * @param start_epoch Epoch to start from
         * @return Training history
         */
        TrainingHistory trainFromEpoch( std::size_t start_epoch )
        {
            if (!checkpoint_manager_)
            {
                checkpoint_manager_ = std::make_unique<CheckpointManager>( config_ );
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

                // Training phase
                double train_loss = trainEpoch();
                history.train_losses.push_back( train_loss );

                // Validation phase
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

                // Logging
                if (config_.getVerbose())
                {
                    Utils::Logger::info_fmt( "Epoch {}/{}: loss = {:.6f}",
                        epoch + 1, config_.getEpochs(), train_loss );

                    if (config_.getValidationSplit() > 0.0)
                    {
                        Utils::Logger::info_fmt( ", val_loss = {:.6f}", val_loss );
                    }
                }

                // Checkpoint saving
                if ((epoch + 1) % config_.getCheckpointFrequency() == 0)
                {
                    CheckpointMetadata metadata{
                        .epoch = epoch,
                        .train_loss = train_loss,
                        .val_loss = val_loss,
                        .timestamp = std::chrono::system_clock::now(),
                        .filepath = {}
                    };

                    checkpoint_manager_->saveCheckpoint(
                        [this]( ModelSerializer& serializer ) {
                            network_->serialize( serializer, "network/" );
                            optimizer_->serialize( serializer, "optimizer/" );
                        },
                        metadata
                    );
                }

                // Early stopping
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