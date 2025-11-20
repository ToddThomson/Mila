/**
 * @file CheckpointManager.ixx
 * @brief Checkpoint manager for saving and querying model checkpoints.
 *
 * Provides a rolling-window of persisted checkpoints and utilities to select
 * the latest or best checkpoint by validation loss.
 */

module;
#include <vector>
#include <string>
#include <filesystem>
#include <optional>
#include <chrono>
#include <format>
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <system_error>
#include <exception>

export module Modeling.CheckpointManager;

import Dnn.CompositeModule;
import Compute.DeviceType;
import Modeling.ModelConfig;
import Modeling.CheckpointMetaData;
import Serialization.ZipSerializer;
import Utils.Logger;

namespace Mila::Dnn::Modeling
{
    /**
     * @brief Responsible for persisting model checkpoints and maintaining history.
     *
     * This class writes checkpoint files to the directory specified by the provided
     * ModelConfig and keeps an in-memory history limited by `ModelConfig::max_checkpoints()`.
     *
     * Threading: not thread-safe; callers must synchronize access if used concurrently.
     */
    export class CheckpointManager
    {
    public:

        using SerializationCallback = std::function<void( Serialization::ZipSerializer& )>;
        using DeserializationCallback = std::function<void( Serialization::ZipSerializer& )>;

        /**
         * @brief Construct a CheckpointManager.
         *
         * Creates the checkpoint directory if it does not exist.
         *
         * @param config Reference to model configuration that provides checkpoint_dir(),
         *               max_checkpoints(), and verbose() settings. The caller retains ownership.
         */
        explicit CheckpointManager( const ModelConfig& config )
            : config_( config )
        {
            std::filesystem::create_directories( config.getCheckpointDir() );
        }

        /**
         * @brief Persist a model checkpoint and update the rolling history.
         *
         * Side-effects:
         *  - Serializes the provided model to disk at a generated filename.
         *  - Appends checkpoint metadata to the in-memory history.
         *  - Removes the oldest checkpoint file when history exceeds the configured maximum.
         *
         * Errors: file-system operations may throw std::filesystem::filesystem_error.
         *
         * @param model The model to serialize. Ownership is not transferred.
         * @param metadata Metadata describing this checkpoint (epoch, losses, timestamp).
         */
        void saveCheckpoint( SerializationCallback serialize_fn,
            const CheckpointMetadata& metadata )
        {
            auto filename = generateCheckpointFilename( metadata.epoch );
            auto filepath = config_.getCheckpointDir() / filename;

            // Create serializer
            Serialization::ZipSerializer serializer;
            if (!serializer.openForWrite( filepath.string() ))
            {
                throw std::runtime_error(
                    std::format( "Failed to open checkpoint for writing: {}", filepath.string() )
                );
            }

            try
            {
                // Add metadata to the checkpoint
                serializer.addMetadata( "version", "1.0.0" );
                serializer.addMetadata( "epoch", std::to_string( metadata.epoch ) );
                serializer.addMetadata( "train_loss", std::to_string( metadata.train_loss ) );
                serializer.addMetadata( "val_loss", std::to_string( metadata.val_loss ) );

                // Format timestamp
                auto time_t = std::chrono::system_clock::to_time_t( metadata.timestamp );
                serializer.addMetadata( "timestamp", std::format( "{}", time_t ) );

                // Call the user-provided serialization function
                // This allows Model to serialize itself without CheckpointManager knowing about Model
                serialize_fn( serializer );

                serializer.close();
            }
            catch (const std::exception& e)
            {
                serializer.close();
                throw std::runtime_error(
                    std::format( "Failed to save checkpoint: {}", e.what() )
                );
            }

            // Update checkpoint metadata with actual filepath
            CheckpointMetadata saved_metadata = metadata;
            saved_metadata.filepath = filepath;

            // Add to checkpoint history
            checkpoints_.push_back( saved_metadata );

            // Maintain rolling window of checkpoints
            if (checkpoints_.size() > config_.getMaxCheckpoints())
            {
                auto oldest = checkpoints_.front();

                // Delete the oldest checkpoint file
                std::error_code ec;
                std::filesystem::remove( oldest.filepath, ec );
                
                if (ec && config_.getVerbose())
                {
                    Utils::Logger::warning_fmt( 
                        "Warning: Failed to delete old checkpoint: {}", 
                        oldest.filepath.string() );
                }

                checkpoints_.erase( checkpoints_.begin() );
            }

            if (config_.getVerbose())
            {
                Utils::Logger::info_fmt(
                    "Checkpoint saved: {} (epoch {}, train_loss: {:.6f}, val_loss: {:.6f})",
                    filename, metadata.epoch, metadata.train_loss, metadata.val_loss );
            }
        }

        /**
         * @brief Loads a checkpoint from a file
         * @param filepath Path to the checkpoint file
         * @param deserialize_fn Callback function to deserialize the model state
         * @return Metadata of the loaded checkpoint
         * @throws std::runtime_error if loading fails
         */
        CheckpointMetadata loadCheckpoint( const std::filesystem::path& filepath,
            DeserializationCallback deserialize_fn )
        {
            if (!std::filesystem::exists( filepath ))
            {
                throw std::runtime_error(
                    std::format( "Checkpoint file does not exist: {}", filepath.string() )
                );
            }

            // Create serializer for reading
            Serialization::ZipSerializer serializer;
            if (!serializer.openForRead( filepath.string() ))
            {
                throw std::runtime_error(
                    std::format( "Failed to open checkpoint for reading: {}", filepath.string() )
                );
            }

            try
            {
                // Read and validate metadata
                auto version = serializer.getMetadata( "version" );
                if (version.empty())
                {
                    throw std::runtime_error( "Invalid checkpoint: missing version metadata" );
                }

                // Parse metadata
                CheckpointMetadata metadata;
                metadata.filepath = filepath;

                auto epoch_str = serializer.getMetadata( "epoch" );
                if (!epoch_str.empty())
                {
                    metadata.epoch = std::stoull( epoch_str );
                }

                auto train_loss_str = serializer.getMetadata( "train_loss" );
                if (!train_loss_str.empty())
                {
                    metadata.train_loss = std::stod( train_loss_str );
                }

                auto val_loss_str = serializer.getMetadata( "val_loss" );
                if (!val_loss_str.empty())
                {
                    metadata.val_loss = std::stod( val_loss_str );
                }

                auto timestamp_str = serializer.getMetadata( "timestamp" );
                if (!timestamp_str.empty())
                {
                    auto time_t_val = std::stoll( timestamp_str );
                    metadata.timestamp = std::chrono::system_clock::from_time_t( time_t_val );
                }

                // Call user-provided deserialization function
                deserialize_fn( serializer );

                serializer.close();

                if (config_.getVerbose())
                {
                    Utils::Logger::info_fmt( "Checkpoint loaded: {} (epoch {}, train_loss: {:.6f}, val_loss: {:.6f})",
                        filepath.filename().string(),
                        metadata.epoch,
                        metadata.train_loss,
                        metadata.val_loss );
                }

                return metadata;
            }
            catch (const std::exception& e)
            {
                serializer.close();
                throw std::runtime_error( std::format( "Failed to load checkpoint: {}", e.what() ) );
            }
        }

        /**
         * @brief Loads the latest checkpoint
         * @param deserialize_fn Callback function to deserialize the model state
         * @return Metadata of the loaded checkpoint, or nullopt if no checkpoints exist
         */
        std::optional<CheckpointMetadata> loadLatestCheckpoint( DeserializationCallback deserialize_fn )
        {
            auto latest = getLatestCheckpoint();
            if (!latest)
            {
                return std::nullopt;
            }

            return loadCheckpoint( latest->filepath, deserialize_fn );
        }
        
        std::optional<CheckpointMetadata> loadBestCheckpoint( DeserializationCallback deserialize_fn )
        {
            auto best = getBestCheckpoint();
            if (!best)
            {
                return std::nullopt;
            }

            return loadCheckpoint( best->filepath, deserialize_fn );
        }


        /**
         * @brief Return the checkpoint with the lowest validation loss.
         *
         * Selection is based on `val_loss` field of CheckpointMetadata. If multiple checkpoints
         * tie, the first encountered minimum is returned.
         *
         * @return Optional containing the best CheckpointMetadata, or std::nullopt if none exist.
         */
        std::optional<CheckpointMetadata> getBestCheckpoint() const
        {
            if (checkpoints_.empty())
            {
                return std::nullopt;
            }

            return *std::ranges::min_element( checkpoints_,
                []( const auto& a, const auto& b ) {
                    return a.val_loss < b.val_loss;
                } );
        }

        /**
         * @brief Access the full in-memory list of checkpoint metadata.
         *
         * The returned reference is valid as long as this CheckpointManager instance is alive
         * and no mutating operations are performed on it.
         *
         * @return const reference to vector of CheckpointMetadata.
         */
        const std::vector<CheckpointMetadata>& getAllCheckpoints() const
        {
            return checkpoints_;
        }


        // ========== QUERY ==========

        std::optional<CheckpointMetadata> getLatestCheckpoint() const
        {
            if (checkpoints_.empty())
            {
                return std::nullopt;
            }
            return checkpoints_.back();
        }

        // ========== UTILITY ==========

        void scanCheckpoints()
        {
            checkpoints_.clear();

            if (!std::filesystem::exists( config_.getCheckpointDir() ))
            {
                return;
            }

            for (const auto& entry : std::filesystem::directory_iterator( config_.getCheckpointDir() ))
            {
                if (entry.path().extension() == ".mila")
                {
                    try
                    {
                        Serialization::ZipSerializer serializer;
                        if (!serializer.openForRead( entry.path().string() ))
                        {
                            continue;
                        }

                        CheckpointMetadata metadata;
                        metadata.filepath = entry.path();

                        auto epoch_str = serializer.getMetadata( "epoch" );
                        auto train_loss_str = serializer.getMetadata( "train_loss" );
                        auto val_loss_str = serializer.getMetadata( "val_loss" );
                        auto timestamp_str = serializer.getMetadata( "timestamp" );

                        if (!epoch_str.empty()) metadata.epoch = std::stoull( epoch_str );
                        if (!train_loss_str.empty()) metadata.train_loss = std::stod( train_loss_str );
                        if (!val_loss_str.empty()) metadata.val_loss = std::stod( val_loss_str );
                        if (!timestamp_str.empty())
                        {
                            auto time_t_val = std::stoll( timestamp_str );
                            metadata.timestamp = std::chrono::system_clock::from_time_t( time_t_val );
                        }

                        serializer.close();
                        checkpoints_.push_back( metadata );
                    }
                    catch (const std::exception& e)
                    {
                        if (config_.getVerbose())
                        {
                            Utils::Logger::warning_fmt( 
                                "Warning: Failed to read checkpoint {}: {}",
                                entry.path().string(), e.what() );
                        }
                    }
                }
            }

            // Sort by epoch
            std::ranges::sort( checkpoints_,
                []( const auto& a, const auto& b ) { return a.epoch < b.epoch; } );

            if (config_.getVerbose() && !checkpoints_.empty())
            {
                Utils::Logger::info_fmt( 
                    "Found {} checkpoint(s) in {}",
                    checkpoints_.size(), config_.getCheckpointDir().string() );
            }
        }
    private:
        const ModelConfig& config_;
        std::vector<CheckpointMetadata> checkpoints_;

        /**
         * @brief Generate a filename for a checkpoint given its epoch.
         *
         * @param epoch Epoch number used in the filename.
         * @return Formatted filename (no path).
         */
        std::string generateCheckpointFilename( std::size_t epoch ) const
        {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t( now );
            
            return std::format( "checkpoint_epoch_{:04d}.mila", epoch );
        }

        /**
         * @brief Serialize `model` to the given filesystem path.
         *
         * Implementation is backend-specific and must persist weights, optimizer state,
         * and any other information required to resume training.
         *
         * Errors: implementations may throw on I/O or serialization failure.
         *
         * @param model Model to serialize (caller retains ownership).
         * @param filepath Destination file path for the serialized checkpoint.
         */
		template<Compute::DeviceType TDevice>
        void saveModelState( const CompositeModule<TDevice>& model, const std::filesystem::path& filepath )
        {
            Mila::Dnn::Serialization::ZipSerializer serializer;

            if (!serializer.openForWrite( filepath.string() ))
            {
                throw std::runtime_error( "Failed to open checkpoint for writing" );
            }

            // Add metadata
            serializer.addMetadata( "version", "1.0.0" );
            serializer.addMetadata( "timestamp", std::format( "{}", std::chrono::system_clock::now() ) );
            // FIXME: serializer.addMetadata( "mila_version", Mila::getApiVersion().toString() );

            // Serialize network architecture (if needed)
            // serializer.addData( "network/architecture.json", arch_json.data(), arch_json.size() );

            // Serialize weights - delegate to network
            //FIXME: model.network().serialize( serializer, "network/" );

            // Serialize optimizer state
            // FIXME: model.optimizer().serialize( serializer, "optimizer/" );

            // Serialize training history
            // ... 

            serializer.close();
        }
    };
}