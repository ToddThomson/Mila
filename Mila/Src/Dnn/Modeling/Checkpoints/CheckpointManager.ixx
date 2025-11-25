/**
 * @file CheckpointManager.ixx
 * @brief Checkpoint metadata and history management.
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
#include <system_error>

export module Modeling.CheckpointManager;

import Dnn.ModelConfig;
import Modeling.CheckpointMetaData;
import Serialization.OpenMode;
import Serialization.ZipSerializer;
import Utils.Logger;
import nlohmann.json;

namespace Mila::Dnn::Modeling
{
    using json = nlohmann::json;
	using namespace Mila::Dnn::Serialization;

    /**
     * @brief Manages checkpoint metadata, history, and rolling-window cleanup.
     *
     * Responsibilities:
     *  - Track checkpoint metadata in memory
     *  - Generate standardized checkpoint filenames
     *  - Maintain rolling window of checkpoints (automatic cleanup)
     *  - Query checkpoint history (latest, best by validation loss)
     *  - Scan checkpoint directory to rebuild history
     *
     * Does NOT handle serialization directly - Model owns ModelArchive lifecycle.
     *
     * Threading: not thread-safe; callers must synchronize if used concurrently.
     */
    export class CheckpointManager
    {
    public:

        explicit CheckpointManager( const ModelConfig& config )
            : config_( config )
        {
            std::filesystem::create_directories( config.getCheckpointDir() );
        }

        void addCheckpoint( const CheckpointMetadata& metadata )
        {
            checkpoints_.push_back( metadata );

            if (checkpoints_.size() > config_.getMaxCheckpoints())
            {
                auto oldest = checkpoints_.front();

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
        }

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

        const std::vector<CheckpointMetadata>& getAllCheckpoints() const
        {
            return checkpoints_;
        }

        std::optional<CheckpointMetadata> getLatestCheckpoint() const
        {
            if (checkpoints_.empty())
            {
                return std::nullopt;
            }
            return checkpoints_.back();
        }

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
                        ZipSerializer serializer;
                        
                        if (!serializer.open( entry.path().string(), OpenMode::Read ) )
                        {
                            continue;
                        }

                        auto meta_str = serializer.getMetadata( "model/meta.json" );
                        if (meta_str.empty())
                        {
                            serializer.close();
                            continue;
                        }

                        json model_meta = json::parse( meta_str );

                        CheckpointMetadata metadata;
                        metadata.filepath = entry.path();
                        metadata.epoch = model_meta.value( "epoch", 0ULL );
                        metadata.train_loss = model_meta.value( "train_loss", 0.0 );
                        metadata.val_loss = model_meta.value( "val_loss", 0.0 );

                        auto timestamp_val = model_meta.value( "timestamp", 0LL );
                        if (timestamp_val > 0)
                        {
                            metadata.timestamp = std::chrono::system_clock::from_time_t( timestamp_val );
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

            std::ranges::sort( checkpoints_,
                []( const auto& a, const auto& b ) { return a.epoch < b.epoch; } );

            if (config_.getVerbose() && !checkpoints_.empty())
            {
                Utils::Logger::info_fmt(
                    "Found {} checkpoint(s) in {}",
                    checkpoints_.size(), config_.getCheckpointDir().string() );
            }
        }

        std::string generateCheckpointFilename( std::size_t epoch ) const
        {
            return std::format( "checkpoint_epoch_{:04d}.mila", epoch );
        }

    private:
        const ModelConfig& config_;
        std::vector<CheckpointMetadata> checkpoints_;
    };
}