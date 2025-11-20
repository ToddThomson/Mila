/**
 * @file ModelConfig.ixx
 * @brief Configuration holder for model training and checkpointing.
 *
 * Provides a small, fluent API used by ModelBuilder and training utilities to
 * configure epochs, batch size, checkpointing and early-stopping behaviour.
 */

module;
#include <cstddef>
#include <filesystem>
#include <vector>
#include <limits>
#include <utility>

export module Modeling.ModelConfig;

namespace Mila::Dnn::Modeling
{
    /**
     * @brief Runtime configuration used to control model training and persistence.
     *
     * ModelConfig is a lightweight value-type that exposes fluent setters for
     * common training parameters and corresponding getters. Setters do not
     * validate values; callers should ensure sensible inputs before use.
     *
     * Threading: this object is not synchronized; callers must synchronize access
     * if it is shared between threads.
     */
    export class ModelConfig
    {
    public:
        ModelConfig() = default;

        // Setters (fluent)

        /**
         * @brief Set number of training epochs.
         *
         * @param epochs Number of epochs to run.
         * @return Reference to this ModelConfig for chaining.
         */
        auto& epochs( std::size_t epochs )
        {
            epochs_ = epochs;
            return *this;
        }

        /**
         * @brief Set batch size used for training/validation.
         *
         * @param batch_size Number of samples per batch.
         * @return Reference to this ModelConfig for chaining.
         */
        auto& batchSize( std::size_t batch_size )
        {
            batch_size_ = batch_size;
            return *this;
        }

        /**
         * @brief Set initial learning rate.
         *
         * The learning rate semantics are optimizer-specific; this value is
         * supplied to optimizer factories or passed to optimizers directly.
         *
         * @param lr Learning rate value.
         * @return Reference to this ModelConfig for chaining.
         */
        auto& learningRate( double lr )
        {
            learning_rate_ = lr;
            return *this;
        }

        /**
         * @brief Fraction of training data reserved for validation.
         *
         * @param split Fraction in range [0.0, 1.0] indicating validation split.
         * @return Reference to this ModelConfig for chaining.
         */
        auto& validationSplit( double split )
        {
            validation_split_ = split;
            return *this;
        }

        /**
         * @brief Frequency (in epochs) at which checkpoints are written.
         *
         * @param frequency Number of epochs between checkpoint saves.
         * @return Reference to this ModelConfig for chaining.
         */
        auto& checkpointFrequency( std::size_t frequency )
        {
            checkpoint_frequency_ = frequency;
            return *this;
        }

        /**
         * @brief Maximum number of saved checkpoints to retain.
         *
         * Older checkpoints are removed when this limit is exceeded.
         *
         * @param max_checkpoints Maximum number of checkpoints to keep.
         * @return Reference to this ModelConfig for chaining.
         */
        auto& maxCheckpoints( std::size_t max_checkpoints )
        {
            max_checkpoints_ = max_checkpoints;
            return *this;
        }

        /**
         * @brief Directory path where checkpoints will be stored.
         *
         * The path is moved into the configuration object.
         *
         * @param dir Filesystem path for checkpoint files.
         * @return Reference to this ModelConfig for chaining.
         */
        auto& checkpointDir( std::filesystem::path dir )
        {
            checkpoint_dir_ = std::move( dir );
            return *this;
        }

        /**
         * @brief Enable early stopping with patience (number of epochs).
         *
         * Sets early-stopping enabled flag and the patience value.
         *
         * @param patience Number of epochs without improvement before stopping.
         * @return Reference to this ModelConfig for chaining.
         */
        auto& enableEarlyStopping( std::size_t patience )
        {
            early_stopping_enabled_ = true;
            early_stopping_patience_ = patience;
            return *this;
        }

        /**
         * @brief Enable or disable verbose logging during training.
         *
         * @param verbose true to enable verbose logging, false to disable.
         * @return Reference to this ModelConfig for chaining.
         */
        auto& verbose( bool verbose )
        {
            verbose_ = verbose;
            return *this;
        }

        // Getters (prefixed with 'get')

        /**
         * @brief Get number of configured epochs.
         *
         * @return Configured epochs.
         */
        std::size_t getEpochs() const
        {
            return epochs_;
        }

        /**
         * @brief Get configured batch size.
         *
         * @return Batch size.
         */
        std::size_t getBatchSize() const
        {
            return batch_size_;
        }

        /**
         * @brief Get configured learning rate.
         *
         * Note: interpretation of this value is optimizer-specific.
         *
         * @return Learning rate.
         */
        double getLearningRate() const
        {
            return learning_rate_;
        }

        /**
         * @brief Get validation split fraction.
         *
         * @return Validation split in [0.0, 1.0].
         */
        double getValidationSplit() const
        {
            return validation_split_;
        }

        /**
         * @brief Get checkpoint saving frequency (in epochs).
         *
         * @return Checkpoint frequency.
         */
        std::size_t getCheckpointFrequency() const
        {
            return checkpoint_frequency_;
        }

        /**
         * @brief Get maximum number of checkpoints to retain.
         *
         * @return Maximum checkpoints.
         */
        std::size_t getMaxCheckpoints() const
        {
            return max_checkpoints_;
        }

        /**
         * @brief Get configured checkpoint directory path.
         *
         * @return Reference to the checkpoint directory path.
         */
        const std::filesystem::path& getCheckpointDir() const
        {
            return checkpoint_dir_;
        }

        /**
         * @brief Return whether early stopping is enabled.
         *
         * @return true if early stopping is enabled, false otherwise.
         */
        bool getEarlyStoppingEnabled() const
        {
            return early_stopping_enabled_;
        }

        /**
         * @brief Get early stopping patience (epochs).
         *
         * @return Patience parameter for early stopping.
         */
        std::size_t getEarlyStoppingPatience() const
        {
            return early_stopping_patience_;
        }

        /**
         * @brief Whether verbose logging is enabled.
         *
         * @return true if verbose logging is enabled.
         */
        bool getVerbose() const
        {
            return verbose_;
        }

    private:

        std::size_t epochs_ = 10;
        std::size_t batch_size_ = 32;
        double learning_rate_ = 0.001;
        double validation_split_ = 0.0;
        std::size_t checkpoint_frequency_ = 1;
        std::size_t max_checkpoints_ = 5;
        std::filesystem::path checkpoint_dir_ = "checkpoints";
        bool early_stopping_enabled_ = false;
        std::size_t early_stopping_patience_ = 5;
        bool verbose_ = false;
    };
}