module;
#include <vector>
#include <limits>

export module Modeling.TrainingHistory;

namespace Mila::Dnn
{
    /**
     * @brief Struct to hold training history data.
     *
     * Records training and validation losses and metrics per epoch,
     * as well as tracking the current epoch, best validation loss,
     * and epochs without improvement for early stopping.
     */
    export struct TrainingHistory
    {
        std::vector<double> train_losses;
        std::vector<double> val_losses;
        std::vector<double> train_metrics;
        std::vector<double> val_metrics;
        std::size_t current_epoch = 0;
        double best_val_loss = std::numeric_limits<double>::max();
        std::size_t epochs_without_improvement = 0;
    };
}