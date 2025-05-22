module;
#include <cuda_runtime.h>
#include "miniz.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <functional>
#include <chrono>

export module Dnn.Model;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.ComputeDevice;
import Compute.CpuDevice;
import Compute.CudaDevice;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Data.DataLoader;
import Dnn.ModelCallback;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Data;

    /**
    * @brief Configuration for training a model.
    */
    export struct TrainingConfig {
        size_t batch_size = 16;            ///< Batch size for training
        size_t epochs = 10;                ///< Number of epochs to train
        float learning_rate = 1e-3f;       ///< Learning rate for optimization
        float weight_decay = 0.0f;         ///< Weight decay (L2 regularization)
        float beta1 = 0.9f;                ///< Beta1 for Adam optimizer
        float beta2 = 0.999f;              ///< Beta2 for Adam optimizer
        float epsilon = 1e-8f;             ///< Epsilon for Adam optimizer
        size_t validation_interval = 1;    ///< Validate every N epochs
        std::string checkpoint_dir = "";   ///< Directory to save checkpoints
        bool save_best_only = true;        ///< Save only the best model
        size_t early_stopping = 0;         ///< Stop after N epochs with no improvement (0 = disabled)
        bool verbose = true;               ///< Print training progress
    };

    /**
    * @brief A class representing a neural network model.
    *
    * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
    * @tparam TInput The input data type for the model.
    * @tparam TOutput The output data type for the model, defaults to TInput.
    */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TInput = float, typename TOutput = TInput>
        requires ValidTensorTypes<TInput, TOutput>
    class Model : public Module<TDeviceType, TInput, TOutput> {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>; ///< Memory resource type
        using ModuleBase = Module<TDeviceType, TInput, TOutput>; ///< Base class type for the module

        /**
        * @brief Constructs a new Model object with the default device context.
        */
        Model()
            : ModuleBase() {
            initializeDevice();
        }

        /**
        * @brief Constructs a new Model object with a specific device context.
        *
        * @param context The device context to use for this model.
        */
        Model( std::shared_ptr<DeviceContext> context )
            : ModuleBase( context ) {
            initializeDevice();
        }

        /**
        * @brief Constructs a new Model object with a specific device name.
        *
        * @param device_name The name of the device to use (e.g., "CUDA:0", "CPU").
        */
        Model( const std::string& device_name )
            : ModuleBase( device_name ) {
            initializeDevice();
        }

        /**
        * @brief Destroys the Model object.
        *
        * Cleans up resources such as CUDA streams.
        */
        ~Model() {
            // Clean up resources
            if ( stream_created_ && stream_ != nullptr ) {
                auto device_type = this->getDeviceContext()->getDevice()->getDeviceType();
                if ( device_type == Compute::DeviceType::Cuda ) {
                    cudaStreamDestroy( stream_ );
                }
            }
        }

        /**
        * @brief Sets the device to use for this model by name.
        *
        * @param device_name The name of the device to use (e.g., "CUDA:0", "CPU").
        */
        void setDevice( const std::string& device_name ) {
            auto context = std::make_shared<Compute::DeviceContext>( device_name );
            this->setDeviceContext( context );
            initializeDevice();
        }

        /**
        * @brief Sets the device to use for this model by CUDA device ID.
        *
        * @param device_id The ID of the CUDA device to use.
        */
        void setDevice( int device_id ) {
            setDevice( "CUDA:" + std::to_string( device_id ) );
        }

        /**
        * @brief Start capturing operations for a CUDA graph.
        *
        * This begins recording operations to a CUDA graph for later replay.
        * Only applicable for CUDA devices.
        */
        void captureGraphBegin() {
            if ( this->getDeviceContext()->getDevice()->getDeviceType() == Compute::DeviceType::Cuda ) {
                cudaStreamBeginCapture( stream_, cudaStreamCaptureModeGlobal );
                graph_capture_active_ = true;
            }
        }

        /**
        * @brief End capturing operations for a CUDA graph.
        *
        * This finalizes the CUDA graph and prepares it for execution.
        * Only applicable for CUDA devices.
        */
        void captureGraphEnd() {
            if ( graph_capture_active_ ) {
                cudaStreamEndCapture( stream_, &cuda_graph_ );
                cudaGraphInstantiate( &cuda_graph_exec_, cuda_graph_, nullptr, nullptr, 0 );
                graph_initialized_ = true;
                graph_capture_active_ = false;
            }
        }

        /**
        * @brief Execute the captured CUDA graph.
        *
        * Replays the previously captured operations for fast execution.
        * Only applicable for CUDA devices with a previously captured graph.
        */
        void executeGraph() {
            if ( graph_initialized_ ) {
                cudaGraphLaunch( cuda_graph_exec_, stream_ );
                cudaStreamSynchronize( stream_ );
            }
        }

        /**
        * @brief Gets the current CUDA stream.
        *
        * @return The CUDA stream used by this model.
        */
        cudaStream_t getStream() const {
            return stream_;
        }

        /**
        * @brief Saves the model's state to a checkpoint file.
        *
        * @param filename The path where the checkpoint will be saved.
        */
        void saveCheckpoint( const std::string& filename ) const {
            mz_zip_archive zip;
            memset( &zip, 0, sizeof( zip ) );
            mz_zip_writer_init_file( &zip, filename.c_str(), 0 );

            for ( const auto& [name, module] : this->getModules() ) {
                module->save( zip );
            }

            mz_zip_writer_finalize_archive( &zip );
            mz_zip_writer_end( &zip );
        }

        /**
        * @brief Loads the model's state from a checkpoint file.
        *
        * @param filename The path to the checkpoint file to load.
        */
        void loadCheckpoint( const std::string& filename ) {
            mz_zip_archive zip;
            memset( &zip, 0, sizeof( zip ) );
            mz_zip_reader_init_file( &zip, filename.c_str(), 0 );

            for ( const auto& [name, module] : this->getModules() ) {
                module->load( zip );
            }

            mz_zip_reader_end( &zip );
        }

        /**
        * @brief Performs a forward pass through the model.
        *
        * @param inputs The input tensor.
        * @param targets Optional target tensor for loss calculation.
        * @throws std::runtime_error if the model has not been built.
        * @return The loss value if targets are provided, otherwise -1.0.
        */
        float forward( const Tensor<TInput, MR>& inputs, const Tensor<TOutput, MR>& targets ) {
            if ( !is_built_ ) {
                throw std::runtime_error( "Model has not been built. Call build() before forward()." );
            }

            // Implementation will be provided by derived classes
            // This is just a base implementation that should be overridden

            last_inputs_ = inputs;
            if ( !targets.empty() ) {
                last_targets_ = targets;
                return calculateLoss( targets );
            }

            return -1.0f;
        }

        /**
        * @brief Performs a backward pass through the model.
        *
        * @throws std::runtime_error if the model has not been built or if forward was not called with targets.
        */
        virtual void backward() {
            if ( !is_built_ ) {
                throw std::runtime_error( "Model has not been built. Call build() before backward()." );
            }

            if ( !this->isTraining() ) return;

            if ( last_targets_.empty() ) {
                throw std::runtime_error( "No targets provided in the last forward pass. Cannot perform backward pass." );
            }

            // Implementation will be provided by derived classes
        }

        /**
        * @brief Zeros out all gradients in the model.
        */
        virtual void zeroGrads() {
            // Implementation will be provided by derived classes
        }

        /**
        * @brief Updates the model parameters using the computed gradients.
        *
        * @param learning_rate The learning rate for the update.
        * @param beta1 Beta1 parameter for Adam optimizer.
        * @param beta2 Beta2 parameter for Adam optimizer.
        * @param epsilon Epsilon parameter for Adam optimizer.
        * @param weight_decay Weight decay parameter for regularization.
        * @param step Current optimization step for Adam.
        */
        virtual void updateParameters(
            float learning_rate,
            float beta1 = 0.9f,
            float beta2 = 0.999f,
            float epsilon = 1e-8f,
            float weight_decay = 0.0f,
            size_t step = 1
        ) {
            // Implementation will be provided by derived classes
        }

        /**
        * @brief Builds the model.
        *
        * Sets the training mode for all modules and performs any necessary graph validation or optimizations.
        * @throws std::runtime_error if the model has already been built.
        */
        void build() {
            if ( is_built_ ) {
                throw std::runtime_error( "Model has already been built." );
            }

            for ( auto& [_, module] : this->getModules() ) {
                module->setTraining( is_training_ );
            }

            is_built_ = true;
        }

        /**
        * @brief Sets the training mode for the model.
        *
        * @param training The training mode to set.
        */
        void setTrainingMode( bool training ) {
            is_training_ = training;

            if ( is_built_ ) {
                for ( auto& [_, module] : this->getModules() ) {
                    module->setTraining( is_training_ );
                }
            }
        }

        /**
        * @brief Train the model using the provided data loader and configuration.
        *
        * @param train_loader The data loader for training data.
        * @param val_loader Optional data loader for validation data.
        * @param config Training configuration parameters.
        * @param callbacks Optional list of callbacks to be invoked during training.
        * @return A map of final training metrics.
        */
        template<typename TDataLoader>
        std::unordered_map<std::string, float> train(
            TDataLoader& train_loader,
            TDataLoader* val_loader = nullptr,
            const TrainingConfig& config = {},
            const std::vector<ModelCallback<TInput, TOutput>*>& callbacks = {}
        ) {
            if ( !is_built_ ) {
                build();
            }

            setTrainingMode( true );

            // Initialize tensors for input and target data
            Tensor<TInput, MR> inputs;
            Tensor<TOutput, MR> targets;

            // Metrics to track during training
            std::unordered_map<std::string, float> metrics;
            float best_val_loss = std::numeric_limits<float>::max();
            size_t epochs_without_improvement = 0;

            // Notify callbacks that training is beginning
            for ( auto callback : callbacks ) {
                callback->onTrainingBegin();
            }

            // Main training loop
            for ( size_t epoch = 0; epoch < config.epochs; ++epoch ) {
                // Notify callbacks that epoch is beginning
                for ( auto callback : callbacks ) {
                    callback->onEpochBegin( epoch );
                }

                // Training phase
                train_loader.reset();
                float epoch_loss = 0.0f;
                size_t batch_count = 0;

                auto start_time = std::chrono::high_resolution_clock::now();

                while ( train_loader.nextBatch( inputs, targets ) ) {
                    // Notify callbacks that batch is beginning
                    for ( auto callback : callbacks ) {
                        callback->onBatchBegin( batch_count );
                    }

                    // Forward pass
                    zeroGrads();
                    float batch_loss = forward( inputs, targets );

                    // Backward pass and parameter update
                    backward();
                    updateParameters(
                        config.learning_rate,
                        config.beta1,
                        config.beta2,
                        config.epsilon,
                        config.weight_decay,
                        epoch * train_loader.numBatches() + batch_count + 1
                    );

                    epoch_loss += batch_loss;
                    batch_count++;

                    // Metrics for this batch
                    std::unordered_map<std::string, float> batch_metrics = {
                        {"loss", batch_loss}
                    };

                    // Notify callbacks that batch is ending
                    for ( auto callback : callbacks ) {
                        callback->onBatchEnd( batch_count, batch_metrics );
                    }

                    // Optional progress reporting
                    if ( config.verbose && batch_count % 10 == 0 ) {
                        std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs
                            << " - Batch " << batch_count << "/" << train_loader.numBatches()
                            << " - Loss: " << batch_loss << std::endl;
                    }
                }

                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end_time - start_time;

                // Calculate average loss for the epoch
                epoch_loss /= batch_count;
                metrics[ "train_loss" ] = epoch_loss;

                // Validation phase
                if ( val_loader && (epoch + 1) % config.validation_interval == 0 ) {
                    float val_loss = evaluate( *val_loader );
                    metrics[ "val_loss" ] = val_loss;

                    // Check for improvement for early stopping
                    if ( val_loss < best_val_loss ) {
                        best_val_loss = val_loss;
                        epochs_without_improvement = 0;

                        // Save best model if requested
                        if ( config.save_best_only && !config.checkpoint_dir.empty() ) {
                            saveCheckpoint( config.checkpoint_dir + "/best_model.ckpt" );
                        }
                    }
                    else {
                        epochs_without_improvement++;
                    }
                }

                // Save checkpoint for this epoch if directory is provided
                if ( !config.checkpoint_dir.empty() && !config.save_best_only ) {
                    saveCheckpoint( config.checkpoint_dir + "/model_epoch_" + std::to_string( epoch + 1 ) + ".ckpt" );
                }

                // Report progress
                if ( config.verbose ) {
                    std::cout << "Epoch " << (epoch + 1) << "/" << config.epochs << " - Time: "
                        << duration.count() << "s - Train Loss: " << epoch_loss;

                    if ( val_loader && (epoch + 1) % config.validation_interval == 0 ) {
                        std::cout << " - Val Loss: " << metrics[ "val_loss" ];
                    }

                    std::cout << std::endl;
                }

                // Notify callbacks that epoch is ending
                for ( auto callback : callbacks ) {
                    callback->onEpochEnd( epoch, metrics );
                }

                // Check for early stopping
                if ( config.early_stopping > 0 && epochs_without_improvement >= config.early_stopping ) {
                    if ( config.verbose ) {
                        std::cout << "Early stopping triggered after " << (epoch + 1) << " epochs." << std::endl;
                    }
                    break;
                }
            }

            // Notify callbacks that training is ending
            for ( auto callback : callbacks ) {
                callback->onTrainingEnd();
            }

            return metrics;
        }

        /**
        * @brief Evaluate the model on a dataset.
        *
        * @param data_loader The data loader for evaluation data.
        * @param verbose Whether to print evaluation progress.
        * @return The average loss on the evaluation dataset.
        */
        template<typename TDataLoader>
        float evaluate(
            TDataLoader& data_loader,
            bool verbose = false
        ) {
            setTrainingMode( false );

            // Initialize tensors for input and target data
            Tensor<TInput, MR> inputs;
            Tensor<TOutput, MR> targets;

            // Evaluation metrics
            float total_loss = 0.0f;
            size_t batch_count = 0;

            data_loader.reset();

            if ( verbose ) {
                std::cout << "Evaluating model..." << std::endl;
            }

            while ( data_loader.nextBatch( inputs, targets ) ) {
                float batch_loss = forward( inputs, targets );
                total_loss += batch_loss;
                batch_count++;

                if ( verbose && batch_count % 10 == 0 ) {
                    std::cout << "Batch " << batch_count << "/" << data_loader.numBatches()
                        << " - Loss: " << batch_loss << std::endl;
                }
            }

            float avg_loss = batch_count > 0 ? total_loss / batch_count : 0.0f;

            if ( verbose ) {
                std::cout << "Evaluation complete - Average Loss: " << avg_loss << std::endl;
            }

            return avg_loss;
        }

        /**
        * @brief Predict outputs for the given inputs.
        *
        * @param inputs The input tensor.
        * @return The output tensor.
        */
        template<typename TMR>
        Tensor<TOutput, TMR> predict( const Tensor<TInput, TMR>& inputs ) {
            setTrainingMode( false );
            forward( inputs );
            // This should be overridden by derived classes to return the actual output
            return {};
        }

        /**
        * @brief Calculate the loss for the given targets and current model outputs.
        *
        * @param targets The target tensor.
        * @return The loss value.
        */
        template<typename TMR>
        float calculateLoss( const Tensor<TOutput, TMR>& targets ) {
            // This should be overridden by derived classes
            return 0.0f;
        }

        /**
        * @brief Calculates the total number of parameters in the model.
        *
        * @return size_t The total number of parameters.
        */
        size_t parameters() const {
            size_t total_parameters = 0;
            for ( const auto& [_, module] : this->getModules() ) {
                total_parameters += module->parameterCount();
            }
            return total_parameters;
        }

        /**
        * @brief Prints the model's structure and total number of parameters.
        */
        void print() const {
            std::cout << "Model Summary:" << std::endl;
            std::cout << "=============" << std::endl;
            std::cout << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;

            std::cout << "Modules: " << std::endl;
            for ( const auto& [name, module] : this->getModules() ) {
                std::cout << "  " << name << ": ";
                std::cout << module->toString() << std::endl;
            }

            std::cout << "Total parameters: " << parameters() << std::endl;
            std::cout << "Training mode: " << (is_training_ ? "ON" : "OFF") << std::endl;
            std::cout << "Built: " << (is_built_ ? "YES" : "NO") << std::endl;
        }

        /**
        * @brief Gets the compute device for this model.
        *
        * @return Reference to the model's compute device.
        */
        Compute::ComputeDevice& getDevice() const {
            return *this->getDeviceContext()->getDevice();
        }

    protected:
        /**
        * @brief The most recent input tensor provided to forward().
        */
        Tensor<TInput, MR> last_inputs_;

        /**
        * @brief The most recent target tensor provided to forward().
        */
        Tensor<TOutput, MR> last_targets_;

    private:
        /**
        * @brief Initializes device-specific resources.
        */
        void initializeDevice() {
            // Clean up old resources if needed
            if ( stream_created_ && stream_ != nullptr ) {
                auto old_device_type = old_device_type_;
                if ( old_device_type == Compute::DeviceType::Cuda ) {
                    cudaStreamDestroy( stream_ );
                }
                stream_ = nullptr;
                stream_created_ = false;
            }

            // Get the current device type
            auto device_type = this->getDeviceContext()->getDevice()->getDeviceType();
            old_device_type_ = device_type;

            // Initialize resources for the new device
            if ( device_type == Compute::DeviceType::Cuda ) {
                cudaStreamCreate( &stream_ );
                stream_created_ = true;

                // Set the stream on the device context
                this->getDeviceContext()->setStream( stream_ );
            }
        }

        /**
        * @brief CUDA graph for optimized execution.
        */
        cudaGraph_t cuda_graph_{ nullptr };

        /**
        * @brief Executable instance of the CUDA graph.
        */
        cudaGraphExec_t cuda_graph_exec_{ nullptr };

        /**
        * @brief Flag indicating whether CUDA graph has been initialized.
        */
        bool graph_initialized_{ false };

        /**
        * @brief Flag indicating whether CUDA graph capture is active.
        */
        bool graph_capture_active_{ false };

        bool is_built_{ false }; ///< Indicates whether the model has been built.
        bool is_training_{ false }; ///< Indicates whether the model is in training mode.

        cudaStream_t stream_{ nullptr }; ///< The CUDA stream for device memory resource.
        bool stream_created_{ false }; ///< Flag indicating whether we created the stream.
        Compute::DeviceType old_device_type_{ Compute::DeviceType::Cpu }; ///< Previous device type for cleanup.
    };

    /**
     * @brief Type alias for CPU-based models with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CpuModel = Model<DeviceType::Cpu, TInput, TOutput>;

    /**
     * @brief Type alias for CUDA-based models with customizable tensor types.
     *
     * @tparam TInput Data type of the input tensor elements.
     * @tparam TOutput Data type of the output tensor elements, defaults to TInput.
     */
    export template<typename TInput = float, typename TOutput = TInput>
        using CudaModel = Model<DeviceType::Cuda, TInput, TOutput>;
}