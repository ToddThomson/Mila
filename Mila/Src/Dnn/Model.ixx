module;
#include <cuda_runtime.h>
#include "miniz.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <functional>
#include <chrono>

export module Dnn.Model;

import Dnn.Module;
import Dnn.TensorTraits;
import Compute.DeviceType;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;
import Data.DataLoader;
import Dnn.ModelCallback;

namespace Mila::Dnn
{
	using namespace Mila::Data;

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
	* @tparam TInput The input data type for the model.
	* @tparam TPrecision The precision type used for model computations, defaults to input type.
	* @tparam TDeviceType The device type (CPU or CUDA) the model will run on.
	*/
	export
		template<typename TInput, typename TPrecision = TInput, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
		requires ValidTensorTypes<TInput, TPrecision>
	class Model : public Module<TInput, TPrecision, TDeviceType> {
	public:
		/**
		* @brief Type alias for memory resource based on device type.
		*/
		using MR = std::conditional_t<TDeviceType == Compute::DeviceType::Cuda, Compute::DeviceMemoryResource, Compute::HostMemoryResource>;

		/**
		* @brief Constructs a new Model object.
		*
		* Initializes CUDA stream if the memory resource is a device memory resource.
		*/
		Model() {
			if constexpr ( std::is_same_v<MR, Compute::DeviceMemoryResource> ) {
				cudaStreamCreate( &stream_ );
			}
		}

		/**
		* @brief Destroys the Model object.
		*
		* Destroys CUDA stream if the memory resource is a device memory resource.
		*/
		~Model() {
			if constexpr ( std::is_same_v<MR, Compute::DeviceMemoryResource> ) {
				cudaStreamDestroy( stream_ );
			}
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
		virtual float forward( const Tensor<TInput, MR>& inputs, const Tensor<TInput, MR>& targets = {} ) {
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

			if ( !is_training_ ) return;

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
				module->setTrainingMode( is_training_ );
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
					module->setTrainingMode( is_training_ );
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
		std::unordered_map<std::string, float> train(
			DataLoader<TInput, TPrecision, TDeviceType>& train_loader,
			DataLoader<TInput, TPrecision, TDeviceType>* val_loader = nullptr,
			const TrainingConfig& config = {},
			const std::vector<ModelCallback<TInput, TPrecision>*>& callbacks = {}
		) {
			if ( !is_built_ ) {
				build();
			}

			setTrainingMode( true );

			// Initialize tensors for input and target data
			Tensor<TInput, MR> inputs;
			Tensor<TInput, MR> targets;

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
		float evaluate(
			DataLoader<TInput, TPrecision, TDeviceType>& data_loader,
			bool verbose = false
		) {
			setTrainingMode( false );

			// Initialize tensors for input and target data
			Tensor<TInput, MR> inputs;
			Tensor<TInput, MR> targets;

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
		virtual Tensor<TPrecision, MR> predict( const Tensor<TInput, MR>& inputs ) {
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
		virtual float calculateLoss( const Tensor<TInput, MR>& targets ) {
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
				total_parameters += module->parameters();
			}
			return total_parameters;
		}

		/**
		* @brief Prints the model's structure and total number of parameters.
		*/
		void print() const {
			std::cout << "Model Summary:" << std::endl;
			std::cout << "=============" << std::endl;

			std::cout << "Modules: " << std::endl;
			for ( const auto& [name, module] : this->getModules() ) {
				std::cout << "  " << name << ": ";
				module->print();
			}

			std::cout << "Total parameters: " << parameters() << std::endl;
			std::cout << "Training mode: " << (is_training_ ? "ON" : "OFF") << std::endl;
			std::cout << "Built: " << (is_built_ ? "YES" : "NO") << std::endl;
		}

	protected:
		/**
		* @brief The most recent input tensor provided to forward().
		*/
		Tensor<TInput, MR> last_inputs_;

		/**
		* @brief The most recent target tensor provided to forward().
		*/
		Tensor<TInput, MR> last_targets_;

	private:
		/**
		* @brief CUDA graph for optimized execution.
		*/
		cudaGraph_t cuda_graph;

		/**
		* @brief Executable instance of the CUDA graph.
		*/
		cudaGraphExec_t cuda_graph_exec;

		/**
		* @brief Flag indicating whether CUDA graph has been initialized.
		*/
		bool cuda_graph_initialized = false;

		bool is_built_{ false }; ///< Indicates whether the model has been built.
		bool is_training_{ false }; ///< Indicates whether the model is in training mode.

		cudaStream_t stream_{}; ///< The CUDA stream for device memory resource.
	};
}
