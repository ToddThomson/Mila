module;
#include <unordered_map>
#include <string>

export module Dnn.ModelCallback;

namespace Mila::Dnn
{
	/**
		* @brief Interface for callbacks during training.
		*/
	export template<typename TInput, typename TPrecision>
		class ModelCallback {
		public:
			virtual ~ModelCallback() = default;

			/**
			* @brief Called at the beginning of training.
			*/
			virtual void onTrainingBegin() {}

			/**
			* @brief Called at the end of training.
			*/
			virtual void onTrainingEnd() {}

			/**
			* @brief Called at the beginning of each epoch.
			* @param epoch The current epoch number.
			*/
			virtual void onEpochBegin( size_t epoch ) {}

			/**
			* @brief Called at the end of each epoch.
			* @param epoch The current epoch number.
			* @param metrics The metrics from the epoch.
			*/
			virtual void onEpochEnd( size_t epoch, const std::unordered_map<std::string, float>& metrics ) {}

			/**
			* @brief Called at the beginning of each batch.
			* @param batch The current batch number.
			*/
			virtual void onBatchBegin( size_t batch ) {}

			/**
			* @brief Called at the end of each batch.
			* @param batch The current batch number.
			* @param metrics The metrics from the batch.
			*/
			virtual void onBatchEnd( size_t batch, const std::unordered_map<std::string, float>& metrics ) {}
	};
}