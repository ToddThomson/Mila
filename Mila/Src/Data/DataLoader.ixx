export module Data.DataLoader;

import Compute.DeviceType;

namespace Mila::Data
{
	using namespace Mila::Dnn;

	/**
	* @brief Template for data loaders used in training and evaluation.
	*
	* @tparam TInput The input data type.
	* @tparam TDataType The precision type for computations.
	* @tparam TDeviceType The device type (CPU or CUDA).
	*/
	export template<typename TInput, typename TPrecision, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
		class DataLoader {
		public:
			virtual ~DataLoader() = default;

			/**
			* @brief Get the number of batches in the dataset.
			* @return The number of batches.
			*/
			virtual size_t numBatches() const = 0;

			/**
			* @brief Get the size of each batch.
			* @return The batch size.
			*/
			virtual size_t batchSize() const = 0;

			/**
			* @brief Reset the loader to the beginning of the dataset.
			*/
			virtual void reset() = 0;

			/**
			* @brief Get the next batch of data.
			* @param inputs The tensor to store input data.
			* @param targets The tensor to store target data.
			* @return True if there was data to load, false if end of dataset.
			*/
			virtual bool nextBatch() = 0;
	};
}