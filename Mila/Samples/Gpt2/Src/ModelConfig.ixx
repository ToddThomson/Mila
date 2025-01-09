export module Gpt2.Gpt2Config;

namespace Mila::Dnn::Gpt2
{
	/**
	* @brief Configuration structure for GPT-2 model.
	*/
	export struct ModelConfig {
		/**
		* @brief Maximum sequence length.
		* @details Example: 1024
		*/
		int max_seq_len{ 1024 };

		/**
		* @brief Vocabulary size.
		* @details Example: 50257
		*/
		int vocab_size{ 50257 };

		/**
		* @brief Padded vocabulary size.
		* @details Example: 50304 (padded to %128==0)
		*/
		int padded_vocab_size{ 50304 };

		/**
		* @brief Number of layers.
		* @details Example: 12
		*/
		int num_layers{ 12 };

		/**
		* @brief Number of heads in attention.
		* @details Example: 12
		*/
		int num_heads{ 12 };

		/**
		* @brief Number of channels.
		* @details Example: 768
		*/
		int channels{ 768 };
	};
}
