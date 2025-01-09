module;
#include <iostream>

export module Model.LayerNorm;

import Mila;

using namespace Mila::Dnn;

namespace App::Model::LayerNorm
{
	export template<typename T>
	class LayerNormModel : public Mila::Dnn::Model<T> {
	public:
		LayerNormModel( std::string name, size_t batch_size, size_t sequence_length, size_t channels )
		: name_( name ), batch_size_( batch_size ), seq_len_( sequence_length ), channels_( channels ) {
		}

		std::string name() override {
			return name_;
		}

		void print() {
			std::cout << "Model: " << name_ << std::endl;
			std::cout << "Batch size: " << batch_size_ << std::endl;
			std::cout << "Sequence length: " << seq_len_ << std::endl;
			std::cout << "Channels: " << channels_ << std::endl;

			Mila::Dnn::Model<T>::print();
		}

	private:
		size_t batch_size_{ 0 };
		size_t seq_len_{ 0 };
		size_t channels_{ 0 };

		std::string name_{ "LayerNorm" };
	};
}