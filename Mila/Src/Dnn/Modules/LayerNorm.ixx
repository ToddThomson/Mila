module;
#include <math.h>
#include <iostream>
#include <unordered_map>

export module Dnn.Modules.LayerNorm;

import Dnn.Tensor;
import Dnn.Module;
import Compute.OperationBase;
import Compute.OperationsFactory;

namespace Mila::Dnn::Modules
{
	export template<typename T>
		class LayerNorm : public Module<T> {
		public:
			LayerNorm( std::string name, int64_t batch_size, int64_t sequence_length, int64_t channels, bool is_training = false )
				: name_( name ), B_( batch_size ), T_( sequence_length ), C_( channels ), is_training_( is_training ) {

				createOperation();

				init_tensors();
			}

			// --------------------------------------------------------------------
			// Properties..

			Tensor<float>& Weight() {
				return weight_;
			}

			Tensor<float>& Bias() {
				return bias_;
			}

			Tensor<float>& Epsilon() {
				return epsilon_;
			}

			// --------------------------------------------------------------------
			// Module interface methods..

			std::shared_ptr<Tensor<float>> forward( const std::shared_ptr<Tensor<float>>& input ) override {
				auto output = std::make_shared<Tensor<float>>( std::vector<size_t>{ B_, T_, C_ } );
				operation_->forward( input, input_attributes_, output, output_attributes_ );

				return output;
			}

			size_t parameters() const override {
				return C_ * 2;
			}

			std::string name() const override {
				return name_;
			}

			void print() const override {
				std::cout << "Module: " << name_ << std::endl;
				std::cout << "Parameters: " << parameters() << std::endl;
			}

		private:
			std::string name_{ "LayerNorm" };
			float epsilon_{ 1e-05f };
			size_t B_{ 0 };
			size_t T_{ 0 };
			size_t C_{ 0 };

			// TODO: Feature not yet implemented
			bool is_training_{ false };

			Tensor<float> weight_ = Tensor<float>( std::vector<size_t>{ C_ } );
			Tensor<float> bias_ = Tensor<float>( std::vector<size_t>{ C_ } );

			Tensor<float> mean_ = Tensor<float>( std::vector<size_t>{ B_* T_ } );
			Tensor<float> rstd_ = Tensor<float>( std::vector<size_t>{ B_* T_ } );

			std::vector<std::shared_ptr<Tensor<T>>> input_attributes_;
			std::vector<std::shared_ptr<Tensor<T>>> output_attributes_;
			std::vector<std::shared_ptr<Tensor<T>>> scalars_;

			std::shared_ptr<Dnn::Compute::OperationBase<float>> operation_;

			void createOperation() {
				input_attributes_.emplace_back( std::make_shared<Tensor<float>>( weight_ ) );
				input_attributes_.emplace_back( std::make_shared<Tensor<float>>( bias_ ) );

				output_attributes_.emplace_back( std::make_shared<Tensor<float>>( mean_ ) );
				output_attributes_.emplace_back( std::make_shared<Tensor<float>>( rstd_ ) );

				//scalars_[ scalar_names_::EPSILON ] = std::make_shared<Tensor>( epsilon_ );*/

				operation_ = Compute::OperationsFactory<float>::createOperation( "CPU", "Cpu::LayerNormOp<float>" );
			}

			void init_tensors() {
				//xavier( weight_, C_, C_ );
			}
	};
}
