module;
#include <math.h>
#include <iostream>
#include <unordered_map>
#include <thrust/host_vector.h>

export module Dnn.Modules.LayerNorm;

import Dnn.Tensor;
import Dnn.Module;
import Compute.OperationBase;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.DeviceMemoryResource;

namespace Mila::Dnn::Modules
{
	using namespace Mila::Dnn::Compute;

	export
	template<typename T, typename MR> requires std::is_same_v<MR, CpuMemoryResource> || std::is_same_v<MR, DeviceMemoryResource>
	class LayerNorm : public Module<T,MR> {
	public:
		LayerNorm( std::string name, int64_t batch_size, int64_t sequence_length, int64_t channels, bool is_training = false, std::string engine_ns = "" )
			: name_( name ), B_( batch_size ), T_( sequence_length ), C_( channels ), is_training_( is_training ) {
			createOperation( engine_ns );
		}

		Tensor<float,MR>& Weight() {
			return weight_;
		}

		Tensor<float,MR>& Bias() {
			return bias_;
		}

		Tensor<float,MR>& Epsilon() {
			return epsilon_;
		}

		size_t parameters() const override {
			return C_ * 2;
		}

		std::string name() const override {
			return name_;
		}

		std::shared_ptr<Tensor<float,MR>> forward( const std::shared_ptr<Tensor<float,MR>> input ) override {
			auto output = std::make_shared<Tensor<float,MR>>( std::vector<size_t>{ B_, T_, C_ } );
			operation_->forward( input, parameters_, output, output_attributes_ );

			return output;
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

		Tensor<float,MR> weight_ = Tensor<float,MR>( std::vector<size_t>{ C_ } );
		Tensor<float,MR> bias_ = Tensor<float,MR>( std::vector<size_t>{ C_ } );

		Tensor<float,MR> mean_ = Tensor<float,MR>( std::vector<size_t>{ B_* T_ } );
		Tensor<float,MR> rstd_ = Tensor<float,MR>( std::vector<size_t>{ B_* T_ } );

		std::vector<std::shared_ptr<Tensor<T,MR>>> parameters_;
		std::vector<std::shared_ptr<Tensor<T,MR>>> output_attributes_;
		std::vector<std::shared_ptr<Tensor<T,MR>>> scalars_;

		std::shared_ptr<Dnn::Compute::OperationBase<float,MR>> operation_;

		void createOperation( std::string op_engine ) {
			parameters_.emplace_back( std::make_shared<Tensor<float,MR>>( weight_ ) );
			parameters_.emplace_back( std::make_shared<Tensor<float,MR>>( bias_ ) );

			output_attributes_.emplace_back( std::make_shared<Tensor<float,MR>>( mean_ ) );
			output_attributes_.emplace_back( std::make_shared<Tensor<float,MR>>( rstd_ ) );

			//scalars_[ scalar_names_::EPSILON ] = std::make_shared<Tensor>( epsilon_ );*/

			if constexpr ( std::is_same_v<MR, Compute::CpuMemoryResource> ) {
				operation_ = OperationRegistry<float, CpuMemoryResource>::instance().createOperation( DeviceType::Cpu, "Cpu::LayerNormOp" );
			}
			/*else {
				operation_ = OperationRegistry<float, DeviceMemoryResource>::instance().createOperation( DeviceType::Cuda, "Cuda::LayerNormOp" );
			}*/
		}
	};
}
