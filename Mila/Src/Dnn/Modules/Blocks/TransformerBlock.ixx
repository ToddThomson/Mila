module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <type_traits>
#include <stdexcept>

export module Dnn.Blocks.TransformerBlock;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.ComputeDevice;
import Compute.CpuDevice;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Dnn.Module;
import Dnn.Modules.LayerNorm;
import Dnn.Modules.Linear;
import Dnn.Modules.Attention;
import Dnn.Modules.Residual;
import Dnn.Blocks.MLP;

namespace Mila::Dnn
{
	export
	template<typename TInput, typename TCompute = TInput, typename TDevice = Compute::CpuDevice>
		requires ValidTensorTypes<TInput, TCompute> && std::is_base_of_v<Compute::ComputeDevice, TDevice>
	class TransformerBlock : public Module<TInput, TCompute, TDevice> {
	public:
		using MR = TDevice::MR;

		TransformerBlock( std::string name, const std::vector<size_t>& input_shape, const size_t num_heads, bool is_training = false )
			: input_shape_{ validate_shape( input_shape ) }, num_heads_{ num_heads } {
			this->setName( name );
			this->setTraining( is_training );

			auto B = input_shape_[ 0 ];
			auto T = input_shape_[ 1 ];
			auto C = input_shape_[ 2 ];

			ln_1_ = std::make_shared<LayerNorm<TInput, TCompute, TDevice>>( "ln_1", input_shape_ );
			fc_ = std::make_shared<Linear<TInput, TCompute, TDevice>>( "fc_", C, 3 * C );
			attn_ = std::make_shared<MultiHeadAttention<TInput, TCompute, TDevice>>( "attn_", input_shape_, num_heads_ );
			ln_2_ = std::make_shared<LayerNorm<TInput, TCompute, TDevice>>( "ln_2", input_shape_ );
			mlp_ = std::make_shared<MLP<TInput, TCompute, TDevice>>( "mlp_", input_shape_, 4 * C);
			residual_ = std::make_shared<Residual<TInput, TCompute, TDevice>>( "res_" );

			this->addModule( ln_1_ );
			this->addModule( fc_ );
			this->addModule( attn_ );
			this->addModule( ln_2_ );
			this->addModule( mlp_ );
			this->addModule( residual_ );

			// Pre-allocate output tensors for the Transformer block layers
			ln_1_output_ = Tensor<TCompute, MR>( input_shape_ );

			fc_output_ = Tensor<TCompute, MR>( { B, T, 3 * C } );
			attn_output_ = Tensor<TCompute, MR>( input_shape_ );
			ln_2_output_ = Tensor<TCompute, MR>( input_shape_ );
			mlp_output_ = Tensor<TCompute, MR>( input_shape_ );
			residual_output_ = Tensor<TCompute, MR>( input_shape_ );
		}

		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute,MR>& output ) override {
			ln_1_->forward( input, ln_1_output_ );
			//std::cout << "ln1_output_" << std::endl;
			//ln1_output_.print();

			fc_->forward( ln_1_output_, fc_output_ );
			//std::cout << "fc1_output_" << std::endl;
			//fc1_output_.print();

			attn_->forward( fc_output_, attn_output_ );
			//std::cout << "attn_output_" << std::endl;
			//attn_output_.print();

			residual_->forward( attn_output_, residual_output_ );
			//std::cout << "residual_output_" << std::endl;
			//residual_output_.print();
			
			ln_2_->forward( residual_output_, ln_2_output_ );
			//std::cout << "ln2_output_" << std::endl;
			//ln2_output_.print();

			mlp_->forward( ln_2_output_, output );
			//std::cout << "mlp_output_" << std::endl;
			//output.print();
		}

		//const std::vector<std::shared_ptr<Module<TInput, TCompute, TDevice>>>& getSubModules() const override {
		//	return {};// { ln_1_, fc_, attn_, ln_2_, mlp_, residual_ };
		//}

		const std::vector<std::shared_ptr<Tensor<TCompute, MR>>>& getParameters() const override {
			return {};
		}

        size_t parameterCount() const override {
			return 0;
		}

		void save( mz_zip_archive& zip ) const override {
			// Save the state of the child modules
			for ( const auto& module : this->getSubModules() ) {
				module->save( zip );
			}
		}

		void load( mz_zip_archive& zip ) override {
			for ( const auto& module : this->getSubModules() ) {
				module->load( zip );
			}
		}

		void print() const override {
			std::cout << "Module: " << this->getName() << std::endl;
			std::cout << "Parameter count: " << parameterCount() << std::endl;
		}

	private:
		std::vector<size_t> input_shape_; ///< The input shape.
		size_t num_heads_; ///< The number of attention heads.

		std::shared_ptr<LayerNorm<TInput, TCompute, TDevice>> ln_1_{ nullptr };
		std::shared_ptr<Linear<TInput, TCompute, TDevice>> fc_{ nullptr };
		std::shared_ptr<MultiHeadAttention<TInput, TCompute, TDevice>> attn_{ nullptr };
		std::shared_ptr<LayerNorm<TInput, TCompute, TDevice>> ln_2_{ nullptr };
		std::shared_ptr<MLP<TInput, TCompute, TDevice>> mlp_{ nullptr };
		std::shared_ptr<Residual<TInput, TCompute, TDevice>> residual_{ nullptr };

		Tensor<TCompute, MR> ln_1_output_;
		Tensor<TCompute, MR> fc_output_;
		Tensor<TCompute, MR> attn_output_;
		Tensor<TCompute, MR> ln_2_output_;
		Tensor<TCompute, MR> mlp_output_;
		Tensor<TCompute, MR> residual_output_;

		std::vector<size_t> validate_shape( const std::vector<size_t>& shape ) {
			if ( shape.size() != 3 ) {
				throw std::invalid_argument( "The input shape must have rank of 3." );
			}

			return shape;
		}
	};
}