module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream> 
#include <stdexcept>
#include <iosfwd>

export module Dnn.Blocks.TransformerBlock;

import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.ComputeDevice;
import Compute.CpuDevice;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Dnn.Module;
import Dnn.Modules.LayerNorm;
import Dnn.Modules.FullyConnected;
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

            ln_1_ = std::make_shared<LayerNorm<TInput, TCompute, TDevice>>( this->getName() + ".ln_1", input_shape_ );
            fc_qkv_ = std::make_shared<FullyConnected<TInput, TCompute, TDevice>>( this->getName() + ".fc_qkv", C, 3 * C );
            attn_ = std::make_shared<MultiHeadAttention<TInput, TCompute, TDevice>>( this->getName() + ".attn", input_shape_, num_heads_ );
			fc_attn_proj_ = std::make_shared<FullyConnected<TInput, TCompute, TDevice>>( this->getName() + ".fc_attn_proj", C, C );
			res_1_ = std::make_shared<Residual<TInput, TCompute, TDevice>>( this->getName() + ".res_1" );
            ln_2_ = std::make_shared<LayerNorm<TInput, TCompute, TDevice>>( this->getName() + ".ln_2", input_shape_ );
            mlp_ = std::make_shared<MLP<TInput, TCompute, TDevice>>( this->getName() + ".mlp", input_shape_, 4 * C);
            res_2_ = std::make_shared<Residual<TInput, TCompute, TDevice>>( this->getName() + ".res_2" );

			this->addModule( ln_1_ );
			this->addModule( fc_qkv_ ); // qkv
			this->addModule( attn_ ); // attn
			this->addModule( fc_attn_proj_ ); // fc_proj
			this->addModule( res_1_ ); // residual
			this->addModule( ln_2_ );
			this->addModule( mlp_ );
			this->addModule( res_2_ );

			// Pre-allocate output tensors for the Transformer block layers
			ln_1_output_ = Tensor<TCompute, MR>( input_shape_ );
			fc_qkv_output_ = Tensor<TCompute, MR>( { B, T, 3 * C } );
			attn_output_ = Tensor<TCompute, MR>( input_shape_ );
			fc_attn_proj_output_ = Tensor<TCompute, MR>( { B, T, C } );
			res_1_output_ = Tensor<TCompute, MR>( input_shape_ );
			ln_2_output_ = Tensor<TCompute, MR>( input_shape_ );
			mlp_output_ = Tensor<TCompute, MR>( input_shape_ );
			res_2_output_ = Tensor<TCompute, MR>( input_shape_ );
		}

		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute,MR>& output ) {
			ln_1_->forward( input, ln_1_output_ );
			//std::cout << "ln1_output_" << std::endl;
			//ln1_output_.print();

			fc_qkv_->forward( ln_1_output_, fc_qkv_output_ );
			//std::cout << "fc1_output_" << std::endl;
			//fc1_output_.print();

			attn_->forward( fc_qkv_output_, attn_output_ );
			//std::cout << "attn_output_" << std::endl;
			//attn_output_.print();

			fc_attn_proj_->forward( attn_output_, fc_attn_proj_output_ );

			res_1_->forward( input, fc_attn_proj_output_, res_1_output_ );
			//std::cout << "residual_output_" << std::endl;
			//residual_output_.print();
			
			ln_2_->forward( res_1_output_, ln_2_output_ );
			//std::cout << "ln2_output_" << std::endl;
			//ln2_output_.print();

			mlp_->forward( ln_2_output_, mlp_output_ );
			//std::cout << "mlp_output_" << std::endl;
			//output.print();

			res_2_->forward( res_1_output_, mlp_output_, output );
		}

		size_t parameterCount() const override {
			size_t total_parameters = 0;
			for ( const auto& module : this->getSubModules() ) {
				total_parameters += module->parameterCount();
			}
			return total_parameters;
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

		std::string toString() const override {
			std::ostringstream oss;
			oss << "====================" << std::endl;
			oss << "Transformer: " << this->getName() << std::endl;
			oss << "Parameter count: " << parameterCount() << std::endl;
			oss << "Sub-Modules..." << std::endl;

			for ( const auto& module : this->getSubModules() ) {
				oss << *module;
			}

			return oss.str();
		}

	private:
		std::vector<size_t> input_shape_; ///< The input shape.
		size_t num_heads_; ///< The number of attention heads.

		std::shared_ptr<LayerNorm<TInput, TCompute, TDevice>> ln_1_{ nullptr };
		std::shared_ptr<FullyConnected<TInput, TCompute, TDevice>> fc_qkv_{ nullptr };
		std::shared_ptr<MultiHeadAttention<TInput, TCompute, TDevice>> attn_{ nullptr };
		std::shared_ptr<FullyConnected<TInput, TCompute, TDevice>> fc_attn_proj_{ nullptr };
		std::shared_ptr<Residual<TInput, TCompute, TDevice>> res_1_{ nullptr };
		std::shared_ptr<LayerNorm<TInput, TCompute, TDevice>> ln_2_{ nullptr };
		std::shared_ptr<MLP<TInput, TCompute, TDevice>> mlp_{ nullptr };
		std::shared_ptr<Residual<TInput, TCompute, TDevice>> res_2_{ nullptr };

		Tensor<TCompute, MR> ln_1_output_;
		Tensor<TCompute, MR> fc_qkv_output_;
		Tensor<TCompute, MR> attn_output_;
		Tensor<TCompute, MR> fc_attn_proj_output_;
		Tensor<TCompute, MR> res_1_output_;
		Tensor<TCompute, MR> ln_2_output_;
		Tensor<TCompute, MR> mlp_output_;
		Tensor<TCompute, MR> res_2_output_;

		std::vector<size_t> validate_shape( const std::vector<size_t>& shape ) {
			if ( shape.size() != 3 ) {
				throw std::invalid_argument( "The input shape must have rank of 3." );
			}

			return shape;
		}
	};
}