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
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Dnn.Module;
import Dnn.Modules.LayerNorm;
import Dnn.Modules.Attention;
import Dnn.Modules.Residual;
import Dnn.Blocks.MLP;

namespace Mila::Dnn::Blocks
{
	using namespace Mila::Dnn;

	export
		template<typename TInput, typename TCompute = TInput, typename MR = Compute::CpuMemoryResource>
		requires ValidTensorTypes<TInput, TCompute>&& std::is_base_of_v<Compute::MemoryResource, MR>
	class TransformerBlock : public Module<TInput, TCompute, MR> {
	public:
		TransformerBlock( const std::vector<size_t>& input_shape, const size_t num_heads )
			: input_shape_{ validate_shape( input_shape ) }, num_heads_{ num_heads } {
			// The input shape is [batch_size, sequence_length, channels] from the previous layer. Initially form the encoder block.
			auto B = input_shape_[ 0 ];
			auto T = input_shape_[ 1 ];
			auto C = input_shape_[ 2 ];

			ln_1_ = std::make_unique<Modules::LayerNorm<TInput, TCompute, MR>>( "ln_1", input_shape_ );
			//attn_ = std::make_unique<Modules::Attention<TInput, TCompute, MR>>( "attn_1", input_shape_, num_heads_ );
			//ln_2_ = std::make_unique<Modules::LayerNorm<TInput,TCompute, MR>>( "ln_2", input_shape_ );
			//mlp_ = std::make_unique<MLP<TInput, TCompute, MR>>();
			//residual_ = std::make_unique < Modules::Residual<TInput, TCompute, MR>>( "res_1", input_shape_ );
		}

		Tensor<TCompute, MR>&& forward( const Tensor<TInput, MR>& input ) {
			Tensor<TCompute,MR>&& Y = ln_1_->forward( input );
			/*auto attn = attn_->forward( x );
			auto y = residual_->forward( attn );
			auto z = ln_2_->forward( y );
			auto output = mlp_->forward( z );*/
			
			return std::move( Y ); // output );
		}

		size_t parameters() const override {
			return 0;
		}

		std::string name() const override {
			return name_;
		}

		void print() const override {
			std::cout << "Module: " << name_ << std::endl;
			std::cout << "Parameters: " << parameters() << std::endl;
		}

	private:
		std::string name_; ///< The name of the module.
		std::vector<size_t> input_shape_; ///< The input shape.
		size_t num_heads_; ///< The number of attention heads.

		std::unique_ptr<Modules::LayerNorm<TInput, TCompute, MR>> ln_1_{ nullptr };
		std::unique_ptr<Modules::Attention<TInput, TCompute, MR>> attn_{ nullptr };
		std::unique_ptr<Modules::LayerNorm<TInput, TCompute, MR>> ln_2_{ nullptr };
		std::unique_ptr<MLP<TInput, TCompute, MR>> mlp_{ nullptr };
		std::unique_ptr<Modules::Residual<TInput, TCompute, MR>> residual_{ nullptr };

		std::vector<size_t> validate_shape( const std::vector<size_t>& shape ) {
			if ( shape.size() != 3 ) {
				throw std::invalid_argument( "The input shape must have 3 dimensions." );
			}

			return shape;
		}
	};
}