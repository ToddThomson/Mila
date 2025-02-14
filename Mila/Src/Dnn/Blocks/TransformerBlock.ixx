module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <type_traits>

export module Dnn.Blocks.TransformerBlock;

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
		TransformerBlock( ) {

			auto shape = std::vector<size_t>{ 4, 4, 4 };
			auto num_heads = 4;
			auto output_channels = 4;

			ln_1_ = std::make_unique<Modules::LayerNorm<TInput, TCompute, MR>>( "ln_1", shape );
			attn_ = std::make_unique < Modules::Attention<TInput, TCompute, MR>>( "attn_1", shape, num_heads );
			ln_2_ = std::make_unique < Modules::LayerNorm<TInput,TCompute, MR>>( "ln_2", shape );
			mlp_ = std::make_unique < MLP<TInput, TCompute, MR>>();
			residual_ = std::make_unique < Modules::Residual<TInput, TCompute, MR>>( "res_1", shape );
		}

		Tensor<TCompute, MR>&& forward( const Tensor<TInput, MR>& input ) {
			auto x = ln_1_->forward( input );
			auto attn = attn_->forward( x );
			auto y = residual_->forward( attn );
			auto z = ln_2_->forward( y );
			auto output = mlp_->forward( z );
			
			return std::move( output );
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

		std::unique_ptr<Modules::LayerNorm<TInput, TCompute, MR>> ln_1_{ nullptr };
		std::unique_ptr<Modules::Attention<TInput, TCompute, MR>> attn_{ nullptr };
		std::unique_ptr<Modules::LayerNorm<TInput, TCompute, MR>> ln_2_{ nullptr };
		std::unique_ptr<MLP<TInput, TCompute, MR>> mlp_{ nullptr };
		std::unique_ptr<Modules::Residual<TInput, TCompute, MR>> residual_{ nullptr };
	};
}