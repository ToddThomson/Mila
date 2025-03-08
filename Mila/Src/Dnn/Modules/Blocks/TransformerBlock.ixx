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

		TransformerBlock( const std::vector<size_t>& input_shape, const size_t num_heads )
			: input_shape_{ validate_shape( input_shape ) }, num_heads_{ num_heads } {
			// The input shape is [batch_size, sequence_length, channels] from the previous layer. Initially from the encoder block.
			auto B = input_shape_[ 0 ];
			auto T = input_shape_[ 1 ];
			auto C = input_shape_[ 2 ];

			ln1_ = std::make_unique<LayerNorm<TInput, TCompute, TDevice>>( "ln_1", input_shape_ );
			fc_1 = std::make_unique<Linear<TInput, TCompute, TDevice>>( "fc_1", C, 3 * C );
			attn_ = std::make_unique< Attention<TInput, TCompute, TDevice>>( "attn_1", input_shape_, num_heads_ );
			ln2_ = std::make_unique< LayerNorm<TInput, TCompute, TDevice>>( "ln_2", input_shape_ );
			mlp_ = std::make_unique< MLP<TInput, TCompute, TDevice>>( "mlp_1", input_shape_, 4 * C);
			residual_ = std::make_unique<Residual<TInput, TCompute, TDevice>>( "res_1" );

			// Pre-allocate output tensors for the Transformer block layers
			ln1_output_ = Tensor<TCompute, MR>( input_shape_ );

			fc1_output_ = Tensor<TCompute, MR>( { B, T, 3 * C } );
			attn_output_ = Tensor<TCompute, MR>( input_shape_ );
			ln2_output_ = Tensor<TCompute, MR>( input_shape_ );
			mlp_output_ = Tensor<TCompute, MR>( input_shape_ );
			residual_output_ = Tensor<TCompute, MR>( input_shape_ );
		}

		void forward( const Tensor<TInput, MR>& input, Tensor<TCompute,MR>& output ) override {
			ln1_->forward( input, ln1_output_ );
			std::cout << "ln1_output_" << std::endl;
			ln1_output_.print();

			fc_1->forward( ln1_output_, fc1_output_ );
			std::cout << "fc1_output_" << std::endl;
			fc1_output_.print();

			attn_->forward( fc1_output_, attn_output_ );
			std::cout << "attn_output_" << std::endl;
			attn_output_.print();

			residual_->forward( attn_output_, residual_output_ );
			std::cout << "residual_output_" << std::endl;
			residual_output_.print();
			
			ln2_->forward( residual_output_, ln2_output_ );
			std::cout << "ln2_output_" << std::endl;
			ln2_output_.print();

			mlp_->forward( ln2_output_, output );
			std::cout << "mlp_output_" << std::endl;
			output.print();
		}

		size_t parameters() const override {
			return 0;
		}

		std::string name() const override {
			return name_;
		}

		void save( mz_zip_archive& zip ) const override {
			// Save the state of the child modules
			for ( const auto& [name, module] : this->child_modules_ ) {
				module->save( zip );
			}

			// Save the state of the parameters
			for ( const auto& [name, tensor] : this->named_parameters_ ) {
				// Save tensor data to zip archive
			}
		}

		void load( mz_zip_archive& zip ) override {
			// Load the state of the child modules
			for ( const auto& [name, module] : this->child_modules_ ) {
				module->load( zip );
			}

			// Load the state of the parameters
			for ( const auto& [name, tensor] : this->named_parameters_ ) {
				// Load tensor data from zip archive
			}
		}

		void print() const override {
			std::cout << "Module: " << name_ << std::endl;
			std::cout << "Parameters: " << parameters() << std::endl;
		}

	private:
		std::string name_; ///< The name of the module.
		std::vector<size_t> input_shape_; ///< The input shape.
		size_t num_heads_; ///< The number of attention heads.

		std::unique_ptr<LayerNorm<TInput, TCompute, TDevice>> ln1_{ nullptr };
		std::unique_ptr<Linear<TInput, TCompute, TDevice>> fc_1{ nullptr };
		std::unique_ptr < Attention<TInput, TCompute, TDevice>> attn_{ nullptr };
		std::unique_ptr < LayerNorm<TInput, TCompute, TDevice>> ln2_{ nullptr };
		std::unique_ptr < MLP<TInput, TCompute, TDevice>> mlp_{ nullptr };
		std::unique_ptr < Residual<TInput, TCompute, TDevice>> residual_{ nullptr };

		Tensor<TCompute, MR> ln1_output_;
		Tensor<TCompute, MR> fc1_output_;
		Tensor<TCompute, MR> attn_output_;
		Tensor<TCompute, MR> ln2_output_;
		Tensor<TCompute, MR> mlp_output_;
		Tensor<TCompute, MR> residual_output_;
	

		std::vector<size_t> validate_shape( const std::vector<size_t>& shape ) {
			if ( shape.size() != 3 ) {
				throw std::invalid_argument( "The input shape must have 3 dimensions." );
			}

			return shape;
		}
	};
}